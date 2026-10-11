"""Walk-forward forecasts and gates based exclusively on matured OOS evidence."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from .evaluation import hac_mean, rank_ic

MODELS = ('single', 'equal', 'ic', 'ridge')


def evidence_gate(history, model, date, settings):
    v = history.loc[history.trained & history.label_end.le(date)].copy()
    dates = sorted(v.date.unique())[-settings.evidence_weeks:]
    v = v.loc[v.date.isin(dates)]
    if v.empty:
        return 0., np.nan, 0
    if model == 'risk':
        y = np.maximum(v.future_vol.to_numpy()**2, 1e-12)
        def loss(p):
            z = y/np.maximum(p**2, 1e-12)
            return z-np.log(z)-1
        gain = loss(v.prior_vol.to_numpy())-loss(v.pred_vol.to_numpy())
        calibration = 1.
    else:
        y = (v.target-v.prior)/v.scale
        a = (v[model]-v.prior)/v.scale
        gain = y*y-(y-a)**2
        calibration = float(np.clip(np.sum(y*a)/max(np.sum(a*a), 1e-12), 0, 1))
    series = pd.Series(np.asarray(gain), index=v.index).groupby(v.date).mean()
    mean, se, _ = hac_mean(series, settings.horizon_weeks)
    # Five predeclared gates (four return models and one volatility model).
    lower = mean-norm.ppf(1-settings.alpha/5)*se
    passed = len(series) >= settings.min_evidence_weeks and np.isfinite(lower) and lower > 0
    return calibration if passed else 0., lower, len(series)


def family_weights(names, definitions):
    families = [definitions[n].get('family', n) for n in names]
    return np.array([definitions[n].get('direction', 1)/families.count(f) / len(set(families))
                     for n, f in zip(names, families)])


def rolling_forecasts(panel, settings, definitions):
    names = list(settings.factors)
    features = ['z_'+n for n in names]
    result = panel.copy()
    result['trained'] = False
    result['prior'] = np.nan
    for m in MODELS:
        result[m] = np.nan
        result[m+'_strength'] = 0.
    result['pred_vol'] = np.nan
    result['risk_strength'] = 0.
    result['train_label_end'] = pd.NaT
    result['evidence_label_end'] = pd.NaT
    weights, gates = [], []
    fitted, count, last_refit = None, 0, -100000
    fixed = family_weights(names, definitions)
    single = np.zeros(len(names))
    single[names.index('momentum_63') if 'momentum_63' in names else 0] = 1.
    dated_ics = pd.DataFrame({date: [rank_ic(g[f], g.target/g.scale, settings.min_assets) for f in features]
                             for date, g in panel.groupby('date')}).T
    for date, current in result.groupby('date', sort=True):
        ix = current.index
        train = panel.loc[panel.label_end.le(date) & panel.target.notna()].copy()
        eligible_dates = sorted(train.date.unique())[-settings.train_weeks:]
        train = train.loc[train.date.isin(eligible_dates)]
        if len(eligible_dates) < settings.min_train_weeks:
            continue
        if fitted is None or count-last_refit >= settings.refit_weeks:
            X = train[features].to_numpy()
            y = (train.target/train.scale).to_numpy().clip(-10, 10)
            # Equal weight per date: a larger available universe is not more time evidence.
            sample_weight = 1/train.groupby('date').date.transform('size').to_numpy()
            sample_weight *= len(train)/sample_weight.sum()
            asset_prior = pd.Series(y, index=train.index).groupby(train.code).mean()
            n = train.groupby('code').size()
            global_prior = float(np.average(y, weights=sample_weight))
            asset_prior = (asset_prior*n+global_prior*52)/(n+52)
            # ICs and factor covariance are fit only on matured training rows.
            ic = dated_ics.reindex(eligible_dates).mean().to_numpy()
            ic = np.nan_to_num(ic)
            c = np.cov(X, rowvar=False)
            c = np.atleast_2d(c)
            c = .5*c+.5*np.eye(len(names))
            icw = np.linalg.solve(c+np.eye(len(names))*1e-8, ic)
            icw = icw/max(np.abs(icw).sum(), 1e-12)
            score_weights = {'single': single, 'equal': fixed, 'ic': icw}
            models = {m: Ridge(alpha=settings.ridge_alpha).fit((X@w)[:, None], y, sample_weight=sample_weight)
                      for m, w in score_weights.items()}
            models['ridge'] = Ridge(alpha=settings.ridge_alpha).fit(X, y, sample_weight=sample_weight)
            risk_x = np.c_[np.log(train.prior_vol), train.risk_vol_ratio]
            risk_y = np.log(train.future_vol.clip(lower=1e-6)/train.prior_vol).clip(-2, 2)
            risk_model = Ridge(alpha=settings.ridge_alpha).fit(risk_x, risk_y, sample_weight=sample_weight)
            fitted = models, score_weights, asset_prior, global_prior, risk_model, train.label_end.max()
            for m in MODELS:
                coef = models[m].coef_ if m == 'ridge' else score_weights[m]*models[m].coef_[0]
                for name, value in zip(names, coef):
                    weights.append({'date': date, 'model': m, 'factor': name, 'prediction_coefficient': value,
                                    'score_weight': score_weights[m][names.index(name)] if m != 'ridge' else np.nan,
                                    'intercept': models[m].intercept_, 'train_label_end': train.label_end.max()})
            last_refit = count
        count += 1
        models, score_weights, asset_prior, global_prior, risk_model, train_end = fitted
        X = current[features].to_numpy()
        result.loc[ix, 'trained'] = True
        result.loc[ix, 'prior'] = current.code.map(asset_prior).fillna(global_prior).to_numpy()*current.scale
        result.loc[ix, 'train_label_end'] = train_end
        result.loc[ix, 'pred_vol'] = current.prior_vol.to_numpy()*np.exp(risk_model.predict(
            np.c_[np.log(current.prior_vol), current.risk_vol_ratio]).clip(-1, 1))
        past = result.loc[result.date.lt(date)]
        matured = past.loc[past.trained & past.label_end.le(date)]
        evidence_end = matured.label_end.max() if not matured.empty else pd.NaT
        result.loc[ix, 'evidence_label_end'] = evidence_end
        for m in MODELS:
            x = X if m == 'ridge' else (X@score_weights[m])[:, None]
            result.loc[ix, m] = models[m].predict(x).clip(-3, 3)*current.scale
        for m in (*MODELS, 'risk'):
            strength, lower, weeks = evidence_gate(past, m, date, settings)
            result.loc[ix, m+'_strength'] = strength
            gates.append({'date': date, 'model': m, 'strength': strength, 'loss_gain_lower_bound': lower,
                          'matured_oos_weeks': weeks, 'train_label_end': train_end,
                          'evidence_label_end': evidence_end, 'status': 'ACCEPTED' if strength > 0 else 'PRIOR_ONLY'})
        if count % 52 == 0:
            print(f'Forecasts: {date.date()}, {count} OOS weeks', flush=True)
    return result, pd.DataFrame(weights), pd.DataFrame(gates)
