"""Separate factor, forecast, portfolio and allocation evidence; no opaque total score."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def hac_mean(values, lag=1):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max(10, 4*(lag+1)):
        return (float(x.mean()) if n else np.nan), np.nan, np.nan
    z = x-x.mean()
    v = z@z/n
    for k in range(1, min(lag, n-1)+1):
        v += 2*(1-k/(lag+1))*(z[k:]@z[:-k]/n)
    se = np.sqrt(max(0., v)/n)
    p = 2*norm.sf(abs(x.mean()/se)) if se > 0 else np.nan
    return float(x.mean()), float(se), float(p)


def bh_fdr(values):
    p = pd.Series(values, dtype=float)
    valid = p.dropna().sort_values()
    q = valid.to_numpy()*len(valid)/np.arange(1, len(valid)+1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    result = pd.Series(np.nan, index=p.index)
    result.loc[valid.index] = q
    return result


def rank_ic(x, y, minimum=5):
    v = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(v) < minimum or min(v.x.nunique(), v.y.nunique()) < 2:
        return np.nan
    return v.x.rank().corr(v.y.rank())


def block_interval(values, block=20, draws=512, seed=8108, alpha=.05):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < max(30, 3*block):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = rng.integers(n, size=(draws, int(np.ceil(n/block))))
    indices = ((starts[:, :, None]+np.arange(block)) % n).reshape(draws, -1)[:, :n]
    return tuple(np.quantile(x[indices].mean(axis=1), [alpha/2, 1-alpha/2]))


def factor_evaluation(panel, names, settings, classes=None):
    """Overlapping forward group means are diagnostics, never a compounded NAV."""
    rows, daily, groups, redundancy = [], [], [], []
    h = int(panel.horizon_weeks.iloc[0])
    for name in names:
        records, previous_top, autocorrelation = [], None, []
        previous_ranks = None
        for date, g in panel.groupby('date', sort=True):
            v = g.set_index('code')[[name, 'target', 'scale']].dropna()
            ic = rank_ic(v[name], v.target, settings.min_assets)
            normalized_ic = rank_ic(v[name], v.target/v.scale, settings.min_assets)
            if np.isnan(ic):
                continue
            # Ranking ties stay together, avoiding arbitrary asset-code tie breaks.
            percentile = v[name].rank(method='average', pct=True)
            quantile = np.minimum(np.ceil(percentile*settings.quantiles), settings.quantiles).astype(int)
            qret = v.target.groupby(quantile).mean()
            top = set(v.index[quantile == settings.quantiles])
            turnover = 1-len(top & previous_top)/len(top) if previous_top is not None and top else np.nan
            ranks = v[name].rank()
            auto = rank_ic(ranks, previous_ranks, settings.min_assets) if previous_ranks is not None else np.nan
            previous_top, previous_ranks = top, ranks
            within = []
            if classes:
                for group in set(classes.values()):
                    ix = [c for c in v.index if classes.get(c) == group]
                    z = rank_ic(v.loc[ix, name], v.loc[ix, 'target'], settings.min_assets)
                    if np.isfinite(z):
                        within.append(z)
            record = {'date': date, 'factor': name, 'horizon_weeks': h, 'rank_ic': ic,
                      'risk_normalized_rank_ic': normalized_ic, 'within_class_rank_ic': np.mean(within) if within else np.nan,
                      'top_quantile_turnover': turnover, 'rank_autocorrelation': auto,
                      'high_minus_low_forward_return': qret.get(settings.quantiles, np.nan)-qret.get(1, np.nan)}
            records.append(record)
            daily.append(record)
            for q, value in qret.items():
                groups.append({'date': date, 'factor': name, 'horizon_weeks': h, 'quantile': q, 'forward_mean_return': value})
        frame = pd.DataFrame(records)
        if frame.empty:
            rows.append({'factor': name, 'horizon_weeks': h, 'status': 'INSUFFICIENT_HISTORY', 'weeks': 0})
            continue
        mean, se, p = hac_mean(frame.rank_ic, h)
        anchors = panel[['date', 'signal_week']].drop_duplicates().set_index('date').signal_week
        disjoint = frame.loc[frame.date.map(anchors).mod(h).eq(0), 'rank_ic']
        qmeans = pd.DataFrame(groups).query('factor == @name').groupby('quantile').forward_mean_return.mean()
        rows.append({'factor': name, 'horizon_weeks': h, 'weeks': len(frame), 'coverage': panel[name].notna().mean(),
                     'rank_ic_mean': mean, 'rank_ic_ir': mean/frame.rank_ic.std() if frame.rank_ic.std() > 0 else np.nan,
                     'ic_hac_se': se, 'ic_p_value': p, 'positive_ic_fraction': frame.rank_ic.gt(0).mean(),
                     'nonoverlap_ic': disjoint.mean(), 'nonoverlap_weeks': len(disjoint),
                     'risk_normalized_ic': frame.risk_normalized_rank_ic.mean(),
                     'within_class_ic': frame.within_class_rank_ic.mean(),
                     'group_monotonicity': rank_ic(pd.Series(qmeans.index), pd.Series(qmeans.to_numpy()), minimum=3),
                     'mean_forward_spread': frame.high_minus_low_forward_return.mean(),
                     'top_quantile_turnover': frame.top_quantile_turnover.mean(),
                     'rank_autocorrelation': frame.rank_autocorrelation.mean(), 'status': 'EXPLORATORY'})
    for i, a in enumerate(names):
        for b in names[i+1:]:
            correlations = [rank_ic(g[a], g[b], settings.min_assets) for _, g in panel.groupby('date')]
            redundancy.append({'factor_a': a, 'factor_b': b, 'horizon_weeks': h,
                               'mean_rank_correlation': np.nanmean(correlations),
                               'mean_absolute_rank_correlation': np.nanmean(np.abs(correlations))})
    return pd.DataFrame(rows), pd.DataFrame(daily), pd.DataFrame(groups), pd.DataFrame(redundancy)


def forecast_evaluation(frame, models, settings):
    rows, calibration = [], []
    mature = frame.loc[frame.trained & frame.target.notna()]
    for model in models:
        v = mature.dropna(subset=[model, 'prior', 'scale'])
        if v.empty:
            rows.append({'model': model, 'status': 'INSUFFICIENT_HISTORY'})
            continue
        y, p, prior = (v[k].to_numpy() for k in ('target', model, 'prior'))
        gain = (((y-prior)/v.scale.to_numpy())**2-((y-p)/v.scale.to_numpy())**2)
        dated_gain = pd.Series(gain, index=v.index).groupby(v.date).mean()
        low, high = block_interval(dated_gain, settings.horizon_weeks, settings.bootstrap_draws, settings.seed)
        slope = np.polyfit(p, y, 1) if np.std(p) > 1e-10 else [np.nan, np.nan]
        ics = [rank_ic(g[model], g.target, settings.min_assets) for _, g in v.groupby('date')]
        positive = y > 0
        balanced = np.mean([np.mean((p[k] > 0) == positive[k]) for k in (positive, ~positive) if k.any()])
        for q in range(5):
            # Descriptive OOS forecast bins; never fed back into earlier predictions.
            bins = pd.qcut(v[model].rank(method='average'), 5, labels=False, duplicates='drop')
            group = v.loc[bins.eq(q)]
            calibration.append({'model': model, 'bin': q+1, 'count': len(group),
                                'mean_predicted_return': group[model].mean(), 'mean_realized_return': group.target.mean()})
        rows.append({'model': model, 'rows': len(v), 'weeks': v.date.nunique(),
                     'r2_vs_prior': 1-np.mean((y-p)**2)/max(np.mean((y-prior)**2), 1e-16),
                     'normalized_r2_vs_prior': gain.sum()/max(np.sum(((y-prior)/v.scale.to_numpy())**2), 1e-16),
                     'rmse': np.sqrt(np.mean((y-p)**2)), 'mae': np.mean(np.abs(y-p)),
                     'mean_rank_ic': np.nanmean(ics), 'balanced_direction_accuracy': balanced,
                     'calibration_slope': slope[0], 'calibration_intercept': slope[1],
                     'normalized_loss_gain_low': low, 'normalized_loss_gain_high': high,
                     'gate_acceptance_fraction': v.groupby('date')[model+'_strength'].first().gt(0).mean(),
                     'status': 'EXPLORATORY_PREQUENTIAL'})
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def risk_forecast_evaluation(frame):
    v = frame.loc[frame.trained].dropna(subset=['pred_vol', 'future_vol', 'prior_vol'])
    if v.empty:
        return {'status': 'INSUFFICIENT_HISTORY'}
    y, p, base = (v[k].to_numpy()**2 for k in ('future_vol', 'pred_vol', 'prior_vol'))
    def qlike(pred):
        ratio = np.maximum(y, 1e-12)/np.maximum(pred, 1e-12)
        return ratio-np.log(ratio)-1
    return {'status': 'EXPLORATORY_PREQUENTIAL', 'rows': len(v),
            'qlike_predicted': np.mean(qlike(p)), 'qlike_historical': np.mean(qlike(base)),
            'vol_rmse_predicted': np.sqrt(np.mean((np.sqrt(y)-np.sqrt(p))**2)),
            'vol_rmse_historical': np.sqrt(np.mean((np.sqrt(y)-np.sqrt(base))**2)),
            'gate_acceptance_fraction': v.groupby('date').risk_strength.first().gt(0).mean()}


def portfolio_metrics(curve, cash, benchmark=None):
    r = curve['return'].astype(float)
    cash = cash.reindex(r.index)
    wealth = pd.Series(np.r_[1., (1+r).cumprod().to_numpy()])
    dd = 1-wealth/wealth.cummax()
    vol = r.std(ddof=1)*np.sqrt(252)
    excess = r-cash
    cagr = np.prod(1+r)**(252/len(r))-1
    cash_cagr = np.prod(1+cash)**(252/len(r))-1
    underwater = dd.gt(1e-12)
    durations = underwater.groupby((~underwater).cumsum()).cumsum()
    downside = np.sqrt(np.mean(np.minimum(excess, 0)**2))*np.sqrt(252)
    es = -r.loc[r <= r.quantile(.05)].mean()
    result = {'sessions': len(r), 'annual_return': cagr, 'cash_cagr': cash_cagr, 'annual_excess': cagr-cash_cagr,
              'annual_volatility': vol, 'sharpe_cash_benchmark': excess.mean()*252/vol if vol > 0 else None,
              'sortino_cash': excess.mean()*252/downside if downside > 0 else None,
              'max_drawdown': dd.max(), 'calmar': cagr/dd.max() if dd.max() > 0 else None,
              'daily_expected_shortfall95': es, 'longest_drawdown_sessions': int(durations.max()),
              'underwater_fraction': underwater.mean(), 'mean_exposure': curve.exposure.mean(),
              'annual_turnover': curve.turnover.mean()*252, 'annual_cost_drag_approx': curve.cost_contribution.sum()*252/len(r)}
    if benchmark is not None:
        b = benchmark.reindex(r.index)
        active = r-b
        tracking = active.std()*np.sqrt(252)
        beta = np.cov(r, b, ddof=1)[0, 1]/b.var() if b.var() > 1e-16 else np.nan
        result.update(tracking_error=tracking, information_ratio=active.mean()*252/tracking if tracking > 0 else None,
                      benchmark_beta=beta, benchmark_alpha_annual=(r.mean()-beta*b.mean())*252)
    return result


def incremental_evaluation(curves, baseline, settings):
    base = curves[baseline]
    rows = []
    for name, curve in curves.items():
        active = curve['return']-base['return']
        low, high = block_interval(active, settings.bootstrap_block_sessions, settings.bootstrap_draws, settings.seed)
        rows.append({'strategy': name, 'baseline': baseline, 'mean_daily_increment': active.mean(),
                     'annual_mean_increment': active.mean()*252, 'paired_increment_low': low*252,
                     'paired_increment_high': high*252,
                     'incremental_evidence': 'POSITIVE_EXPLORATORY' if low > 0 else 'NOT_ESTABLISHED'})
    return pd.DataFrame(rows)
