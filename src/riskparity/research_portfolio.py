"""Causal risk-budget portfolios, execution accounting and strategy allocation."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .common import PipelineError

BASE_STRATEGIES = ('equal_weight', 'risk_parity', 'single_factor_rp', 'equal_factor_rp',
                   'ic_factor_rp', 'ridge_factor_rp', 'ridge_ungated_rp', 'ridge_risk_forecast_rp')
STRATEGIES = (*BASE_STRATEGIES, 'meta_equal', 'meta_risk_parity')


def risk_budget_weights(covariance, budget):
    c, b = np.asarray(covariance, float), np.asarray(budget, float)
    if c.shape != (len(b), len(b)) or not np.isfinite(c).all() or not np.isfinite(b).all() or (b <= 0).any():
        raise PipelineError('RISK_BUDGET_INVALID', 'Finite covariance and strictly positive budgets required')
    if not np.allclose(c, c.T) or np.linalg.eigvalsh(c).min() <= 0:
        raise PipelineError('RISK_COVARIANCE_INVALID', 'Positive definite symmetric covariance required')
    c = c/max(np.diag(c).mean(), 1e-12)
    b = b/b.sum()
    x = np.sqrt(b/np.diag(c))
    for _ in range(10000):
        for i in range(len(b)):
            cross = c[i]@x-c[i, i]*x[i]
            d = np.sqrt(cross*cross+4*c[i, i]*b[i])
            x[i] = 2*b[i]/(d+cross) if cross >= 0 else (d-cross)/(2*c[i, i])
        if np.max(np.abs(x*(c@x)-b)) < 1e-11:
            break
    else:
        raise PipelineError('SOLVER_FAILED', 'Risk budget solver did not converge')
    w = x/x.sum()
    rc = w*(c@w)/(w@c@w)
    if np.max(np.abs(rc-b)) > 1e-6:
        raise PipelineError('SOLVER_FAILED', 'Risk contribution residual')
    return w, rc


def forecast_budget(frame, model, settings, gated=True):
    strength = frame[model+'_strength'].to_numpy() if gated else np.ones(len(frame))
    alpha = (frame[model]-frame.prior).to_numpy()*strength
    score = (alpha/frame.scale.to_numpy()).clip(-2, 2)*settings.tilt_strength
    budget = np.exp(score-score.max())
    return budget/budget.sum()


def execute_trade(weights, target, cost_rate):
    """Solve fees on executed notional; target is a fraction of post-fee NAV."""
    fee = cost_rate*np.abs(target-weights).sum()
    for _ in range(100):
        revised = cost_rate*np.abs((1-fee)*target-weights).sum()
        if abs(revised-fee) < 1e-14:
            fee = revised
            break
        fee = revised
    turnover = np.abs((1-fee)*target-weights).sum()
    return fee, turnover


def account_day(weights, returns, cash_return, target, cost_rate):
    contributions = weights*returns
    cash_part = (1-weights.sum())*cash_return
    growth = 1+contributions.sum()+cash_part
    if growth <= 0:
        raise PipelineError('NAV_INVALID', 'Nonpositive account wealth')
    drift = weights*(1+returns)/growth
    fee, turnover = execute_trade(drift, target, cost_rate) if target is not None else (0., 0.)
    result = growth*(1-fee)-1
    final = target.copy() if target is not None else drift
    return final, result, turnover, -growth*fee, cash_part, contributions


def bootstrap_paths(history, settings, seed):
    """Synchronized circular 5-session blocks preserve cross-asset dependence."""
    rng = np.random.default_rng(seed)
    horizon = settings.risk_horizon_sessions
    starts = rng.integers(len(history), size=(settings.risk_paths, int(np.ceil(horizon/5))))
    indices = ((starts[:, :, None]+np.arange(5)) % len(history)).reshape(settings.risk_paths, -1)[:, :horizon]
    return np.cumprod(1+history[indices], axis=1)


def exposure_overlay(base, current, nav, peak, covariance, paths, cash_rate, drawdown, settings, cost_rate):
    """Prospective drawdown uses the account's original peak, with one-day latency."""
    volatility = float(np.sqrt(max(base@covariance@base, 0)))
    cap = min(1., settings.volatility_target/max(volatility, 1e-12))
    grid = np.linspace(0, cap, settings.exposure_grid)
    cash_path = np.power(1+cash_rate, np.arange(1, paths.shape[1]+1))
    # First sampled day earns current holdings; new holdings start the second day.
    latency = paths[:, 0, :]@current+(1-current.sum())*(1+cash_rate)
    after = paths/paths[:, :1, :]
    after_cash = cash_path/cash_path[0]
    basket = np.einsum('pha,a->ph', after, base)
    candidate = after_cash[None, :, None]+(basket-after_cash[None, :]*base.sum())[:, :, None]*grid
    fees = np.array([execute_trade(current, base*g, cost_rate)[0] for g in grid])
    wealth = nav*latency[:, None, None]*candidate*(1-fees)
    future_peak = np.maximum(peak, np.maximum.accumulate(wealth, axis=1))
    maxdd = (1-wealth/future_peak).max(axis=1)
    quantiles = np.quantile(maxdd, settings.risk_confidence, axis=0)
    feasible = np.flatnonzero(quantiles <= drawdown)
    chosen = int(feasible[-1]) if len(feasible) else 0
    return base*grid[chosen], {'exposure_multiplier': grid[chosen], 'predicted_drawdown': quantiles[chosen],
                              'risk_feasible': bool(len(feasible)), 'predicted_volatility': volatility*grid[chosen]}


def simulate(prices, cash, forecasts, settings, drawdown, cost_multiplier=1.):
    if set(settings.strategy_components)-set(BASE_STRATEGIES) or len(settings.strategy_components) < 2:
        raise PipelineError('CONFIG_INVALID', 'Choose at least two existing base strategy components')
    if len(set(settings.strategy_components)) != len(settings.strategy_components):
        raise PipelineError('CONFIG_INVALID', 'Duplicate strategy components')
    dates = {d: g.set_index('code') for d, g in forecasts.loc[forecasts.trained].groupby('date')}
    if not dates:
        raise PipelineError('INSUFFICIENT_HISTORY', 'No mature rolling training window')
    returns = prices.pct_change(fill_method=None)
    start = prices.index.get_loc(min(dates))
    n = len(prices.columns)
    codes = list(prices.columns)
    states = {name: {'weights': np.zeros(n), 'nav': 1., 'peak': 1., 'pending': None} for name in STRATEGIES}
    curves = {name: [] for name in STRATEGIES}
    holdings, signals, allocation, strategies, attribution = [], [], [], [], []
    fee_rate = (settings.fee_bps+settings.slippage_bps)*1e-4*cost_multiplier
    for pos in range(start, len(prices)):
        date = prices.index[pos]
        r = returns.iloc[pos].fillna(0).to_numpy()
        c = float(cash.iloc[pos])
        if pos > start:
            for name, state in states.items():
                prior_nav = state['nav']
                w, ret, turnover, cost, cash_part, parts = account_day(state['weights'], r, c, state['pending'], fee_rate)
                state.update(weights=w, nav=prior_nav*(1+ret), pending=None)
                state['peak'] = max(state['peak'], state['nav'])
                curves[name].append({'date': date, 'nav': state['nav'], 'return': ret, 'cash_return': c,
                                     'exposure': w.sum(), 'turnover': turnover, 'cost_contribution': cost,
                                     'drawdown': 1-state['nav']/state['peak']})
                attribution.extend({'date': date, 'strategy': name, 'component': code, 'return_contribution': part,
                                    'wealth_contribution': prior_nav*part} for code, part in zip(codes+['CASH', 'COST'], [*parts, cash_part, cost]))
        if date not in dates or pos+1 >= len(prices):
            continue
        frame = dates[date]
        hist = returns.iloc[pos-settings.covariance_sessions+1:pos+1]
        eligible = hist.columns[hist.notna().all() & prices.iloc[pos].notna()]
        eligible = [code for code in eligible if code in frame.index]
        if len(eligible) < settings.min_assets:
            continue
        loc = np.array([codes.index(code) for code in eligible])
        frame = frame.loc[eligible]
        sample = hist[eligible].to_numpy()
        cov = LedoitWolf().fit(sample).covariance_*252+np.eye(len(loc))*1e-10
        paths = bootstrap_paths(sample, settings, settings.seed+pos)
        targets = {}
        for name in BASE_STRATEGIES:
            b = np.ones(len(loc))/len(loc)
            risk_cov = cov.copy()
            if name not in ('risk_parity', 'equal_weight'):
                model = name.split('_')[0]
                b = forecast_budget(frame, model, settings, gated=name != 'ridge_ungated_rp')
            if name == 'ridge_risk_forecast_rp':
                ratio = frame.pred_vol.to_numpy()/np.sqrt(np.diag(cov))
                scale = 1+frame.risk_strength.to_numpy()*(ratio-1)
                risk_cov *= np.outer(scale, scale)
            w, rc = risk_budget_weights(risk_cov, b)
            if name == 'equal_weight':
                w = np.ones(len(loc))/len(loc)
                rc = w*(risk_cov@w)/(w@risk_cov@w)
            state = states[name]
            # Include current positions excluded from eligibility only via conservative liquidation cost.
            if np.any(state['weights'][np.setdiff1d(np.arange(n), loc)] > 1e-10):
                raise PipelineError('UNIVERSE_EXIT_UNSUPPORTED', 'Held asset lacks complete history; explicit liquidation policy required')
            target, info = exposure_overlay(w, state['weights'][loc], state['nav'], state['peak'], risk_cov,
                                            paths, c, drawdown, settings, fee_rate)
            full = np.zeros(n)
            full[loc] = target
            targets[name] = full
            state['pending'] = full
            signals.append({'date': date, 'fill_date': prices.index[pos+1], 'strategy': name, **info})
            historical_rc = w*(cov@w)/(w@cov@w)
            for i, code in enumerate(eligible):
                allocation.append({'date': date, 'strategy': name, 'code': code, 'risk_budget': b[i],
                                   'model_risk_contribution': rc[i], 'historical_risk_contribution': historical_rc[i],
                                   'capital_weight_before_overlay': w[i], 'target_weight': target[i]})
        components = list(settings.strategy_components)
        history = pd.DataFrame({name: pd.Series({row['date']: row['return'] for row in curves[name][-252:]}) for name in components})
        enough = len(history) >= 63
        scov = LedoitWolf().fit(history.to_numpy()).covariance_*252+np.eye(len(components))*1e-10 if enough else np.eye(len(components))
        for name in ('meta_equal', 'meta_risk_parity'):
            sw = np.ones(len(components))/len(components) if name == 'meta_equal' or not enough else risk_budget_weights(scov, np.ones(len(components)))[0]
            src = sw*(scov@sw)/(sw@scov@sw)
            aggregate = sum(weight*targets[component] for component, weight in zip(components, sw))
            state = states[name]
            target, info = exposure_overlay(aggregate[loc], state['weights'][loc], state['nav'], state['peak'], cov,
                                            paths, c, drawdown, settings, fee_rate)
            full = np.zeros(n)
            full[loc] = target
            state['pending'] = full
            signals.append({'date': date, 'fill_date': prices.index[pos+1], 'strategy': name, **info})
            diversification = float(sw@np.sqrt(np.diag(scov))/np.sqrt(sw@scov@sw))
            for component, weight, risk in zip(components, sw, src):
                strategies.append({'date': date, 'strategy': name, 'component': component, 'weight': weight,
                                   'risk_contribution': risk, 'diversification_ratio': diversification,
                                   'history_sessions': len(history), 'allocation_status': 'ESTIMATED' if enough else 'EQUAL_WARMUP'})
        for name, state in states.items():
            holdings.extend({'signal_date': date, 'fill_date': prices.index[pos+1], 'strategy': name,
                             'code': code, 'target_weight': weight} for code, weight in zip(codes+['CASH'], [*state['pending'], 1-state['pending'].sum()]))
    return ({name: pd.DataFrame(rows).set_index('date') for name, rows in curves.items()},
            pd.DataFrame(holdings), pd.DataFrame(signals), pd.DataFrame(allocation), pd.DataFrame(strategies), pd.DataFrame(attribution))
