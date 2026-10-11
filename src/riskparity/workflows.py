"""Strategy registry and deterministic research workflow adapters."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

from .common import PipelineError, digest, engine, write_json
from .data import check_freshness, check_real, load_portfolio, metadata


def evaluate_goals(metrics, cfg):
    import math
    maximum = cfg['max_drawdown']
    required = maximum * cfg.get('excess_return_multiple', 2)
    excess, drawdown = metrics.get('annual_excess'), metrics.get('max_drawdown')
    reasons = []
    if excess is None or drawdown is None or not all(math.isfinite(x) for x in (excess, drawdown)):
        reasons.append('METRICS_INVALID')
    else:
        if drawdown > maximum:
            reasons.append('DRAWDOWN_EXCEEDED')
        if excess < required:
            reasons.append('EXCESS_RETURN_SHORTFALL')
    return {'goal_passed': not reasons, 'status': 'HISTORICAL_GOALS_PASSED' if not reasons else 'GOALS_FAILED',
            'reasons': reasons, 'max_drawdown_limit': maximum, 'required_annual_excess': required,
            'deployment_status': 'REQUIRES_PAPER_TRADING' if not reasons else 'NOT_VALIDATED',
            'research_only': True, 'validation': 'Previously inspected historical data; not untouched final holdout'}


def erc_weights(cov):
    """Same coordinate solver as archived three-asset pure-ERC diagnostic."""
    import numpy as np
    c = np.asarray(cov, dtype=float)
    c = c / max(np.mean(np.diag(c)), 1e-12)
    b = np.ones(len(c)) / len(c)
    x = np.sqrt(b / np.diag(c))
    for _ in range(10000):
        for i in range(len(x)):
            cross = float(c[i] @ x - c[i, i] * x[i])
            disc = float(np.sqrt(cross * cross + 4 * c[i, i] * b[i]))
            x[i] = 2 * b[i] / (disc + cross) if cross >= 0 else (disc - cross) / (2 * c[i, i])
        if np.max(np.abs(x * (c @ x) - b)) < 1e-11:
            break
    else:
        raise PipelineError('SOLVER_FAILED', 'ERC coordinate solver did not converge')
    w = x / x.sum()
    rc = w * (c @ w)
    rc /= rc.sum()
    if np.max(np.abs(rc - b)) > 1e-6:
        raise PipelineError('SOLVER_FAILED', 'ERC risk contribution residual')
    return w, rc


def run_portfolio(cfg, out):
    import numpy as np
    import pandas as pd
    from sklearn.covariance import LedoitWolf
    engine()
    import rotation as old
    import rotation_v2 as v
    prices, cash, assets, audit, price_meta, cash_meta = load_portfolio(cfg)
    audit.to_csv(out / 'cash_audit.csv', index=False)
    params = dict(cfg.get('parameters', {}))
    strategy = cfg['strategy']
    d = cfg['max_drawdown']
    if strategy == 'erc':
        allowed = {'fee_bps', 'slippage_bps'}
        if set(params) - allowed:
            raise PipelineError('CONFIG_INVALID', 'ERC parameters support fee_bps and slippage_bps only')
        codes = cfg.get('erc_assets') or (['510300.SH', 'CBA00101.CS', '518880.SH'] if cfg['market'] == 'CN'
                                        else ['SPY.P', 'IEF.O', 'GLD.P'])
        if not codes or len(codes) != len(set(codes)) or set(codes) - set(assets):
            raise PipelineError('CONFIG_INVALID', 'ERC assets must be unique members of the universe')
        diagnostics = []
        def allocate(history, current, *unused):
            r = history[codes].pct_change(fill_method=None).iloc[-252:]
            if not np.isfinite(r.to_numpy()).all():
                raise PipelineError('DATA_INVALID', 'ERC selected assets need complete 252-session return history')
            cov = LedoitWolf().fit(r.to_numpy()).covariance_ + np.eye(len(codes)) * 1e-12
            w, rc = erc_weights(cov)
            target = pd.Series(0., index=history.columns)
            target.loc[codes] = w
            diagnostics.append({'date': history.index[-1], **dict(zip(codes, rc))})
            return target, {'status': 'equal_risk_contribution'}
        previous = old.allocate
        settings = old.Settings(d, method='hrp', frequency='weekly', overlay_mode='none', **params)
        try:
            old.allocate = allocate
            result = old.backtest(prices.iloc[252:], assets, {}, settings, cash_returns=cash)
        finally:
            old.allocate = previous
        names = ['curve', 'holdings', 'signals']
        pd.DataFrame(diagnostics).to_csv(out / 'risk_contributions.csv', index=False)
        stats = v.evaluate(result[0], v.Model(d))
    elif strategy == 'flat':
        import rotation_flat as f
        settings = f.FlatModel(max_drawdown=d, excess_return_multiple=cfg.get('excess_return_multiple', 2), **params)
        result = f.backtest(prices, cash, assets, settings, progress=True)
        names = ['curve', 'holdings', 'signals', 'risk_contributions', 'dedup']
        stats = f.evaluate(result[0], settings)
    elif strategy in {'nested_target', 'protective_nested'}:
        settings = v.Model(max_drawdown=d, method=strategy, frequency='weekly', risk_check_frequency='weekly',
                           excess_return_multiple=cfg.get('excess_return_multiple', 2), **params)
        print('Building causal class sleeves', flush=True)
        sleeves = v.build_sleeves(prices, cash, assets, settings)
        print('Backtesting weekly portfolio', flush=True)
        result = v.backtest_v2(prices, cash, assets, settings, sleeves=sleeves)
        names = ['curve', 'holdings', 'signals', 'events']
        stats = v.evaluate(result[0], settings)
    else:
        import forecast_adaptive as f
        import rotation_adaptive as a
        fs = f.ForecastSettings(**cfg.get('forecast_parameters', {}))
        print('Building causal forecasts', flush=True)
        forecasts = f.prequential_forecasts(f.build_panel(prices, cash, fs), fs, progress=True)
        if forecasts.empty:
            raise PipelineError('INSUFFICIENT_HISTORY', 'No prequential forecasts; supply a longer interval')
        forecasts.to_csv(out / 'forecasts.csv', index=False)
        f.forecast_metrics(forecasts).to_csv(out / 'forecast_metrics.csv', index=False)
        write_json(out / 'forecast_settings.json', asdict(fs))
        settings = a.AdaptiveSettings(max_drawdown=d, **params)
        if 'gamma0' not in params:
            selected, calibration = a.calibrate_gamma(prices, cash, assets, forecasts, settings)
            calibration.to_csv(out / 'gamma_calibration.csv', index=False)
            if not (calibration.max_drawdown <= d).any():
                raise PipelineError('CALIBRATION_FAILED', 'No gamma meets initial calibration drawdown; refusing least-bad promotion')
            settings = replace(settings, gamma0=selected)
        result = a.backtest(prices, cash, assets, forecasts, settings, progress=True)
        names = ['curve', 'holdings', 'signals', 'risk_contributions', 'dedup', 'covariance_scores']
        stats = a.evaluate(result[0], settings)
    for name, frame in zip(names, result):
        frame.to_csv(out / (name + '.csv'), index=name not in {'dedup', 'covariance_scores'})
    assessment = evaluate_goals(stats, cfg)
    stats['goal_passed'] = assessment['goal_passed']
    stats['required_annual_excess'] = assessment['required_annual_excess']
    write_json(out / 'metrics.json', stats)
    write_json(out / 'assessment.json', assessment)
    write_json(out / 'settings.json', asdict(settings))
    latest = result[2].iloc[-1].to_dict() if len(result[2]) else {}
    write_json(out / 'latest_signal.json', {
        'signal_date': str(result[2].index[-1].date()) if len(result[2]) else None,
        'data_asof': str(prices.index[-1].date()), 'target': latest,
        'last_executed_signal_date': str(result[0].executed_signal_date.iloc[-1]),
        'actual_holdings_asof': result[1].iloc[-1].to_dict(),
        'research_only': True, 'deployment_status': assessment['deployment_status'],
        'execution': 't close signal, t+1 close execution; new returns start t+2; target is not an order'})
    return {'data_start': str(prices.index[0].date()), 'data_end': str(prices.index[-1].date()),
            'price_meta': price_meta, 'cash_meta': cash_meta, 'evaluation_start': str(result[0].index[0].date()),
            'execution': 'weekly first trading day close signal, next close fill, following-session new return'}


def run_single(cfg, out):
    engine('single')
    import model
    meta = metadata(cfg['prices'])
    check_real(meta)
    if meta['source'] not in {'Wind WSD', 'AKShare Eastmoney', 'AKShare Sina Futures', 'Yahoo Finance'}:
        raise PipelineError('PROVENANCE_INVALID', 'Unverified OHLCV source')
    if meta.get('symbol') and meta['symbol'] != cfg['symbol']:
        raise PipelineError('PROVENANCE_INVALID', 'OHLCV symbol mismatch')
    d = model.read_data(cfg['prices'])
    if cfg.get('start'):
        d = d.loc[cfg['start']:]
    if cfg.get('end'):
        d = d.loc[:cfg['end']]
    if d.empty:
        raise PipelineError('DATA_INVALID', 'Empty OHLCV interval')
    check_freshness(d.index[-1], cfg)
    settings = model.Config(**cfg.get('parameters', {}))
    stats, latest = model.run(d, settings, out, cfg['symbol'])
    write_json(out / 'assessment.json', {'status': latest['deployment_status'], 'research_only': True,
                                       'deployment_status': latest['deployment_status'],
                                       'reasons': latest.get('validation_gaps', [])})
    write_json(out / 'provenance.json', {'input': cfg['prices'], 'sha256': digest(cfg['prices']),
                                       'model_sha256': digest(model.__file__), 'data_source': meta})
    return {'data_start': str(d.index[0].date()), 'data_end': str(d.index[-1].date()), 'price_meta': meta,
            'execution': 'signal at close, next open entry; OHLC barrier simulation, stop first on ambiguous bars'}


def code_receipt(snapshot=None):
    import shutil
    paths = {'riskparity/' + p.name: p for p in Path(__file__).parent.glob('*.py')}
    core = ['__init__', 'rotation', 'cash_rates', 'rotation_v2', 'rotation_flat',
            'forecast_adaptive', 'rotation_adaptive', 'wind_fetch', 'setup_wind']
    for which, modules in [('rotation', core), ('single', ['__init__', 'model', 'fetch'])]:
        directory = engine(which)
        paths.update({'engine_' + which + '/' + name + '.py': directory / (name + '.py') for name in modules})
    hashes = {name: digest(p) for name, p in sorted(paths.items())}
    if snapshot is not None:
        for name, p in paths.items():
            target = Path(snapshot) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            if digest(target) != hashes[name]:
                raise PipelineError('CODE_CHANGED', 'Runtime code changed while creating snapshot')
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'sha256': combined, 'files': hashes}
