"""Validated snapshots and whole-panel Wind-first provider failover."""
from pathlib import Path
from types import SimpleNamespace
import datetime as dt
import io
import json
import urllib.request
import urllib.parse

from .common import PipelineError, engine, now, write_json


def metadata(path):
    receipt = Path(path).with_suffix('.meta.json')
    if not receipt.exists():
        raise PipelineError('PROVENANCE_MISSING', str(receipt))
    return json.loads(receipt.read_text(encoding='utf-8'))


def check_real(meta):
    if meta.get('synthetic') is not False or not meta.get('source'):
        raise PipelineError('PROVENANCE_INVALID', 'Real prices require source and synthetic=false')


def load_portfolio(cfg):
    import numpy as np
    import pandas as pd
    engine()
    from rotation import read_prices
    from rotation_v2 import lagged_cash
    universe = json.loads(Path(cfg['universe']).read_text())
    entry = universe['markets'][cfg['market']]
    assets = entry['assets']
    meta = metadata(cfg['prices'])
    check_real(meta)
    if (meta.get('market') != cfg['market'] or meta.get('currency') != entry['currency']
            or meta.get('price_type') != 'adjusted_close'):
        raise PipelineError('PROVENANCE_INVALID', 'Market/currency/adjustment mismatch')
    prices = read_prices(Path(cfg['prices']), assets)
    if cfg.get('start'):
        prices = prices.loc[cfg['start']:]
    if cfg.get('end'):
        prices = prices.loc[:cfg['end']]
    if len(prices) < 507:
        raise PipelineError('INSUFFICIENT_HISTORY', 'At least 507 price rows including warm-up are required')
    # Validate cash before the legacy accrual layer; never silently use zero yield.
    yields = pd.read_csv(cfg['cash'], index_col=0, parse_dates=True)
    if (list(yields.columns) != ['annual_yield'] or yields.empty
            or yields.index.has_duplicates or not yields.index.is_monotonic_increasing
            or yields.index.hasnans or not np.isfinite(yields.to_numpy()).all()
            or not yields.annual_yield.between(-.1, .5).all()):
        raise PipelineError('CASH_INVALID', 'Cash must contain unique ascending dates and decimal annual yields')
    if (prices.index[-1] - yields.index[-1]).days > cfg.get('max_cash_age_days', 10):
        raise PipelineError('CASH_STALE', 'Cash observations do not cover the selected price interval')
    cash, audit, cash_meta = lagged_cash(prices, cfg['market'], cfg['cash'])
    if not np.isfinite(cash.to_numpy()).all():
        raise PipelineError('CASH_INVALID', 'Cash accrual has non-finite values')
    # Check the age of every actually used observation, not just the last row.
    observed = pd.to_datetime(audit['last_observation_date'])
    ages = (pd.to_datetime(audit['date']) - observed).dt.days
    if observed.isna().any() or (ages > cfg.get('max_cash_age_days', 10)).any():
        raise PipelineError('CASH_STALE', 'Cash history contains unavailable or stale observations')
    check_freshness(prices.index[-1], cfg)
    return prices, cash, assets, audit, meta, cash_meta


def check_freshness(last_date, cfg):
    if cfg.get('max_price_age_days') is None:
        return
    asof = dt.date.fromisoformat(cfg.get('as_of') or dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat())
    age = (asof - last_date.date()).days
    if age < 0 or age > cfg['max_price_age_days']:
        raise PipelineError('PRICE_STALE', f'Last price {last_date.date()}, as-of {asof}, age {age} days')


def worker_prices(request, provider, destination):
    import pandas as pd
    engine()
    import wind_fetch
    universe = json.loads(Path(request['universe']).read_text())
    entry = universe['markets'][request['market']]
    assets = entry['assets']
    if provider == 'wind':
        from WindPy import w
        started = w.start(waitTime=20)
        if started.ErrorCode or not w.isconnected():
            raise RuntimeError(f'Wind not connected: {started.ErrorCode}')
        frame, options = wind_fetch.fetch(w, list(assets), request['start'], request['end'], request['market'])
        details = {'options': options, 'adjustment': 'Wind PriceAdj=F; wealth index close includes reinvested income',
                   'volume_unit': 'not requested; close-only research panel'}
    else:
        # Require one provider for the complete panel. A partial provider result is discarded.
        engine('single')
        import fetch
        series, details = [], {}
        for code, asset in assets.items():
            if asset.get('instrument_type') == 'index_proxy' or code.endswith('.CS'):
                raise RuntimeError(f'{provider} has no verified equivalent for wealth index {code}; no proxy substitution')
            args = SimpleNamespace(provider=provider, symbol=code, start=request['start'], end=request['end'],
                                   calendar='SSE' if request['market'] == 'CN' else 'NYSE',
                                   adjustment='F', instrument='etf')
            bars, receipt = fetch.provider_fetch(args)
            # Reuse OHLCV validation; its normalized adjusted close supplies the panel.
            bars = bars.loc[request['start']:request['end']]
            first = bars.close.first_valid_index()
            if first is None:
                raise RuntimeError(f'{code}: no observations')
            series.append(bars.loc[first:, 'close'].rename(code))
            details[code] = receipt
        frame = pd.concat(series, axis=1).sort_index()
        details = {'per_asset': details, 'adjustment': 'provider adjusted OHLC, see per_asset',
                   'volume_unit': 'not used by close-only panel; see per_asset for provider units'}
    frame.index.name = 'date'
    frame.to_csv(destination)
    from rotation import read_prices
    read_prices(Path(destination), assets)
    if frame.empty or frame.loc[request['start']:request['end']].empty:
        raise RuntimeError('empty price interval')
    write_json(Path(destination).with_suffix('.meta.json'), {
        'source': {'wind': 'Wind WSD', 'akshare': 'AKShare Eastmoney', 'yahoo': 'Yahoo Finance'}[provider],
        'market': request['market'], 'currency': entry['currency'], 'price_type': 'adjusted_close',
        'synthetic': False, 'requested_start': request['start'], 'requested_end': request['end'],
        'start': str(frame.index[0].date()), 'end': str(frame.index[-1].date()),
        'retrieved_at_utc': now(), 'primary_source': 'Wind API', 'full_panel_single_source': True, **details})


def worker_cash(request, provider, destination):
    import pandas as pd
    if provider == 'wind':
        from WindPy import w
        started = w.start(waitTime=20)
        if started.ErrorCode or not w.isconnected():
            raise RuntimeError(f'Wind not connected: {started.ErrorCode}')
        code = request.get('wind_cash_code') or ('M1010885' if request['market'] == 'CN' else None)
        if not code:
            raise RuntimeError('US cash Wind EDB identifier not configured; refusing to invent a series mapping')
        result = w.edb(code, request['start'], request['end'], 'Fill=Blank')
        if result.ErrorCode or len(result.Data) != 1:
            raise RuntimeError(f'Wind EDB failure: {result.ErrorCode}')
        y = pd.Series(result.Data[0], index=pd.to_datetime(result.Times), dtype=float) / 100
        source, details = 'Wind EDB', {'code': code, 'original_unit': 'percent'}
    elif provider == 'fred' and request['market'] == 'US':
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?' + urllib.parse.urlencode(
            {'id': 'DGS3MO', 'cosd': request['start'], 'coed': request['end']})
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = pd.read_csv(io.BytesIO(response.read()), index_col=0, parse_dates=True)
        y = pd.to_numeric(raw.iloc[:, 0], errors='coerce').dropna() / 100
        source, details = 'Federal Reserve H.15 via FRED', {'code': 'DGS3MO', 'url': url, 'original_unit': 'percent'}
    else:
        raise RuntimeError('No verified alternate for the exact CNY AAA 1-year NCD yield; no different-rate substitution')
    y = y.loc[request['start']:request['end']]
    if (y.empty or y.isna().any() or y.index.has_duplicates or not y.index.is_monotonic_increasing
            or not y.between(-.1, .5).all()):
        raise RuntimeError('Invalid cash yield observations/units')
    y.name, y.index.name = 'annual_yield', 'date'
    y.to_csv(destination)
    write_json(Path(destination).with_suffix('.meta.json'), {
        'source': source, 'currency': 'CNY' if request['market'] == 'CN' else 'USD', 'unit': 'decimal',
        'start': str(y.index[0].date()), 'end': str(y.index[-1].date()),
        'requested_start': request['start'], 'requested_end': request['end'],
        'retrieved_at_utc': now(), 'primary_source': 'Wind API', 'adjustment': 'not applicable',
        'volume_unit': 'not applicable', 'availability': 'Lag applied by cash accrual engine; US uses two business days plus one-day accrual lag', **details})
