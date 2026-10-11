"""Standalone Wind-first factor library. No trading, forecasting or backtesting."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DAILY_FIELDS = 'open,high,low,close,volume,amt'
FUNDAMENTALS = {
    'pe_ttm': ('earnings_yield', 'inverse'),
    'pb_lf': ('book_to_price', 'inverse'),
    'roe_ttm': ('roe', 'percent'),
    'yoynetprofit': ('profit_growth', 'percent'),
    'yoy_or': ('revenue_growth', 'percent'),
    'grossprofitmargin': ('gross_margin', 'percent'),
    'debttoassets': ('debt_to_assets', 'percent'),
}


def daily_factors(data: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """One asset, ascending daily dates. Prices adjusted together; amount in currency."""
    d = data.sort_index()
    c = d.close.where(d.close > 0)
    r = np.log(c / c.shift(1))
    out = pd.DataFrame(index=d.index)
    for n in (5, 20, 60):
        out[f'vol_{n}'] = r.rolling(n).std(ddof=1) * np.sqrt(252)
        out[f'momentum_{n}'] = c / c.shift(n) - 1
    out['downside_vol_20'] = np.sqrt(r.clip(upper=0).pow(2).rolling(20).mean() * 252)
    out['parkinson_vol_20'] = np.sqrt(np.log(d.high / d.low).pow(2).rolling(20).mean() * 252 / (4*np.log(2)))
    out['vol_ratio_20_60'] = out.vol_20 / out.vol_60.replace(0, np.nan)
    out['vol_of_vol_20'] = out.vol_20.rolling(20).std()
    out['atr_pct_20'] = pd.concat([d.high-d.low, (d.high-c.shift()).abs(), (d.low-c.shift()).abs()], axis=1).max(axis=1).rolling(20).mean() / c
    active = d.volume.gt(0) & d.amt.gt(0)
    amount = d.amt.where(active)
    out['amount_20'] = amount.rolling(20).mean()
    out['log_amount_20'] = np.log(out.amount_20)
    out['amihud_20'] = (r.abs() / amount).rolling(20).mean()
    out['amount_ratio_20'] = amount / amount.shift().rolling(20).mean()
    if 'turn' in d:
        out['turnover_20'] = (d.turn / 100).where(active).rolling(20).mean()
    else:
        out['turnover_20'] = np.nan
    out['zero_volume_fraction_20'] = d.volume.eq(0).astype(float).where(d.volume.notna()).rolling(20).mean()
    out['reversal_5'] = -out.momentum_5
    out['momentum_60_skip_5'] = c.shift(5) / c.shift(60) - 1
    out['overnight_return'] = d.open / c.shift() - 1
    out['intraday_return'] = c / d.open - 1
    if benchmark is not None:
        m = np.log(benchmark / benchmark.shift()).reindex(d.index)
        pair = r.notna() & m.notna()
        x, y = r.where(pair), m.where(pair)
        vm = y.rolling(60).var()
        beta = x.rolling(60).cov(y) / vm.replace(0, np.nan)
        out['beta_60'] = beta
        out['correlation_60'] = x.rolling(60).corr(y)
        # OLS residual variance within each window, with intercept, sample denominator n-1.
        residual_var = x.rolling(60).var() - beta.pow(2) * vm
        out['idiosyncratic_vol_60'] = np.sqrt(residual_var.clip(lower=0) * 252)
        out['relative_momentum_20'] = out.momentum_20 - (benchmark / benchmark.shift(20) - 1).reindex(d.index)
    else:
        for name in ('beta_60', 'correlation_60', 'idiosyncratic_vol_60', 'relative_momentum_20'):
            out[name] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def intraday_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """5-minute bars, daily aggregation. Returns do not cross session boundaries."""
    b = bars.sort_index().copy()
    day = b.index.normalize()
    # Includes opening bar return; close-to-next-open gaps are excluded, incl. lunch.
    r = np.log(b.close / b.open)
    grouped = r.pow(2).groupby(day)
    out = pd.DataFrame({'realized_variance': grouped.sum(min_count=1)})
    out['realized_vol'] = np.sqrt(out.realized_variance)
    out['downside_realized_variance'] = r.clip(upper=0).pow(2).groupby(day).sum(min_count=1)
    amount = b.amt.where(b.amt >= 0)
    total = amount.groupby(day).sum(min_count=1).replace(0, np.nan)
    out['amount_concentration'] = (amount / day.map(total)).pow(2).groupby(day).sum(min_count=1)
    tail = (b.index.hour == 14) & (b.index.minute > 30) | (b.index.hour == 15)
    out['tail_amount_fraction'] = amount.where(tail, 0).groupby(day).sum(min_count=1) / total
    out['bar_count'] = b.close.groupby(day).count()
    # Complete A-share 5-minute session: 48 bars. Partial days are not comparable.
    out.loc[out.bar_count != 48, out.columns != 'bar_count'] = np.nan
    return out


def quote_factors(quotes: pd.DataFrame) -> pd.DataFrame:
    """Optional Wind quote export: timestamp index; bid/ask, bid_depth/ask_depth in CNY.
    Equally spaced snapshots required; averages are sampling-weighted, not trade-weighted.
    """
    q = quotes.sort_index()
    valid = (q.ask >= q.bid) & (q.bid > 0) & (q.bid_depth >= 0) & (q.ask_depth >= 0)
    q = q.where(valid)
    depth = q.bid_depth + q.ask_depth
    f = pd.DataFrame({'relative_spread': (q.ask-q.bid)/((q.ask+q.bid)/2),
                      'book_depth': depth,
                      'book_imbalance': (q.bid_depth-q.ask_depth)/depth.replace(0, np.nan)})
    return f.groupby(f.index.normalize()).mean()


def wind_frame(result) -> pd.DataFrame:
    if result.ErrorCode:
        raise RuntimeError(f'Wind ErrorCode={result.ErrorCode}; {str(result.Data)[:180]}')
    if not result.Times or not result.Data:
        raise ValueError('Wind empty response')
    f = pd.DataFrame({k.lower(): v for k, v in zip(result.Fields, result.Data)}, index=pd.to_datetime(result.Times))
    f = f.rename(columns={'amount': 'amt'}).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if f.index.has_duplicates or not f.notna().any().any():
        raise ValueError('Wind duplicate timestamps or all-null data')
    return f.sort_index()


def validate_daily(f):
    if set(DAILY_FIELDS.split(',')) - set(f):
        raise ValueError('missing OHLCV/amount columns')
    f = f.dropna(subset=['close']).copy()
    valid = (f[['open','high','low','close']] > 0).all(axis=1)
    valid &= (f.high >= f[['open','close','low']].max(axis=1)) & (f.low <= f[['open','close','high']].min(axis=1))
    if f.empty or not valid.all() or f[['volume','amt']].lt(0).any().any():
        raise ValueError('daily OHLCV failed validation')
    return f


def fallback_daily(code, start, end):
    """A-share fallback only: entire range from AKShare, no source splicing."""
    import akshare as ak
    if not (code.endswith(('.SH','.SZ')) and code[:6].startswith(('0','3','6'))):
        raise ValueError('AKShare adapter supports mainland stocks only')
    try:
        raw = ak.stock_zh_a_hist(symbol=code[:6], period='daily', start_date=start.replace('-',''), end_date=end.replace('-',''), adjust='')
        adjusted = ak.stock_zh_a_hist(symbol=code[:6], period='daily', start_date=start.replace('-',''), end_date=end.replace('-',''), adjust='qfq')
        if raw.empty or adjusted.empty:
            raise ValueError('AKShare Eastmoney empty response')
    except Exception as exc:
        symbol=('sh' if code.endswith('.SH') else 'sz')+code[:6]
        raw=ak.stock_zh_a_daily(symbol=symbol,start_date=start.replace('-',''),end_date=end.replace('-',''),adjust='')
        adjusted=ak.stock_zh_a_daily(symbol=symbol,start_date=start.replace('-',''),end_date=end.replace('-',''),adjust='qfq')
        raw=raw.assign(date=lambda x:pd.to_datetime(x.date)).set_index('date')
        adjusted=adjusted.assign(date=lambda x:pd.to_datetime(x.date)).set_index('date')
        f=adjusted[['open','high','low','close']].join(raw[['volume','amount','turnover']],how='left').rename(columns={'amount':'amt','turnover':'turn'})
        f['turn']*=100  # Sina turnover is a fraction, unlike Wind/EM percent.
        f=validate_daily(f)
        f.attrs.update(source='AKShare/Sina',volume_unit='shares',turnover_definition='Sina outstanding circulating shares',eastmoney_failure=str(exc))
        return f
    names = {'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','成交额':'amt','换手率':'turn'}
    a, u = [x.rename(columns=names).assign(date=lambda x: pd.to_datetime(x.date)).set_index('date') for x in (adjusted,raw)]
    f = a[['open','high','low','close']].join(u[['volume','amt','turn']], how='left')
    f['volume'] *= 100  # AKShare stock history volume is in lots of 100 shares.
    f=validate_daily(f)
    f.attrs.update(source='AKShare/Eastmoney',volume_unit='shares')
    return f


def fallback_minute(code, start, end):
    import akshare as ak
    if not code.endswith(('.SH', '.SZ')):
        raise ValueError('minute fallback supports mainland stocks only')
    raw = ak.stock_zh_a_hist_min_em(symbol=code[:6], start_date=start, end_date=end, period='5', adjust='')
    adjusted = ak.stock_zh_a_hist_min_em(symbol=code[:6], start_date=start, end_date=end, period='5', adjust='qfq')
    names = {'时间':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','成交额':'amt'}
    a, u = [x.rename(columns=names).assign(date=lambda x: pd.to_datetime(x.date)).set_index('date') for x in (adjusted,raw)]
    f = a[['open','high','low','close']].join(u[['volume','amt']], how='left')
    f['volume'] *= 100
    f = f.loc[(f.index >= pd.Timestamp(start)) & (f.index <= pd.Timestamp(end))]
    return validate_daily(f)


def main():
    today = dt.datetime.now(ZoneInfo('Asia/Shanghai')).date()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codes', default='600519.SH,000333.SZ,600036.SH')
    p.add_argument('--benchmark', default='000300.SH')
    p.add_argument('--start', default=str(today-dt.timedelta(days=183)))
    p.add_argument('--end', default=str(today-dt.timedelta(days=1)))
    p.add_argument('--minute-days', type=int, default=10, help='0 disables minute retrieval')
    p.add_argument('--out', type=Path, default=Path('outputs/factor_library')/today.isoformat())
    a = p.parse_args()
    if dt.date.fromisoformat(a.start) >= dt.date.fromisoformat(a.end) or a.minute_days < 0:
        p.error('start must precede end; minute-days must be nonnegative')
    if a.out.exists():
        p.error('output already exists; use a fresh --out directory')
    a.out.mkdir(parents=True)
    rawdir = a.out/'raw'; rawdir.mkdir()
    audit, frames = [], []
    w, connection_error = None, None
    try:
        from WindPy import w
        status = w.start(waitTime=15)
        if status.ErrorCode or not w.isconnected():
            raise RuntimeError(f'Wind connection failed: {status.ErrorCode}')
    except Exception as exc:
        connection_error = str(exc); w = None
    def query(code, fields, start, end, kind='daily'):
        record = {'code':code, 'fields':fields, 'requested_start':start, 'requested_end':end,
                  'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'source':'Wind', 'kind':kind}
        audit.append(record)
        try:
            if w is None:
                raise RuntimeError(connection_error)
            options = 'BarSize=5;PriceAdj=F' if kind == 'minute' else 'PriceAdj=F;Fill=Blank;Days=Trading'
            record['options'] = options
            f = wind_frame(w.wsi(code, fields, start, end, options) if kind == 'minute' else w.wsd(code, fields, start, end, options))
            if fields == DAILY_FIELDS:
                f = validate_daily(f)
            record.update(status='ok', rows=len(f), actual_start=str(f.index.min()), actual_end=str(f.index.max()))
            f.index.name='date'
            name=f'{code}_{kind}_{fields.replace(",","_")}.csv'
            f.to_csv(rawdir/name)
            record['file']='raw/'+name
            return f, record
        except Exception as exc:
            record.update(status='unavailable', wind_failure=str(exc))
            if fields == DAILY_FIELDS:
                try:
                    f = fallback_minute(code, start, end) if kind == 'minute' else fallback_daily(code, start, end)
                    name=f'{code}_akshare_{kind}.csv'; f.index.name='date'; f.to_csv(rawdir/name)
                    record.update(status='ok',source='AKShare',adjustment='qfq OHLC; unadjusted volume/amount',volume_unit='shares',amount_unit='CNY',rows=len(f),file='raw/'+name,actual_start=str(f.index.min()),actual_end=str(f.index.max()))
                    return f, record
                except Exception as backup:
                    record['fallback_failure']=str(backup)
            else:
                record['fallback_status']='no equivalent audited adapter; left missing'
            return None, record
    try:
        benchmark, _ = query(a.benchmark, 'close', a.start, a.end)
        for code in dict.fromkeys(a.codes.split(',')):
            code=code.strip()
            d, provenance = query(code, DAILY_FIELDS, a.start, a.end)
            if d is None:
                print(f'{code}: daily data unavailable', flush=True); continue
            if provenance['source'] == 'Wind':
                turn, _ = query(code,'turn',a.start,a.end)
                if turn is not None: d['turn']=turn.turn.reindex(d.index)
                provenance.update(adjustment='Wind PriceAdj=F (forward-adjusted OHLC)',volume_unit='Wind native; no share-based calculation',amount_unit='CNY for mainland equities')
            f=daily_factors(d, None if benchmark is None else benchmark.close)
            for field, (name, transform) in FUNDAMENTALS.items():
                fundamental, rec = query(code,field,a.start,a.end)
                rec['pit_verified']=False
                if fundamental is not None:
                    s=fundamental[field].reindex(f.index)
                    f[name]=(1/s.where(s != 0)) if transform == 'inverse' else s/100
                else: f[name]=np.nan
            if a.minute_days:
                ms=max(dt.date.fromisoformat(a.start),dt.date.fromisoformat(a.end)-dt.timedelta(days=a.minute_days))
                bars, _ = query(code,DAILY_FIELDS,str(ms)+' 09:00:00',a.end+' 15:00:00','minute')
                if bars is not None: f=f.join(intraday_factors(bars))
            for name in ('realized_variance', 'realized_vol', 'downside_realized_variance',
                         'tail_amount_fraction', 'amount_concentration', 'bar_count'):
                if name not in f: f[name] = np.nan
            f['code']=code
            frames.append(f.reset_index(names='date'))
            print(f'{code}: {len(f)} dates, {len(f.columns)-2} factors; {provenance["source"]}',flush=True)
        if frames:
            panel=pd.concat(frames,ignore_index=True)
            # Rank within the requested universe on each date, with at least 3 valid assets.
            columns=[c for c in panel if c not in ('date','code','bar_count')]
            g=panel.groupby('date')[columns]
            ranked=g.rank(pct=True).where(g.transform('count') >= 3)
            ranked=pd.concat([panel[['date','code']],ranked],axis=1)
            panel.to_csv(a.out/'factors.csv',index=False)
            ranked.to_csv(a.out/'cross_section_ranks.csv',index=False)
            panel.sort_values('date').groupby('code').tail(1).to_csv(a.out/'latest.csv',index=False)
            coverage=panel.groupby('code')[columns].count().stack().rename('valid_observations').reset_index()
            coverage.to_csv(a.out/'coverage.csv',index=False)
        manifest={'requested_start':a.start,'requested_end':a.end,'benchmark':a.benchmark,'queries':audit,
                  'factor_timing':'date t factor is available after t close; execution must use later prices',
                  'fundamentals_pit_verified':False,'fundamentals_usage':'descriptive research only; historical release/revision timestamps not audited',
                  'minute_scope':'5-minute intraday only; overnight and lunch gaps excluded; incomplete sessions masked',
                  'rank_scope':'only requested codes, minimum 3; raw rank has no investment direction',
                  'missing_data':'never filled with synthetic data; unsupported fields remain missing',
                  'files':{str(x.relative_to(a.out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in a.out.rglob('*.csv')}}
        (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        if w is not None: w.stop()
    if not frames: raise SystemExit('No usable daily data; see manifest.json')


if __name__ == '__main__':
    main()
