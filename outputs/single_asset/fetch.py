"""Wind first; on failure try verified full-series fallback without mixing sources."""
import argparse, datetime as dt, json, sqlite3, os, subprocess, sys, tempfile, re
from pathlib import Path
import pandas as pd

FIELDS=['open','high','low','close','volume']
SOURCES={'wind':'Wind WSD','akshare':'AKShare Eastmoney','yahoo':'Yahoo Finance'}

def fetch(w,symbol,start,end,calendar,adjustment):
    options=f'Period=D;Days=Trading;TradingCalendar={calendar};PriceAdj={adjustment};Fill=Blank'
    r=w.wsd(symbol,','.join(FIELDS),start,end,options)
    if r.ErrorCode: raise RuntimeError(f'Wind WSD failed: {r.ErrorCode}')
    if len(r.Data)!=5 or any(len(v)!=len(r.Times) for v in r.Data): raise ValueError('unexpected Wind OHLCV layout')
    if [s.lower() for s in r.Fields]!=FIELDS: raise ValueError('unexpected Wind field order')
    d=pd.DataFrame(dict(zip(FIELDS,r.Data)),index=pd.to_datetime(r.Times));d.index.name='date'
    if d.empty: raise RuntimeError('Wind returned no data')
    return d,options

def normalize_cn(d):
    d=d.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
    d['date']=pd.to_datetime(d['date']);return d.set_index('date')[FIELDS].astype(float)

def yahoo_symbol(symbol):
    code,_,suffix=symbol.rpartition('.')
    if suffix=='SH': return code+'.SS'
    if suffix in {'SZ','HK'}: return symbol
    if suffix in {'O','N','P'}: return code
    if '.' not in symbol: return symbol
    raise ValueError('No verified Yahoo mapping for '+symbol)

def fallback_providers(symbol,instrument):
    domestic=symbol.endswith(('.SH','.SZ'))
    futures=symbol.endswith(('.CFE','.SHF','.DCE','.CZC','.INE','.GFE'))
    if futures: return ['akshare']
    if instrument=='index': return []  # Index points require an explicit tradable mapping.
    return ['akshare','yahoo'] if domestic else ['yahoo']

def provider_fetch(a):
    if a.provider=='wind':
        print('Loading official WindPy SDK',flush=True)
        from WindPy import w
        print('Connecting Wind API',flush=True);r=w.start(waitTime=20)
        if r.ErrorCode or not w.isconnected(): raise RuntimeError(f'Wind not connected: {r.ErrorCode}')
        d,options=fetch(w,a.symbol,a.start,a.end,a.calendar,a.adjustment)
        return d,{'source':'Wind WSD','options':options,'adjustment':a.adjustment,'volume_unit':'Wind reported units','series_type':'contract' if a.instrument=='future' else a.instrument}
    if a.provider=='akshare':
        import akshare as ak
        if a.instrument=='future':
            code=a.symbol.split('.')[0]
            if not re.fullmatch(r'[A-Za-z]+\d{3,4}',code): raise ValueError('Fallback futures must specify an actual contract; no automatic continuous-series replacement')
            d=ak.futures_zh_daily_sina(symbol=code.upper());d['date']=pd.to_datetime(d.date);d=d.set_index('date')[FIELDS].astype(float)
            return d,{'source':'AKShare Sina Futures','options':'futures_zh_daily_sina','adjustment':'U','volume_unit':'contracts','series_type':'contract'}
        if not a.symbol.endswith(('.SH','.SZ')): raise ValueError('AKShare fallback configured for mainland instruments only')
        adjust={'F':'qfq','B':'hfq','U':''}[a.adjustment]
        fn=ak.fund_etf_hist_em if a.instrument=='etf' else ak.stock_zh_a_hist
        d=normalize_cn(fn(symbol=a.symbol.split('.')[0],period='daily',start_date=a.start.replace('-',''),end_date=a.end.replace('-',''),adjust=adjust))
        return d,{'source':'AKShare Eastmoney','options':fn.__name__,'adjustment':adjust,'volume_unit':'provider reported lots; unchanged','series_type':a.instrument}
    if a.provider=='yahoo':
        if a.instrument in {'future','index'}: raise ValueError('Yahoo fallback cannot substitute futures/indices with unverified symbols')
        if a.adjustment=='B': raise ValueError('Yahoo does not supply matching Wind backward adjustment')
        import yfinance as yf
        mapped=yahoo_symbol(a.symbol)
        d=yf.download(mapped,start=a.start,end=str(dt.date.fromisoformat(a.end)+dt.timedelta(days=1)),auto_adjust=a.adjustment!='U',progress=False,timeout=20)
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        d=d.rename(columns=str.lower)[FIELDS];d.index=pd.to_datetime(d.index).tz_localize(None);d.index.name='date'
        return d,{'source':'Yahoo Finance','options':f'yfinance.download({mapped})','adjustment':'auto_adjust including dividends' if a.adjustment!='U' else 'raw','volume_unit':'shares','series_type':a.instrument}
    raise ValueError('Unknown provider')

def persist(a,d,meta):
    if d.empty: raise RuntimeError('Provider returned no observations')
    d=d.loc[(d.index>=pd.Timestamp(a.start))&(d.index<=pd.Timestamp(a.end)),FIELDS]
    valid=d[FIELDS[:4]].notna().all(axis=1)
    if not valid.any(): raise RuntimeError('No valid listed OHLC observations')
    d=d.loc[valid[valid].index[0]:];d.index.name='date'
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);path=out/f'{a.symbol}.csv'
    from model import read_data
    # Validate a temporary file before replacing a previously good export.
    with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',dir=out,delete=False) as f:
        candidate=Path(f.name);d.to_csv(f)
    try:
        read_data(candidate)
        candidate.replace(path)
    finally:
        if candidate.exists(): candidate.unlink()
    meta.update(symbol=a.symbol,provider=a.provider,primary_source='Wind API',fields=FIELDS,start=str(d.index[0].date()),end=str(d.index[-1].date()),rows=len(d),requested_start=a.start,requested_end=a.end,
                retrieved_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),synthetic=False,full_series_single_source=True)
    path.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    with sqlite3.connect(out/'market_signals.db') as con:
        con.execute('CREATE TABLE IF NOT EXISTS daily_prices(symbol TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,source TEXT,PRIMARY KEY(symbol,date))')
        # Remove the old series to avoid retaining a mixed-source tail or extra past dates.
        con.execute('DELETE FROM daily_prices WHERE symbol=?',(a.symbol,))
        con.executemany('INSERT INTO daily_prices VALUES(?,?,?,?,?,?,?,?)',[(a.symbol,str(t.date()),*map(float,row),meta['source']) for t,row in d.iterrows()])
    print(json.dumps(meta,indent=2,ensure_ascii=False));return meta

def attempt_provider(a,provider):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--symbol',a.symbol,'--start',a.start,'--end',a.end,'--calendar',a.calendar,'--adjustment',a.adjustment,'--out',a.out,'--instrument',a.instrument,'--provider',provider]
    requested_at=dt.datetime.now(dt.timezone.utc)
    try:
        result=subprocess.run(command,env=dict(os.environ,SINGLE_ASSET_DATA_WORKER='1'),capture_output=True,text=True,timeout=a.timeout)
        if result.returncode:
            return {'provider':provider,'success':False,'reason':result.stderr.strip().splitlines()[-1][-500:] if result.stderr.strip() else result.stdout[-500:]}
        exported=Path(a.out)/f'{a.symbol}.meta.json'
        if not exported.exists():
            return {'provider':provider,'success':False,'reason':'worker exited without exporting valid data metadata: '+result.stdout[-300:]}
        meta=json.loads(exported.read_text())
        if dt.datetime.fromisoformat(meta['retrieved_at_utc'])<requested_at:
            return {'provider':provider,'success':False,'reason':'worker did not refresh data; prior export preserved'}
        if meta.get('provider')!=provider or meta.get('requested_start')!=a.start or meta.get('requested_end')!=a.end:
            return {'provider':provider,'success':False,'reason':'worker receipt does not match requested provider and interval'}
        return {'provider':provider,'success':True,'reason':None}
    except subprocess.TimeoutExpired:
        return {'provider':provider,'success':False,'reason':f'timeout after {a.timeout}s'}

def orchestrate(a,attempt=attempt_provider):
    providers=['wind']+fallback_providers(a.symbol,a.instrument) if a.provider=='auto' else [a.provider]
    attempts=[]
    for provider in providers:
        print(f'Trying {provider}',flush=True);receipt=attempt(a,provider);attempts.append(receipt)
        if receipt['success']:
            path=Path(a.out)/f'{a.symbol}.meta.json';meta=json.loads(path.read_text())
            meta.update(attempts=attempts,fallback_used=provider!='wind');path.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
            print(json.dumps(meta,ensure_ascii=False,indent=2));return meta
        print(f"{provider} failed: {receipt['reason']}",flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.symbol}.fetch_failures.json').write_text(json.dumps(attempts,indent=2,ensure_ascii=False))
    raise RuntimeError('All supported data providers failed; existing valid exports were preserved')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol',default='510300.SH');p.add_argument('--start',default='2005-01-01');p.add_argument('--end',default='2026-10-05')
    p.add_argument('--calendar',default='SSE');p.add_argument('--adjustment',choices=['F','B','U'],default='F')
    p.add_argument('--instrument',choices=['stock','etf','future','index'],default='etf');p.add_argument('--out',default='outputs/single_asset/data')
    p.add_argument('--provider',choices=['auto','wind','akshare','yahoo'],default='auto');p.add_argument('--timeout',type=int,default=55)
    a=p.parse_args()
    if dt.date.fromisoformat(a.start)>=dt.date.fromisoformat(a.end) or a.timeout<1: raise ValueError('invalid date interval or timeout')
    if os.environ.get('SINGLE_ASSET_DATA_WORKER')=='1':
        d,meta=provider_fetch(a);persist(a,d,meta)
    else: orchestrate(a)
if __name__=='__main__':main()
