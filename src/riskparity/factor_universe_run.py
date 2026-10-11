"""Bulk Wind factor run on an explicitly screened A-share universe."""
from pathlib import Path
import argparse,datetime as dt,json,hashlib
import numpy as np,pandas as pd
from riskparity.factor_library import daily_factors,intraday_factors,wind_frame,validate_daily,fallback_daily,fallback_minute,FUNDAMENTALS,DAILY_FIELDS


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--screen',type=Path,default=Path('outputs/factor_library/ashare_screen_20261007'))
 p.add_argument('--out',type=Path,default=Path('outputs/factor_library/ashare_wind_20261007'))
 p.add_argument('--start',default='2026-04-07');p.add_argument('--warmup-start',default='2026-01-01')
 p.add_argument('--minute-start',default='2026-08-03')
 p.add_argument('--resume',action='store_true')
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=a.resume);raw=a.out/'raw';raw.mkdir(exist_ok=a.resume)
 screen=pd.read_csv(a.screen/'selected.csv');meta=json.loads((a.screen/'manifest.json').read_text());end=meta['asof'];codes=screen.code.tolist()
 from WindPy import w
 status=w.start(waitTime=15)
 if status.ErrorCode or not w.isconnected():raise RuntimeError(f'Wind connection failed {status.ErrorCode}')
 audit=json.loads((a.out/'manifest.partial.json').read_text())['queries'] if a.resume else []
 matrices={};minute_frames={};asset_sources={}
 def checkpoint():
  (a.out/'manifest.partial.json').write_text(json.dumps({'queries':audit},ensure_ascii=False,indent=2))
 def snapshot(fields):
  frames=[]
  for i in range(0,len(codes),200):
   batch=codes[i:i+200];r=w.wss(','.join(batch),fields,f'tradeDate={end};industryType=1;unit=1')
   audit.append({'kind':'snapshot','code_count':len(batch),'fields':fields,'source':'Wind','error_code':r.ErrorCode,'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
   if not r.ErrorCode:
    frames.append(pd.DataFrame({k.lower():v for k,v in zip(r.Fields,r.Data)},index=r.Codes))
   checkpoint()
  return pd.concat(frames) if frames else pd.DataFrame(index=codes)
 def history(field):
  if a.resume:
   cached=sorted(raw.glob(f'bulk_{field}_*.csv'))
   if cached:
    matrix=pd.concat([pd.read_csv(path,parse_dates=['date']).set_index('date') for path in cached],axis=1).sort_index()
    print('CACHE',field,len(matrix.columns),flush=True)
    return matrix
   # Failed optional field requests were already attempted with Wind; don't repeat quota failures.
   if any(q.get('kind')=='daily_bulk' and q.get('fields')==field for q in audit):
    print('UNAVAILABLE',field,'previous Wind errors retained',flush=True)
    return pd.DataFrame()
  chunks=[]
  for i in range(0,len(codes),60):
   batch=codes[i:i+60]
   r=w.wsd(','.join(batch),field,a.warmup_start,end,'PriceAdj=F;Days=Trading;TradingCalendar=SSE;Fill=Blank')
   rec={'kind':'daily_bulk','codes':batch,'fields':field,'source':'Wind','requested_start':a.warmup_start,'requested_end':end,'options':'PriceAdj=F;Days=Trading;TradingCalendar=SSE;Fill=Blank','retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'error_code':r.ErrorCode}
   audit.append(rec)
   if r.ErrorCode or not r.Times or len(r.Data)!=len(batch):
    rec.update(status='unavailable',wind_failure=str(r.Data)[:200]);checkpoint();continue
   f=pd.DataFrame(np.array(r.Data,dtype=float).T,index=pd.to_datetime(r.Times),columns=r.Codes).replace([np.inf,-np.inf],np.nan)
   rec.update(status='ok',rows=len(f),valid_cells=int(f.notna().sum().sum()))
   path=raw/f'bulk_{field}_{i:04d}.csv';f.index.name='date';f.to_csv(path);rec['file']=str(path.relative_to(a.out));chunks.append(f);checkpoint()
  matrix=pd.concat(chunks,axis=1).sort_index() if chunks else pd.DataFrame()
  print('DAILY',field,'assets',len(matrix.columns),'valid',int(matrix.notna().sum().sum()),flush=True)
  return matrix
 try:
  if a.resume and (a.out/'industry_snapshot.csv').exists():
   snap=pd.read_csv(a.out/'industry_snapshot.csv',index_col='code')
  else:
   snap=snapshot('industry_sw,ipo_date');snap.index.name='code';snap.to_csv(a.out/'industry_snapshot.csv')
  for field in DAILY_FIELDS.split(',')+['turn','mkt_cap_ard']+list(FUNDAMENTALS):matrices[field]=history(field)
  benchmark_rec={'kind':'benchmark','code':'000300.SH','fields':'close','source':'Wind','status':'ok','file':'raw/benchmark.csv','requested_start':a.warmup_start,'requested_end':end,'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
  try:
   bm=wind_frame(w.wsd('000300.SH','close',a.warmup_start,end,'PriceAdj=F;Days=Trading;TradingCalendar=SSE;Fill=Blank'))
  except Exception as exc:
   benchmark_rec['wind_failure']=str(exc)
   try:
    import akshare as ak
    bm=ak.index_zh_a_hist(symbol='000300',period='daily',start_date=a.warmup_start.replace('-',''),end_date=end.replace('-',''))
    bm=bm.rename(columns={'日期':'date','收盘':'close'}).assign(date=lambda x:pd.to_datetime(x.date)).set_index('date')[['close']]
    if bm.empty or not bm.close.gt(0).all():raise ValueError('Invalid backup benchmark')
    benchmark_rec.update(source='AKShare',adjustment='index price series; no stock adjustment')
   except Exception as backup:
    benchmark_rec['fallback_failure']=str(backup)
    cached=Path('outputs/factor_library/wind_20261007/raw/000300.SH_daily_close.csv')
    bm=pd.read_csv(cached,parse_dates=['date']).set_index('date')
    benchmark_rec.update(source='Wind offline export',reused_file=str(cached.resolve()),warmup_unavailable=True)
  bm.index.name='date';bm.to_csv(raw/'benchmark.csv')
  audit.append(benchmark_rec)
  frames=[]
  for j,code in enumerate(codes):
   rec={'kind':'daily','code':code,'fields':DAILY_FIELDS,'source':'Wind','status':'ok','requested_start':a.warmup_start,'requested_end':end,'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'adjustment':'PriceAdj=F OHLC; native unadjusted volume/amount','amount_unit':'CNY','volume_unit':'Wind native; no share-count calculation'}
   try:
    d=pd.DataFrame({field:matrices[field][code] for field in DAILY_FIELDS.split(',')})
    d=validate_daily(d)
    if d[['volume','amt']].notna().sum().min()<1:raise ValueError('no volume/amount coverage')
   except Exception as exc:
    rec['wind_failure']=str(exc)
    try:d=fallback_daily(code,a.warmup_start,end);rec.update(source='AKShare',volume_unit='shares',adjustment='qfq OHLC; unadjusted amount/volume')
    except Exception as backup:
     rec.update(status='unavailable',fallback_failure=str(backup));audit.append(rec);checkpoint();continue
   if code in matrices['turn'] and rec['source']=='Wind':d['turn']=matrices['turn'][code].reindex(d.index)
   d.index.name='date';path=raw/f'{code}_daily.csv';d.to_csv(path);rec['file']=str(path.relative_to(a.out));rec['rows']=len(d);audit.append(rec);asset_sources[code]=rec
   f=daily_factors(d,bm.close)
   f['market_cap']=matrices['mkt_cap_ard'][code].reindex(d.index) if code in matrices['mkt_cap_ard'] else np.nan
   # Trailing filter available at each date, within the fixed period-end selected pool.
   f['avg_amount_30']=d.amt.rolling(30,min_periods=30).mean()
   f['historically_eligible']=(f.market_cap>20e9)&(f.avg_amount_30>1e9)
   for field,(name,trans) in FUNDAMENTALS.items():
    s=matrices[field][code].reindex(d.index) if code in matrices[field] else pd.Series(np.nan,index=d.index)
    f[name]=1/s.where(s!=0) if trans=='inverse' else s/100
   minute_rec={'kind':'minute','code':code,'fields':DAILY_FIELDS,'source':'Wind','requested_start':a.minute_start,'requested_end':end,'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'options':'BarSize=5;PriceAdj=F','volume_unit':'Wind native','amount_unit':'CNY'}
   try:
    bars=validate_daily(wind_frame(w.wsi(code,DAILY_FIELDS,a.minute_start+' 09:00:00',end+' 15:00:00','BarSize=5;PriceAdj=F')))
    minute_rec['status']='ok'
   except Exception as exc:
    minute_rec.update(status='unavailable',wind_failure=str(exc));bars=None
    try:bars=fallback_minute(code,a.minute_start+' 09:00:00',end+' 15:00:00');minute_rec.update(status='ok',source='AKShare',volume_unit='shares',adjustment='qfq OHLC')
    except Exception as backup:minute_rec['fallback_failure']=str(backup)
   if bars is not None:
    bars.index.name='date';path=raw/f'{code}_minute.csv';bars.to_csv(path);minute_rec.update(file=str(path.relative_to(a.out)),rows=len(bars),actual_start=str(bars.index.min()),actual_end=str(bars.index.max()))
    high=intraday_factors(bars);f=f.join(high)
   for name in ('realized_variance','realized_vol','downside_realized_variance','tail_amount_fraction','amount_concentration','bar_count'):
    if name not in f:f[name]=np.nan
   audit.append(minute_rec)
   f=f.loc[f.index>=pd.Timestamp(a.start)];f['code']=code;f['name']=screen.set_index('code').sec_name.get(code)
   f['industry']=snap.industry_sw.get(code) if 'industry_sw' in snap else None
   frames.append(f.reset_index(names='date'))
   # Persist progress and completed factors after every asset; safe recovery without re-querying.
   pd.concat(frames,ignore_index=True).to_csv(a.out/'factors.partial.csv',index=False)
   checkpoint();print('FACTORS',j+1,'/',len(codes),code,'days',len(f),'minute',int(f.realized_vol.count()),flush=True)
  if not frames:raise RuntimeError('No usable stocks')
  panel=pd.concat(frames,ignore_index=True);panel.to_csv(a.out/'factors.csv',index=False)
  panel.sort_values('date').groupby('code').tail(1).to_csv(a.out/'latest.csv',index=False)
  factors=[c for c in panel if c not in ('date','code','name','industry','market_cap','avg_amount_30','historically_eligible','bar_count')]
  panel.groupby('code')[factors].count().stack().rename('valid_observations').reset_index().to_csv(a.out/'coverage.csv',index=False)
  manifest={'queries':audit,'screen_manifest':str((a.screen/'manifest.json').resolve()),'requested_start':a.start,'requested_end':end,'warmup_start':a.warmup_start,
            'minute_start':a.minute_start,'assets_requested':len(codes),'assets_success':len(frames),'source_priority':'Wind then AKShare; entire asset/range same source for daily OHLCV',
            'screen_asof':end,'bias':'fixed period-end screen has survivorship and selection bias; daily eligibility mask is only within selected names, not the full historical market',
            'industry_timing':'period-end SW level-1 industry snapshot; historical industry changes unverified; neutralization is sensitivity only',
            'fundamentals_pit_verified':False,'factor_timing':'t close; use t+1 or later prices',
            'files':{str(x.relative_to(a.out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in a.out.rglob('*.csv') if '.partial.' not in x.name}}
  (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 finally:w.stop()

if __name__=='__main__':main()
