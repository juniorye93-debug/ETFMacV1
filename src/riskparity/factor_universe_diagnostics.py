"""Large-universe factor diagnostics: quintiles, controls, stability, HAC/FDR."""
from pathlib import Path
import argparse,json,hashlib,datetime as dt
import numpy as np,pandas as pd
from scipy.stats import norm
from riskparity.factor_diagnostics import rank_corr,FUNDAMENTALS,HIGH_FREQ

META={'bar_count','market_cap','avg_amount_30','historically_eligible','name','industry'}


def hac_mean(s,lag):
 x=np.asarray(s,dtype=float);x=x[np.isfinite(x)];n=len(x)
 # Do not issue asymptotic p-values on very short series with a long HAC window.
 if n<max(10,4*(lag+1)):return np.nan,np.nan,np.nan
 u=x-x.mean();v=np.dot(u,u)/n
 for k in range(1,min(lag,n-1)+1):v+=2*(1-k/(lag+1))*np.dot(u[k:],u[:-k])/n
 se=np.sqrt(max(v,0)/n)
 if se<=0:return se,np.nan,np.nan
 t=x.mean()/se;return se,t,2*norm.sf(abs(t))


def bh_fdr(p):
 s=pd.Series(p,dtype=float);valid=s.dropna();order=valid.sort_values();n=len(order)
 q=(order.to_numpy()*n/np.arange(1,n+1));q=np.minimum.accumulate(q[::-1])[::-1].clip(0,1)
 result=pd.Series(np.nan,index=s.index);result.loc[order.index]=q;return result


def partial_rank_ic(v,factor,target):
 if len(v)<30:return np.nan
 x=v[factor].rank().to_numpy(dtype=float);y=v[target].rank().to_numpy(dtype=float)
 cap=np.log(v.market_cap.where(v.market_cap>0));valid=cap.notna() & v.industry.notna()
 if valid.sum()<30:return np.nan
 v=v.loc[valid];x=x[valid.to_numpy()];y=y[valid.to_numpy()];cap=cap.loc[valid]
 cap=(cap-cap.mean())/cap.std()
 industries=pd.get_dummies(v.industry.fillna('Unknown'),drop_first=True,dtype=float)
 design=np.column_stack([np.ones(len(v)),cap,industries])
 if len(v)-np.linalg.matrix_rank(design)<20:return np.nan
 xr=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
 yr=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
 if np.std(xr)<1e-10 or np.std(yr)<1e-10:return np.nan
 return np.corrcoef(xr,yr)[0,1]


def quintile_returns(v,factor,target):
 if len(v)<30 or v[factor].nunique()<5:return [np.nan]*5
 groups=np.ceil(v[factor].rank(method='average',pct=True)*5).clip(1,5).astype(int)
 return [v.loc[groups==i,target].mean() for i in range(1,6)]


def run(root,out):
 out.mkdir(parents=True,exist_ok=False)
 m=json.loads((root/'manifest.json').read_text());f=pd.read_csv(root/'factors.csv',parse_dates=['date']).set_index(['date','code']).sort_index()
 features=[c for c in f if c not in META]
 dates=f.index.get_level_values('date').unique().sort_values();codes=f.index.get_level_values('code').unique();n=len(codes);split=dates[len(dates)//2]
 source={q['code']:q for q in m['queries'] if q['kind']=='daily' and q['status']=='ok'}
 wind_codes={code for code,q in source.items() if q['source'].startswith('Wind')}
 labels=[];profiles=[]
 for code in codes:
  d=pd.read_csv(root/source[code]['file'],parse_dates=['date']).set_index('date').sort_index().reindex(dates);z=pd.DataFrame(index=d.index);r=np.log(d.close/d.close.shift())
  for h in (5,20):
   z[f'return_{h}']=d.close.shift(-h)/d.open.shift(-1)-1
   z.loc[d.volume.shift(-1).le(0)|d.volume.shift(-h).le(0),f'return_{h}']=np.nan
   z[f'exit_date_{h}']=pd.Series(d.index,index=d.index).shift(-h)
   z[f'future_vol_{h}']=r.rolling(h).std().shift(-h)*np.sqrt(252)
   z[f'future_amount_{h}']=d.amt.rolling(h).mean().shift(-h)
   z[f'future_illiquidity_{h}']=(r.abs()/d.amt.where(d.amt>0)).rolling(h).mean().shift(-h)
  z['code']=code;labels.append(z.reset_index())
  d=d.dropna(subset=['close']);wealth=d.close/d.close.iloc[0]
  profiles.append({'code':code,'dates':len(d),'start':str(d.index.min().date()),'end':str(d.index.max().date()),'return':wealth.iloc[-1]-1,'vol':np.log(d.close/d.close.shift()).std()*np.sqrt(252),'max_drawdown':(wealth/wealth.cummax()-1).min()})
 labels=pd.concat(labels).set_index(['date','code']);z=f.join(labels)
 labels.loc[labels.index.get_level_values('date')>=dates[0]].reset_index().to_csv(out/'forward_targets.csv',index=False)
 pd.DataFrame(profiles).to_csv(out/'asset_profiles.csv',index=False)
 quality=[];eligible=[]
 for c in features:
  coverage=f[c].groupby('code').count();uniques=f[c].groupby('code').nunique();good=(coverage>=40).sum()
  state='evaluate'
  if c in FUNDAMENTALS:state='PIT unverified; descriptive only'
  elif good < .8*n:state='insufficient coverage (<80% stocks have 40 dates)'
  elif f[c].nunique()<=1:state='constant'
  else:eligible.append(c)
  quality.append({'factor':c,'valid_cells':int(coverage.sum()),'stocks_with_40_dates':int(good),'stock_count':n,'min_dates':int(coverage.min()),'median_dates':coverage.median(),'max_dates':int(coverage.max()),'unique_values':int(f[c].nunique()),'status':state})
 pd.DataFrame(quality).to_csv(out/'factor_quality.csv',index=False)
 daily=[];results=[];monthly=[];per_asset=[];risk=[]
 for h in (5,20):
  target=f'return_{h}';cutoff=max(30,int(.8*n))
  for factor in eligible:
   records=[]
   for date,d in z.groupby('date'):
    v=d[[factor,target,'market_cap','industry',f'exit_date_{h}','historically_eligible']].dropna(subset=[factor,target])
    if len(v)<cutoff or v[factor].nunique()<2:continue
    ic=rank_corr(v[factor],v[target]);neutral=partial_rank_ic(v,factor,target);qs=quintile_returns(v,factor,target)
    historic=v[v.historically_eligible==True]
    hic=rank_corr(historic[factor],historic[target]) if len(historic)>=30 else np.nan
    wind=v.loc[v.index.get_level_values('code').isin(wind_codes)]
    wic=rank_corr(wind[factor],wind[target]) if len(wind)>=30 else np.nan
    row={'date':date,'factor':factor,'horizon':h,'stock_count':len(v),'rank_ic':ic,'neutral_rank_ic':neutral,'neutral_stock_count':int((v.industry.notna() & v.market_cap.gt(0)).sum()),'daily_eligible_rank_ic':hic,'daily_eligible_stocks':len(historic),'wind_only_rank_ic':wic,'wind_only_stocks':len(wind),
         'q1_return':qs[0],'q2_return':qs[1],'q3_return':qs[2],'q4_return':qs[3],'q5_return':qs[4],'q5_minus_q1':qs[4]-qs[0],
         'early':bool(date<split and v[f'exit_date_{h}'].max()<split),'late':bool(date>=split)}
    records.append(row);daily.append(row)
   g=pd.DataFrame(records)
   if g.empty:continue
   g=g.set_index('date').sort_index();anchors=set(dates[::h]);non=g.loc[g.index.isin(anchors)]
   se,t,pvalue=hac_mean(g.rank_ic,max(20,h-1))
   neutral_se,neutral_t,neutral_p=hac_mean(g.neutral_rank_ic,max(20,h-1))
   for month,sub in g.groupby(g.index.to_period('M')):monthly.append({'month':str(month),'factor':factor,'horizon':h,'ic_days':len(sub),'rank_ic':sub.rank_ic.mean(),'neutral_rank_ic':sub.neutral_rank_ic.mean(),'q5_minus_q1':sub.q5_minus_q1.mean()})
   per=[]
   for code,sub in z.groupby('code'):
    v=sub[[factor,target]].dropna();ic=rank_corr(v[factor],v[target]) if len(v)>=20 else np.nan
    per_asset.append({'code':code,'factor':factor,'horizon':h,'observations':len(v),'time_series_rank_ic':ic});per.append(ic)
   results.append({'factor':factor,'horizon':h,'ic_days':len(g),'median_stock_count':g.stock_count.median(),'mean_rank_ic':g.rank_ic.mean(),'neutral_rank_ic':g.neutral_rank_ic.mean(),
                   'median_neutral_stock_count':g.neutral_stock_count.median(),
                   'early_rank_ic_purged':g.loc[g.early,'rank_ic'].mean(),'late_rank_ic':g.loc[g.late,'rank_ic'].mean(),'early_days':int(g.early.sum()),'late_days':int(g.late.sum()),
                   'early_neutral_ic':g.loc[g.early,'neutral_rank_ic'].mean(),'late_neutral_ic':g.loc[g.late,'neutral_rank_ic'].mean(),
                   'daily_eligible_rank_ic':g.daily_eligible_rank_ic.mean(),'median_daily_eligible_stocks':g.daily_eligible_stocks.median(),
                   'wind_only_rank_ic':g.wind_only_rank_ic.mean(),'median_wind_only_stocks':g.wind_only_stocks.median(),
                   'nonoverlap_rank_ic':non.rank_ic.mean(),'nonoverlap_periods':len(non),'nonoverlap_neutral_ic':non.neutral_rank_ic.mean(),
                   'positive_ic_fraction':(g.rank_ic>0).mean(),'q1_return':g.q1_return.mean(),'q2_return':g.q2_return.mean(),'q3_return':g.q3_return.mean(),'q4_return':g.q4_return.mean(),'q5_return':g.q5_return.mean(),'q5_minus_q1':g.q5_minus_q1.mean(),
                   'ic_hac_se':se,'ic_hac_t':t,'ic_hac_p':pvalue,'neutral_hac_p':neutral_p,'mean_time_series_rank_ic':pd.Series(per).mean()})
   print('EVALUATED',h,factor,'days',len(g),'IC',round(g.rank_ic.mean(),4),'neutral',round(g.neutral_rank_ic.mean(),4),flush=True)
 r=pd.DataFrame(results);r['ic_fdr_q']=bh_fdr(r.ic_hac_p);r['neutral_fdr_q']=bh_fdr(r.neutral_hac_p);r.to_csv(out/'return_diagnostics.csv',index=False)
 pd.DataFrame(daily).to_csv(out/'daily_rank_ic.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_rank_ic.csv',index=False);pd.DataFrame(per_asset).to_csv(out/'per_asset_return_ic.csv',index=False)
 risk_features=['vol_5','vol_20','vol_60','downside_vol_20','parkinson_vol_20','atr_pct_20','idiosyncratic_vol_60','vol_ratio_20_60','vol_of_vol_20']
 for h in (5,20):
  for factor in risk_features+['amount_20','log_amount_20','amihud_20','turnover_20','amount_ratio_20']:
   target=f'future_vol_{h}' if factor in risk_features else f'future_illiquidity_{h}' if factor=='amihud_20' else f'future_amount_{h}'
   for code,sub in z.groupby('code'):
    v=sub[[factor,target,f'exit_date_{h}']].dropna();early=v[(v.index.get_level_values('date')<split)&(v[f'exit_date_{h}']<split)];late=v[v.index.get_level_values('date')>=split]
    risk.append({'code':code,'factor':factor,'horizon':h,'target':target,'observations':len(v),'rank_correlation':rank_corr(v[factor],v[target]) if len(v)>=20 else np.nan,
                 'early_correlation':rank_corr(early[factor],early[target]) if len(early)>=20 else np.nan,'late_correlation':rank_corr(late[factor],late[target]) if len(late)>=20 else np.nan})
 risk=pd.DataFrame(risk);risk.to_csv(out/'risk_liquidity_diagnostics.csv',index=False)
 risk.groupby(['factor','horizon','target'])[['rank_correlation','early_correlation','late_correlation']].agg(['mean','median','count']).to_csv(out/'risk_liquidity_summary.csv')
 # Correlation of cross-sectional ranks, averaged over dates; avoids stock-level scale effects.
 redundancy=[]
 corr=[d[eligible].rank().corr() for _,d in f.groupby('date')]
 average=sum(x.fillna(0) for x in corr)/sum(x.notna().astype(int) for x in corr)
 average.to_csv(out/'factor_rank_correlation.csv')
 for i,a in enumerate(eligible):
  for b in eligible[i+1:]:
   if abs(average.loc[a,b])>=.95:redundancy.append({'factor_a':a,'factor_b':b,'mean_cross_section_rank_corr':average.loc[a,b]})
 pd.DataFrame(redundancy).to_csv(out/'redundancy.csv',index=False)
 profile=f.groupby('date').agg(stock_count=('name','size'),eligible_count=('historically_eligible','sum'));profile.to_csv(out/'pool_coverage.csv')
 analysis={'input':str(root.resolve()),'source':'frozen Wind-first exports; see input manifest for per-series source','screen_asof':m['screen_asof'],'stock_count':n,'dates':len(dates),'evaluation_start':str(dates[0].date()),'evaluation_end':str(dates[-1].date()),'split_date':str(split.date()),'factor_count':len(features),'evaluated_factor_count':len(eligible),
           'label':'common SSE trading calendar, missing provider dates not skipped; t+1 forward-adjusted open to t+h forward-adjusted close; zero-volume entry/exit removed; no costs, limits or borrow model',
           'group_returns':'daily cross-sectional quintile forward returns, overlapping windows; not compounded strategy returns',
           'industry_size_control':'partial Pearson correlation of ranked factor and ranked forward return after log market-cap and period-end SW industry controls; descriptive sensitivity, not PIT industry history',
           'hac':'Bartlett Newey-West mean IC variance, lag max(20,h-1); asymptotic two-sided normal p only when n >= 4*(lag+1); exploratory',
           'fdr':'Benjamini-Hochberg across all tested factor-horizon pairs, raw and neutral separately; exploratory under correlated factors',
           'daily_eligibility':'within fixed endpoint-selected pool only; not full historical universe',
           'bias':'period-end universe creates selection/survivorship bias; early/late are descriptive splits rather than untouched out-of-sample',
           'input_sha256':hashlib.sha256((root/'factors.csv').read_bytes()).hexdigest(),
           'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}}
 (out/'analysis_manifest.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2));return r


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('outputs/factor_library/ashare_wind_20261007'));p.add_argument('--out',type=Path,default=Path('outputs/factor_library/ashare_analysis_20261007'))
 a=p.parse_args();r=run(a.input,a.out);print(r.sort_values('mean_rank_ic').to_string(index=False))

if __name__=='__main__':main()
