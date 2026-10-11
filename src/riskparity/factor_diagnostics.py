"""Descriptive factor diagnostics on frozen Wind snapshots, not a strategy backtest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FUNDAMENTALS = {'earnings_yield','book_to_price','roe','profit_growth','revenue_growth','gross_margin','debt_to_assets'}
HIGH_FREQ = {'realized_variance','realized_vol','downside_realized_variance','amount_concentration','tail_amount_fraction'}


def rank_corr(x, y):
    z = pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
    if len(z) < 3 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return np.nan
    return z.x.rank().corr(z.y.rank())


def mean_safe(values):
    s=pd.Series(values,dtype=float).dropna()
    return s.mean() if len(s) else np.nan


def diagnostics(root: Path, out: Path):
    if out.exists(): raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True)
    f=pd.read_csv(root/'factors.csv',parse_dates=['date']).set_index(['date','code']).sort_index()
    manifest=json.loads((root/'manifest.json').read_text())
    sources={q['code']:q for q in manifest['queries'] if q['kind']=='daily' and q['fields']=='open,high,low,close,volume,amt' and q['status']=='ok'}
    codes=f.index.get_level_values('code').unique()
    prices={}
    for code in codes:
        prices[code]=pd.read_csv(root/sources[code]['file'],parse_dates=['date']).set_index('date').sort_index()
    features=[c for c in f if c != 'bar_count']
    dates=f.index.get_level_values('date').unique().sort_values()
    split=dates[len(dates)//2]
    quality=[]
    eligible=[]
    for name in features:
        counts=f[name].groupby('code').count()
        uniques=f[name].groupby('code').nunique()
        status='evaluate'
        if name in FUNDAMENTALS: status='PIT unverified; descriptive only'
        elif counts.min() < 40: status='insufficient history'
        elif uniques.max() <= 1: status='constant; no discrimination'
        else: eligible.append(name)
        quality.append({'factor':name,'valid_rows':int(counts.sum()),'min_per_asset':int(counts.min()),'max_per_asset':int(counts.max()),'min_unique_per_asset':int(uniques.min()),'status':status})
    pd.DataFrame(quality).to_csv(out/'factor_quality.csv',index=False)
    labels=[]
    targets={}
    for code,d in prices.items():
        z=pd.DataFrame(index=d.index)
        r=np.log(d.close/d.close.shift())
        for h in (5,20):
            z[f'return_{h}']=d.close.shift(-h)/d.open.shift(-1)-1
            z[f'exit_date_{h}']=pd.Series(d.index,index=d.index).shift(-h)
            # Annualized future sample standard deviation of close-to-close log returns.
            z[f'future_vol_{h}']=r.rolling(h).std(ddof=1).shift(-h)*np.sqrt(252)
            z[f'future_amount_{h}']=d.amt.rolling(h).mean().shift(-h)
            illiquidity = r.abs()/d.amt.where(d.amt > 0)
            z[f'future_illiquidity_{h}']=illiquidity.rolling(h).mean().shift(-h)
        targets[code]=z
        z['code']=code; labels.append(z.reset_index())
    labels=pd.concat(labels).set_index(['date','code']).sort_index()
    joined=f.join(labels)
    labels.reset_index().to_csv(out/'forward_targets.csv',index=False)
    records=[];daily_records=[];monthly_records=[];time_records=[]
    for h in (5,20):
        target=f'return_{h}'
        for name in eligible:
            series=[];spreads=[];early=[];late=[]
            for date,d in joined.groupby('date'):
                v=d[[name,target]].dropna()
                if len(v)!=len(codes) or v[name].nunique()<2: continue
                ic=rank_corr(v[name],v[target])
                hi=v.loc[v[name]==v[name].max(),target].mean()
                lo=v.loc[v[name]==v[name].min(),target].mean()
                spread=hi-lo
                daily_records.append({'date':date,'factor':name,'horizon':h,'rank_ic':ic,'high_minus_low_return':spread})
                series.append((date,ic));spreads.append(spread)
                exits=d[f'exit_date_{h}'].dropna()
                if date < split and len(exits) and exits.max() < split: early.append(ic)
                if date >= split: late.append(ic)
            s=pd.Series({date:ic for date,ic in series},dtype=float).sort_index()
            for month,g in s.groupby(s.index.to_period('M')):
                monthly_records.append({'month':str(month),'factor':name,'horizon':h,'days':len(g),'rank_ic':g.mean()})
            # Disjoint holding intervals anchored on the full calendar, no phase selection.
            anchors=set(dates[::h]);nonoverlap=s.loc[s.index.isin(anchors)]
            per_asset=[]
            for code,z in joined.groupby('code'):
                v=z[[name,target]].dropna()
                ic=rank_corr(v[name],v[target]) if len(v)>=20 else np.nan
                time_records.append({'code':code,'factor':name,'horizon':h,'observations':len(v),'time_series_rank_ic':ic})
                per_asset.append(ic)
            records.append({'factor':name,'horizon':h,'ic_days':len(s),'mean_rank_ic':s.mean(),
                            'early_rank_ic_purged':mean_safe(early),'late_rank_ic':mean_safe(late),
                            'early_days':len(early),'late_days':len(late),
                            'nonoverlap_rank_ic':nonoverlap.mean(),'nonoverlap_periods':len(nonoverlap),
                            'positive_ic_fraction':(s>0).mean() if len(s) else np.nan,
                            'mean_high_minus_low_return':mean_safe(spreads),
                            'mean_time_series_rank_ic':mean_safe(per_asset)})
    results=pd.DataFrame(records)
    results.to_csv(out/'return_diagnostics.csv',index=False)
    pd.DataFrame(daily_records).to_csv(out/'daily_rank_ic.csv',index=False)
    pd.DataFrame(monthly_records).to_csv(out/'monthly_rank_ic.csv',index=False)
    pd.DataFrame(time_records).to_csv(out/'per_asset_return_ic.csv',index=False)
    # Evaluate intended risk/liquidity use, independently of return prediction.
    risk=[]
    risk_features=['vol_5','vol_20','vol_60','downside_vol_20','parkinson_vol_20','atr_pct_20','idiosyncratic_vol_60','vol_ratio_20_60','vol_of_vol_20']
    for h in (5,20):
        for name in risk_features+['amount_20','log_amount_20','amihud_20','turnover_20','amount_ratio_20']:
            target=(f'future_vol_{h}' if name in risk_features else
                    f'future_illiquidity_{h}' if name == 'amihud_20' else f'future_amount_{h}')
            for code,z in joined.groupby('code'):
                v=z[[name,target]].dropna()
                ic=rank_corr(v[name],v[target]) if len(v)>=20 else np.nan
                early=v.loc[(v.index.get_level_values('date')<split)&(z.loc[v.index,f'exit_date_{h}']<split)]
                late=v.loc[v.index.get_level_values('date')>=split]
                risk.append({'code':code,'factor':name,'horizon':h,'target':target,'observations':len(v),'rank_correlation':ic,
                             'early_correlation':rank_corr(early[name],early[target]) if len(early)>=20 else np.nan,
                             'late_correlation':rank_corr(late[name],late[target]) if len(late)>=20 else np.nan})
    pd.DataFrame(risk).to_csv(out/'risk_liquidity_diagnostics.csv',index=False)
    redundancy=[]
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            corr=[rank_corr(z[a],z[b]) for _,z in f.groupby('code')]
            if all(pd.notna(x) and abs(x)>=.95 for x in corr):
                redundancy.append({'factor_a':a,'factor_b':b,'mean_within_asset_rank_correlation':np.mean(corr)})
    pd.DataFrame(redundancy).to_csv(out/'redundancy.csv',index=False)
    # Offline portfolio-free descriptive snapshots.
    latest=pd.read_csv(root/'latest.csv')
    latest.to_csv(out/'latest_snapshot.csv',index=False)
    profile=[]
    for code,d in prices.items():
        r=np.log(d.close/d.close.shift())
        wealth=d.close/d.close.iloc[0]
        profile.append({'code':code,'start':str(d.index.min().date()),'end':str(d.index.max().date()),'days':len(d),
                        'interval_price_return':wealth.iloc[-1]-1,'interval_annualized_vol':r.std()*np.sqrt(252),
                        'max_close_drawdown':(wealth/wealth.cummax()-1).min(),'mean_daily_amount':d.amt.mean()})
    pd.DataFrame(profile).to_csv(out/'asset_profile.csv',index=False)
    meta={'input':str(root.resolve()),'input_sha256':hashlib.sha256((root/'factors.csv').read_bytes()).hexdigest(),
          'source':'existing frozen Wind export, no new prices fetched','dates':len(dates),'assets':list(codes),
          'split_date':str(split.date()),'early_sample':'labels exiting on or after split excluded',
          'label':'factor at t close; enter t+1 adjusted open, exit t+h adjusted close; no trading costs',
          'ic':'daily cross-sectional Spearman with all 3 assets; no pooled cross-asset correlation',
          'nonoverlap':'dates[::h] anchored on full calendar; no phase optimization',
          'inference':'descriptive only; no significance claim; overlapping labels, multiple tests, three-asset industry confounding',
          'excluded':'unverified PIT fundamentals; <40 observations per asset; constant factors',
          'files':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in out.glob('*.csv')}}
    (out/'analysis_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/factor_library/wind_20261007'))
    p.add_argument('--out',type=Path,default=Path('outputs/factor_library/analysis_20261007'))
    a=p.parse_args();r=diagnostics(a.input,a.out)
    for h in (5,20):
        print('HORIZON',h)
        z=r[r.horizon==h].copy();z['abs_ic']=z.mean_rank_ic.abs()
        print(z.sort_values('abs_ic',ascending=False).head(12).drop(columns='abs_ic').to_string(index=False))

if __name__=='__main__':main()
