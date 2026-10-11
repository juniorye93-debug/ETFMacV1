"""Fixed-signal ranking diagnostic; no strategy selection or portfolio backtest.

Forward labels start at t+1 execution close. Overlap-aware block intervals do
not make an inspected historical segment an untouched test set.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from equity_signals import SignalSettings,price_signals,valuation_signals,manifest
from rotation import read_prices
from rotation_v2 import lagged_cash

ROOT=Path(__file__).resolve().parent
NAMES=['mom_3m','mom_13612U','mom_13612W','skip_12_1','enhanced_sharpe_maxr','enhanced_path_mstd']
HORIZONS=[4,13,26]

def label_at(prices,cash,positions,j,h):
    fill=int(positions[j])+1
    if j+h>=len(positions):return None,None
    end=int(positions[j+h])+1
    if end>=len(prices):return None,None
    cashgrowth=float(np.prod(1+cash.iloc[fill+1:end+1]))
    return prices.iloc[end]/prices.iloc[fill]/cashgrowth-1,prices.index[end]

def deduplicate(codes,corr,threshold=.90):
    kept=[]
    for c in codes:
        if not any(np.isfinite(corr.loc[c,k]) and corr.loc[c,k]>=threshold for k in kept):kept.append(c)
    return kept

def block_interval(series,h):
    x=np.asarray(series,float);n=len(x)
    if n<2*h:return np.nan,np.nan
    rng=np.random.default_rng(9325+h);draws=4096
    starts=rng.integers(0,n,(draws,int(np.ceil(n/h))))
    idx=((starts[:,:,None]+np.arange(h))%n).reshape(draws,-1)[:,:n]
    means=x[idx].mean(axis=1)
    q=.05/(2*len(NAMES)*len(HORIZONS))
    return tuple(np.quantile(means,[q,1-q]))

def main():
    out=ROOT/'signal_research';assets=json.loads((ROOT/'config.json').read_text())['markets']['CN']['assets']
    allprices=read_prices(ROOT/'data/cn_expanded28_20260924.csv',assets)
    cash,_,cashmeta=lagged_cash(allprices,'CN')
    codes=[c for c,a in assets.items() if a['class']=='equity'];p=allprices[codes]
    settings=SignalSettings();f=price_signals(p,cash,settings)
    period=p.index.to_period('W-FRI');positions=np.flatnonzero(np.r_[True,period[1:]!=period[:-1]])
    returns=p.pct_change(fill_method=None);records=[];feature_rows=[]
    for j,pos in enumerate(positions):
        date=p.index[pos]
        day=pd.DataFrame({name:frame.iloc[pos] for name,frame in f.items()})
        day['name']=[assets[c]['name'] for c in day.index];day['date']=date;day.index.name='code';feature_rows.append(day.reset_index())
        eligible=day[NAMES].dropna().index.tolist()
        if len(eligible)<5:continue
        hist=returns.loc[:date,eligible].tail(252);corr=hist.corr(min_periods=126)
        kept=deduplicate(eligible,corr)
        if len(kept)<5:continue
        for h in HORIZONS:
            target,end=label_at(p,cash,positions,j,h)
            if target is None:continue
            target=target.loc[kept].dropna();use=[c for c in kept if c in target.index]
            if len(use)<5:continue
            for name in NAMES:
                x=day.loc[use,name];y=target.loc[use]
                if x.nunique()<2 or y.nunique()<2:continue
                records.append({'date':date,'label_end':end,'horizon_weeks':h,'signal':name,'asset_count':len(use),'rank_ic':float(spearmanr(x,y).statistic)})
    features=pd.concat(feature_rows,ignore_index=True);features.to_csv(out/'CN_weekly_candidate_signals.csv',index=False)
    latest=features.loc[features.date==features.date.max()].copy()
    vpath=out/'wind_valuation/index_valuation_history.csv'
    if vpath.exists():
        v=pd.read_csv(vpath,parse_dates=['date'])
        pe=v.pivot(index='date',columns='etf',values='pe_ttm');pb=v.pivot(index='date',columns='etf',values='pb_lf')
        val=valuation_signals(pe,pb,settings,allow_unverified_diagnostic=True)
        date=features.date.max()
        for key,frame in val.items():latest[key+'_unverified']=latest.code.map(frame.loc[:date].iloc[-1])
        latest['valuation_pit_verified']=False
    latest.to_csv(out/'CN_latest_signal_diagnostic.csv',index=False)
    detailed=pd.DataFrame(records);detailed.to_csv(out/'CN_rank_ic_observations.csv',index=False)
    stats=[]
    for segment,mask in [('2016_2021_explored',detailed.date<pd.Timestamp('2022-01-01')),('2022_2026_explored',detailed.date>=pd.Timestamp('2022-01-01'))]:
        for (signal,h),g in detailed.loc[mask].groupby(['signal','horizon_weeks']):
            lo,hi=block_interval(g.rank_ic,h)
            stats.append({'segment':segment,'signal':signal,'horizon_weeks':int(h),'weeks':len(g),'first':str(g.date.min().date()),'last_signal':str(g.date.max().date()),'last_label':str(g.label_end.max().date()),'mean_rank_ic':float(g.rank_ic.mean()),'block_ci_lo':float(lo),'block_ci_hi':float(hi),'positive_week_fraction':float((g.rank_ic>0).mean()),'mean_asset_count':float(g.asset_count.mean())})
    pd.DataFrame(stats).to_csv(out/'CN_rank_ic_summary.csv',index=False)
    m=manifest(settings)
    m.update({'price_asof':str(p.index[-1].date()),'latest_weekly_signal':str(features.date.max().date()),'equity_assets':len(codes),'cash':cashmeta,'diagnostic':'cross-sectional Spearman IC for fixed pre-specified signals; not portfolio returns or calibrated expected returns','horizons_weeks':HORIZONS,'inference':'4096 circular block resamples; block = label horizon; Bonferroni interval over 18 signal/horizon hypotheses within each segment; exploratory, not acceptance test','dedup':'prior-only 252-session correlation, minimum126 pairs, threshold0.90, positive only, config priority; diagnostic defaults, not user drawdown constraints','known_limits':['current surviving ETF pool was selected after history; not a historical membership database','2016-2026 prices previously inspected; both segments exploratory','valuation vintage and benchmark-history unverified; excluded from IC and production forecast','IC does not measure absolute return accuracy, transaction costs or achievable drawdown'],'source_code_sha256':hashlib.sha256((ROOT/'equity_signals.py').read_bytes()).hexdigest()})
    (out/'signal_diagnostic_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
    print(json.dumps({'rows':len(features),'ic_observations':len(detailed),'latest_signal':m['latest_weekly_signal'],'price_asof':m['price_asof'],'equity_assets':len(codes)},ensure_ascii=False))

if __name__=='__main__':main()
