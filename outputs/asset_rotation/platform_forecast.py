"""Prequential multi-horizon learning from the expanded price-factor library.
Only matured next-close forward labels train models or authorize their alpha.
Unverified valuation/financial/flow vintages never enter these forecasts.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from forecast_adaptive import mean_se
from factor_practice import load_market,PracticeSettings,OUT

FEATURES=['cash_relative_13612U','cash_relative_13612W','skip_12_1','skip_6_1','risk_adjusted_momentum','path_efficiency','enhanced_sharpe_maxr','enhanced_path_mstd','weighted_trend25','weighted_trend63','bias_velocity','momentum_acceleration','relative_strength','market_residual_momentum','low_volatility','dip_repair','rsrs','rsrs_dampened']
HORIZONS=[4,13,26];NAMES=['ridge','tree']

def build_panel(d,h):
    p=d['prices'];ret=d['returns'];vol=ret.rolling(63,min_periods=63).std().clip(lower=1e-5);positions=d['positions'];rows=[]
    for j,pos in enumerate(positions):
        if pos<252 or pos+1>=len(p):continue
        fill=pos+1;end=int(positions[j+h])+1 if j+h<len(positions) else None
        if end is not None and end>=len(p):end=None
        if end is None:target=pd.Series(np.nan,index=p.columns)
        else:target=p.iloc[end]/p.iloc[fill]/np.prod(1+d['cash'].iloc[fill+1:end+1])-1
        for code in p:
            if not np.isfinite(p.iloc[pos][code]) or not np.isfinite(vol.iloc[pos][code]):continue
            row={'date':p.index[pos],'code':code,'horizon_weeks':h,'fill_date':p.index[fill],'label_end':p.index[end] if end is not None else pd.NaT,'target':target[code],'scale':vol.iloc[pos][code]*np.sqrt(5*h)}
            row.update({k:d['factors'][k].iloc[pos][code] for k in FEATURES if k in d['factors']})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['date','code']).reset_index(drop=True)

def gate(mature,name,h):
    use=mature.dropna(subset=[name,'target','prior','scale'])
    use=use.loc[use.date.isin(sorted(use.date.unique())[-104:])]
    if use.empty:return 0.,np.nan,np.nan
    z=(use.target-use.prior)/use.scale;a=(use[name]-use.prior)/use.scale
    week=pd.DataFrame({'date':use.date,'gain':z*z-(z-a)**2,'base':z*z,'cross':a*z,'alpha2':a*a}).groupby('date').mean()
    r2=float(week.gain.sum()/max(week.base.sum(),1e-12))
    if len(week)<max(52,2*h):return 0.,r2,np.nan
    rng=np.random.default_rng(7307+h);n=len(week);starts=rng.integers(0,n,(256,int(np.ceil(n/h))));idx=((starts[:,:,None]+np.arange(h))%n).reshape(256,-1)[:,:n]
    lcb=float(np.quantile(week.gain.to_numpy()[idx].mean(axis=1),.01/6))
    calibration=float(np.clip(week.cross.sum()/max(week.alpha2.sum(),1e-12),0,1))
    return calibration if lcb>0 and r2>0 else 0.,r2,lcb

def walk_forward(panel,h):
    panel=panel.copy();panel['prior']=np.nan;panel['prior_se']=np.nan
    for n in NAMES:panel[n]=np.nan
    panel['mu']=np.nan;panel['rho']=np.nan;panel['selected']='prior';panel['signal_strength']=0.
    dates=sorted(panel.date.unique());models={};lastfit=-999;train_end=pd.NaT
    for j,date in enumerate(dates):
        group=panel.loc[panel.date==date];mature=panel.loc[panel.label_end.notna()&(panel.label_end<=date)&panel.target.notna()]
        train_dates=sorted(mature.date.unique())[-260:];train=mature.loc[mature.date.isin(train_dates)]
        for code,ix in group.groupby('code').groups.items():
            past=train.loc[train.code==code,'target'].to_numpy();mean,se=mean_se(past,lags=h)
            if not np.isfinite(se):mean=0.;se=.03*np.sqrt(h)
            mean*=len(past)/(len(past)+52)
            panel.loc[ix,'prior']=mean;panel.loc[ix,'prior_se']=se
        group=panel.loc[group.index]
        if len(train_dates)>=104 and (j-lastfit>=13 or not models):
            fitted=train.dropna(subset=['prior','scale'])
            y=np.clip((fitted.target-fitted.prior)/fitted.scale,-6,6)
            models={'ridge':make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=100.)), 'tree':make_pipeline(SimpleImputer(keep_empty_features=True),HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=10.,learning_rate=.04,early_stopping=False,random_state=7107))}
            for n,model in models.items():model.fit(fitted[FEATURES],y)
            train_end=fitted.label_end.max();lastfit=j
        for n,model in models.items():panel.loc[group.index,n]=group.prior+model.predict(group[FEATURES])*group.scale
        mature=panel.loc[panel.label_end.notna()&(panel.label_end<=date)&panel.target.notna()]
        gates={n:gate(mature,n,h) for n in models};accepted=[n for n,x in gates.items() if x[0]>0]
        selected=max(accepted,key=lambda n:gates[n][1]) if accepted else 'prior';strength=gates[selected][0] if accepted else 0.
        pred=panel.loc[group.index,selected] if selected!='prior' else group.prior
        panel.loc[group.index,'mu']=group.prior+strength*(pred-group.prior)
        panel.loc[group.index,'rho']=1.645*group.prior_se
        panel.loc[group.index,'selected']=selected;panel.loc[group.index,'signal_strength']=strength
        panel.loc[group.index,'train_label_end']=train_end;panel.loc[group.index,'evidence_label_end']=mature.label_end.max()
        for n,(strength_n,r2,lcb) in gates.items():
            panel.loc[group.index,n+'_gate_strength']=strength_n;panel.loc[group.index,n+'_evidence_r2']=r2;panel.loc[group.index,n+'_gain_lcb']=lcb
        if j%150==0:print('learn',h,j,str(pd.Timestamp(date).date()),selected,flush=True)
    return panel

def summarize(panel,market):
    rows=[]
    for segment,mask in [('all_explored',panel.date>=pd.Timestamp('2017-01-01')),('2017_2021_explored',(panel.date>=pd.Timestamp('2017-01-01'))&(panel.date<pd.Timestamp('2022-01-01'))),('2022_2026_explored',panel.date>=pd.Timestamp('2022-01-01'))]:
        for name in ['prior','ridge','tree','mu']:
            valid=panel.loc[mask].dropna(subset=[name,'target','prior','scale'])
            if valid.empty:continue
            y=valid.target.to_numpy();pred=valid[name].to_numpy();prior=valid.prior.to_numpy();scale=valid.scale.to_numpy()
            gains=((y-prior)/scale)**2-((y-pred)/scale)**2
            dates=pd.Series(gains,index=valid.index).groupby(valid.date).mean()
            mean,se=mean_se(dates,lags=int(valid.horizon_weeks.iloc[0]))
            ics=[]
            for _,g in valid.groupby('date'):
                if len(g)>=5 and g[name].nunique()>1 and g.target.nunique()>1:ics.append(float(spearmanr(g[name],g.target).statistic))
            rows.append({'market':market,'segment':segment,'horizon_weeks':int(valid.horizon_weeks.iloc[0]),'model':name,'rows':len(valid),'weeks':valid.date.nunique(),'r2_vs_prior':1-np.mean((y-pred)**2)/max(np.mean((y-prior)**2),1e-16),'normalized_r2_vs_prior':float(np.sum(gains)/max(np.sum(((y-prior)/scale)**2),1e-16)),'normalized_gain_hac_low':mean-1.96*se,'mean_rank_ic':np.mean(ics) if ics else np.nan,'rmse':np.sqrt(np.mean((y-pred)**2)),'gate_accepted_week_fraction':float(valid.groupby('date').signal_strength.first().gt(0).mean()),'latest_gate_selected':panel.iloc[-1].selected})
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--markets',nargs='+',default=['CN','US']);args=parser.parse_args();all_stats=[]
    for market in args.markets:
        d=load_market(market);dest=d['dest']/'learning';dest.mkdir(exist_ok=True)
        for h in HORIZONS:
            panel=walk_forward(build_panel(d,h),h);panel.to_csv(dest/f'forecasts_{h}w.csv',index=False)
            all_stats+=summarize(panel,market);print('forecast done',market,h,flush=True)
    pd.DataFrame(all_stats).to_csv(OUT/'forecast_metrics.csv',index=False)
    (OUT/'forecast_method.json').write_text(json.dumps({'features':FEATURES,'horizons_weeks':HORIZONS,'models':NAMES,'min_mature_train_weeks':104,'train_weeks':260,'refit_weeks':13,'gate':'Only prior prequential errors with known endpoints; whole-week blocks length=horizon, minimum52 weeks; conservative lower gain bound across6 forecast families','prior':'Causal per-asset forward cash-excess mean with sample-count shrinkage and overlap-aware HAC mean uncertainty','target':'next trading-day execution close to h-week future execution close; actual compounded cash-relative return','price_scores_are_not_expected_returns':True,'valuation_quality_flow_production_excluded':True,'history_status':'All segments exploratory; no untouched test claim','date_local':'2026-10-07'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
