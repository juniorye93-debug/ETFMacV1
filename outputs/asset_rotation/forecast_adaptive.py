"""Causal weekly forecasts with prequential model selection and conservative alpha gates.
Literature-inspired adaptation, not a replication of Gu/Kelly/Xiu's US-stock experiment.
Every target is observed only at the NEXT execution close, never at the next signal close.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class ForecastSettings:
    min_asset_sessions:int=65
    min_train_weeks:int=104
    train_weeks:int=260
    refit_weeks:int=4
    evidence_weeks:int=104
    min_evidence_weeks:int=52
    block_weeks:int=4
    bootstrap_draws:int=1024
    family_error:float=.05
    annual_weeks:float=52.
    seed:int=7241

MODEL_NAMES=('ridge_100','ridge_1000','tree')
FEATURES=['mom4','mom13','mom26','mom52','reversal1','vol_ratio','draw13','draw52','skew13','market_mom4','market_vol_ratio']

def signal_positions(index):
    periods=index.to_period('W-FRI')
    return np.flatnonzero(np.r_[True,periods[1:]!=periods[:-1]])

def mean_se(values,lags=4):
    """Newey-West SE of the sample mean; no claimed posterior probability."""
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if n<8:return 0.,np.inf
    m=float(x.mean());z=x-m;var=float(z@z/n)
    lrv=var
    for k in range(1,min(lags,n-1)+1):lrv+=2*(1-k/(lags+1))*float(z[k:]@z[:-k]/n)
    return m,float(np.sqrt(max(lrv,var*.25,1e-16)/n))

def prior_estimate(y):
    m,se=mean_se(y)
    if not np.isfinite(se):return 0.,.02
    shrink=max(0.,1-se*se/max(m*m,1e-16))
    return m*shrink,se

def build_panel(prices,cash,s=ForecastSettings()):
    """Features at t close; target: fill(t+1) -> next weekly fill, net cash.
    Five-session-equivalent target removes holiday-length differences, not realized P&L.
    New ETFs need only min_asset_sessions; longer features remain missing, never backfilled.
    """
    ret=prices.pct_change(fill_method=None);excess=ret.sub(cash,axis=0)
    logex=np.log1p(ret).sub(np.log1p(cash),axis=0)
    vol=excess.ewm(halflife=32.5,min_periods=s.min_asset_sessions).std().clip(lower=1e-5)
    short=excess.ewm(halflife=10,min_periods=20).std()
    features={}
    for name,n in [('mom4',20),('mom13',65),('mom26',130),('mom52',260),('reversal1',5)]:
        features[name]=logex.rolling(n,min_periods=n).sum()/(vol*np.sqrt(n))
    features['vol_ratio']=short/vol
    features['draw13']=prices/prices.rolling(65,min_periods=20).max()-1
    features['draw52']=prices/prices.rolling(260,min_periods=65).max()-1
    features['skew13']=excess.rolling(65,min_periods=30).skew()
    marketmom=features['mom4'].median(axis=1);marketvol=features['vol_ratio'].median(axis=1)
    positions=signal_positions(prices.index);rows=[]
    for j,pos in enumerate(positions):
        fill=pos+1
        if fill>=len(prices):continue
        end=positions[j+1]+1 if j+1<len(positions) else len(prices)
        if end>=len(prices):end=None
        cash_target=np.prod(1+cash.iloc[fill+1:end+1]) -1 if end is not None else np.nan
        for code in prices:
            scale=float(vol.iloc[pos][code]*np.sqrt(5))
            if not np.isfinite(scale) or not np.isfinite(prices.iloc[pos][code]):continue
            target=(prices.iloc[end][code]/prices.iloc[fill][code]-1-cash_target)*5/(end-fill) if end is not None else np.nan
            row={'date':prices.index[pos],'code':code,'label_end':prices.index[end] if end is not None else pd.NaT,'target':target,'scale':scale}
            row.update({name:float(frame.iloc[pos][code]) for name,frame in features.items()})
            row.update(market_mom4=float(marketmom.iloc[pos]),market_vol_ratio=float(marketvol.iloc[pos]))
            rows.append(row)
    return pd.DataFrame(rows)

def fit_model(name,X,y):
    if name.startswith('ridge'):
        model=make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=float(name.split('_')[1])))
    else:
        model=make_pipeline(SimpleImputer(add_indicator=True),HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=10.,learning_rate=.04,early_stopping=False,random_state=7241))
    return model.fit(X,np.clip(y,-6,6))

def evidence_gate(frame,name,s):
    """Compare normalized squared errors against the causal, shrunk mean prior.
    Resample whole weeks in blocks; correlated ETFs never count as independent weeks.
    Return signal shrinkage, not probability of profit or a confidence level.
    """
    use=frame.dropna(subset=[name,'target','prior'])
    if use.empty:return 0.,{'weeks':0,'r2':np.nan,'loss_gain_lcb':np.nan,'calibration':0.}
    use=use.loc[use.date.isin(sorted(use.date.unique())[-s.evidence_weeks:])]
    z=(use.target-use.prior)/use.scale;a=(use[name]-use.prior)/use.scale
    tmp=pd.DataFrame({'date':use.date,'gain':z*z-(z-a)**2,'cross':a*z,'alpha2':a*a,'base2':z*z})
    week=tmp.groupby('date').mean();n=len(week)
    r2=float(week.gain.sum()/max(week.base2.sum(),1e-12))
    calibration=float(np.clip(week.cross.sum()/max(week.alpha2.sum(),1e-12),0,1))
    if n<s.min_evidence_weeks:return 0.,{'weeks':n,'r2':r2,'loss_gain_lcb':np.nan,'calibration':calibration}
    rng=np.random.default_rng(s.seed);starts=rng.integers(0,n,(s.bootstrap_draws,int(np.ceil(n/s.block_weeks))))
    idx=(starts[:,:,None]+np.arange(s.block_weeks))%n;idx=idx.reshape(s.bootstrap_draws,-1)[:,:n]
    lcb=float(np.quantile(week.gain.to_numpy()[idx].mean(axis=1),s.family_error/len(MODEL_NAMES)))
    slope_lcb=float(np.quantile(week.cross.to_numpy()[idx].sum(axis=1)/np.maximum(week.alpha2.to_numpy()[idx].sum(axis=1),1e-12),s.family_error/len(MODEL_NAMES)))
    strength=float(np.clip(min(calibration,slope_lcb),0,1)) if lcb>0 and r2>0 else 0.
    return strength,{'weeks':n,'r2':r2,'loss_gain_lcb':lcb,'calibration':calibration,'calibration_lcb':slope_lcb}

def prequential_forecasts(panel,s=ForecastSettings(),progress=False):
    panel=panel.sort_values(['date','code']).reset_index(drop=True).copy()
    # These priors use only targets whose execution endpoint has passed at feature time.
    for date,group in panel.groupby('date',sort=True):
        past=panel.loc[panel.label_end.notna()&(panel.label_end<=date)]
        for code,idx in group.groupby('code').groups.items():
            mean,se=prior_estimate(past.loc[past.code==code,'target'])
            panel.loc[idx,'prior']=mean;panel.loc[idx,'prior_se']=se
    panel['selected']='prior';panel['signal_strength']=0.;panel['mu']=panel.prior;panel['rho']=norm.ppf(.95)*panel.prior_se
    for name in MODEL_NAMES:panel[name]=np.nan
    models={};lastfit=-999;train_end=pd.NaT
    for j,(date,group) in enumerate(panel.groupby('date',sort=True)):
        mature=panel.loc[panel.label_end.notna()&(panel.label_end<=date)&panel.target.notna()].copy()
        train_dates=sorted(mature.date.unique())[-s.train_weeks:];train=mature.loc[mature.date.isin(train_dates)]
        if len(train_dates)>=s.min_train_weeks and (j-lastfit>=s.refit_weeks or not models):
            y=(train.target-train.prior)/train.scale
            models={name:fit_model(name,train[FEATURES],y) for name in MODEL_NAMES}
            lastfit=j;train_end=train.label_end.max()
        for name,model in models.items():
            pred=group.prior.to_numpy()+model.predict(group[FEATURES])*group.scale.to_numpy()
            panel.loc[group.index,name]=pred
        gates={name:evidence_gate(mature,name,s) for name in models}
        accepted=[name for name,(strength,info) in gates.items() if strength>0]
        chosen=max(accepted,key=lambda name:gates[name][1]['r2']) if accepted else 'prior'
        global_strength=gates[chosen][0] if chosen!='prior' else 0.
        for idx,row in group.iterrows():
            own=mature.loc[mature.code==row.code].dropna(subset=[chosen]) if chosen!='prior' else pd.DataFrame()
            own=own.tail(s.evidence_weeks)
            reliability=0.;rho=float(norm.ppf(.95)*row.prior_se)
            if len(own)>=s.min_evidence_weeks:
                z=(own.target-own.prior)/own.scale;a=(own[chosen]-own.prior)/own.scale
                r2=1-float(((z-a)**2).sum()/max((z*z).sum(),1e-12))
                reliability=float(np.clip(r2,0,1)*len(own)/(len(own)+s.min_evidence_weeks))
                # Model disagreement enlarges uncertainty; no self-reported tree confidence.
                disagreement=float(np.std([panel.loc[idx,name] for name in models]))
                rho+=global_strength*disagreement
            strength=global_strength*reliability
            mu=float(row.prior+strength*(panel.loc[idx,chosen]-row.prior)) if chosen!='prior' else float(row.prior)
            panel.loc[idx,['selected','signal_strength','mu','rho','train_label_end','evidence_label_end']]=[chosen,strength,mu,rho,train_end,mature.label_end.max()]
        for name,(strength,info) in gates.items():
            for key,val in info.items():panel.loc[group.index,f'{name}_{key}']=val
        if progress and j%100==0:print('forecast weeks',j,str(date.date()),chosen,'strength',round(global_strength,3),flush=True)
    return panel

def forecast_metrics(frame):
    use=frame.loc[frame.target.notna()].copy();rows=[]
    for name in ['zero','prior',*MODEL_NAMES,'mu']:
        valid=use if name=='zero' else use.dropna(subset=[name])
        if valid.empty:continue
        pred=np.zeros(len(valid)) if name=='zero' else valid[name].to_numpy()
        y=valid.target.to_numpy();prior=valid.prior.to_numpy();scale=valid.scale.to_numpy()
        mse=np.mean((y-pred)**2);base=np.mean((y-prior)**2);normmse=np.mean(((y-pred)/scale)**2);normbase=np.mean(((y-prior)/scale)**2)
        rows.append({'model':name,'rows':len(valid),'weeks':valid.date.nunique(),'start':str(valid.date.min().date()),'end':str(valid.date.max().date()),'rmse_weekly':float(np.sqrt(mse)),'r2_vs_prior':float(1-mse/max(base,1e-16)),'normalized_r2_vs_prior':float(1-normmse/max(normbase,1e-16)),'sign_accuracy':float(np.mean(np.sign(pred[np.abs(pred)>1e-8])==np.sign(y[np.abs(pred)>1e-8]))) if np.any(np.abs(pred)>1e-8) else np.nan,'nonzero_prediction_fraction':float(np.mean(np.abs(pred)>1e-8))})
    return pd.DataFrame(rows)

def main():
    import argparse,sys
    from rotation import read_prices
    from rotation_v2 import lagged_cash
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser();parser.add_argument('--market',required=True,choices=['CN','US']);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'forecasts.csv').exists():raise ValueError('refusing to overwrite forecasts')
    assets=json.loads((root/'config.json').read_text())['markets'][args.market]['assets']
    prices=read_prices(root/'data'/('cn_expanded28_20260924.csv' if args.market=='CN' else 'us_expanded_20260924.csv'),assets);cash,_,_=lagged_cash(prices,args.market)
    settings=ForecastSettings();panel=build_panel(prices,cash,settings);forecasts=prequential_forecasts(panel,settings,progress=True)
    forecasts.to_csv(args.out/'forecasts.csv',index=False);forecast_metrics(forecasts).to_csv(args.out/'forecast_metrics.csv',index=False)
    (args.out/'forecast_settings.json').write_text(json.dumps(asdict(settings),indent=2))
    print(forecast_metrics(forecasts).to_string(index=False),flush=True)
if __name__=='__main__':main()
