"""Weekly robust MPC adaptation with actual-account drawdown feedback.
No historical 252-day drawdown constraint, class caps, breadth gate or asset cap.
Forecast skill is earned prequentially; forward scenario checks are not guarantees.
"""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import cvxpy as cp
from rotation import read_prices,transact,metrics,Settings
from rotation_v2 import lagged_cash
from forecast_adaptive import signal_positions

@dataclass(frozen=True)
class AdaptiveSettings:
    max_drawdown:float
    gamma0:float=5.
    planning_weeks:int=4
    scenario_weeks:int=13
    scenario_paths:int=192
    scenario_confidence:float=.95
    scenario_block_sessions:int=5
    scenario_history_sessions:int=1040
    student_df:float=5.
    min_asset_sessions:int=65
    covariance_half_lives_weeks:tuple=(6.,13.,26.,52.)
    covariance_evidence_weeks:int=52
    covariance_min_evidence_weeks:int=26
    correlation_threshold:float=.90
    cost_rate:float=.0008
    annual_weeks:float=52.
    annual_sessions:float=252.
    scale_grid_points:int=21
    seed:int=4983
    def __post_init__(self):
        if not 0<self.max_drawdown<1:raise ValueError('D must lie in (0,1)')
        if min(self.gamma0,self.planning_weeks,self.scenario_weeks,self.scenario_paths,self.min_asset_sessions)<=0:raise ValueError('invalid settings')
        if not 0<self.scenario_confidence<1 or self.student_df<=2 or not 0<self.correlation_threshold<=1:raise ValueError('invalid risk settings')
        if self.scenario_weeks<self.planning_weeks:raise ValueError('scenario horizon must cover plan')

@dataclass(frozen=True)
class Account:
    nav:float
    peak:float
    def __post_init__(self):
        if not 0<self.nav<=self.peak*(1+1e-10):raise ValueError('invalid account wealth/peak')
    @property
    def drawdown(self):return max(0.,1-self.nav/self.peak)
    def cushion(self,D):return (D-self.drawdown)/(1-self.drawdown)

def dynamic_gamma(account,s):
    return s.gamma0*s.max_drawdown/max(s.max_drawdown-account.drawdown,1e-6)

def psd_correlation(corr):
    corr=(corr+corr.T)/2;values,vectors=np.linalg.eigh(corr)
    c=(vectors*np.maximum(values,1e-5))@vectors.T;d=np.sqrt(np.diag(c));return c/np.outer(d,d)

def covariance_estimate(history,half_life_weeks,min_sessions=65):
    """Pairwise causal EW covariance. Shrink CORRELATION, preserving marginal scales.
    Do not shrink bond volatility towards average equity volatility.
    """
    a=history.to_numpy(dtype=float);mask=np.isfinite(a)
    if (mask.sum(axis=0)<min_sessions).any():raise ValueError('insufficient covariance history')
    ages=np.arange(len(a)-1,-1,-1);weights=np.exp2(-ages/(5*half_life_weeks));weights/=weights.sum()
    available=mask*weights[:,None];means=(np.nan_to_num(a)*weights[:,None]).sum(axis=0)/available.sum(axis=0)
    x=np.where(mask,a-means,0.);denom=mask.T.astype(float)@available
    cov=(x*weights[:,None]).T@x/np.maximum(denom,1e-16)
    variance=np.maximum(np.diag(cov),1e-10);sd=np.sqrt(variance);corr=np.clip(cov/np.outer(sd,sd),-1,1)
    neff=available.sum(axis=0)**2/np.maximum((available**2).sum(axis=0),1e-16)
    delta=float(np.clip(len(variance)/max(neff.min(),1),.02,.5))
    rawcorr=psd_correlation(corr)
    corr=psd_correlation((1-delta)*rawcorr+delta*np.eye(len(sd)))
    return corr*np.outer(sd,sd),rawcorr,{'effective_observations_min':float(neff.min()),'correlation_shrinkage':delta}

def representatives(corr,codes,threshold):
    keep=[];rows=[]
    for i,c in enumerate(codes):
        matches=[j for j in keep if corr[i,j]>=threshold]
        rep=max(matches,key=lambda j:corr[i,j]) if matches else i
        rows.append({'code':c,'representative':codes[rep],'kept':not bool(matches),'correlation':float(corr[i,rep]),'threshold':threshold})
        if not matches:keep.append(i)
    return keep,pd.DataFrame(rows)

class RiskForecaster:
    """Choose covariance memory using only matured PREQUENTIAL Gaussian scores.
    Scores use identical asset sets across candidate memories and whole weekly outcomes.
    """
    def __init__(self,s):self.s=s;self.pending=[];self.scores=[];self.audit=[]
    def update(self,prices,cash,date):
        ready=[p for p in self.pending if p['end']<=date]
        self.pending=[p for p in self.pending if p['end']>date]
        for p in ready:
            cashret=np.prod(1+cash.loc[(cash.index>p['fill'])&(cash.index<=p['end'])])-1
            n=len(prices.loc[(prices.index>p['fill'])&(prices.index<=p['end'])])
            realized=(prices.loc[p['end'],p['codes']]/prices.loc[p['fill'],p['codes']]-1-cashret).to_numpy()*np.sqrt(5/n)
            for half,cov in p['covs'].items():
                sign,ld=np.linalg.slogdet(cov);quad=float(realized@np.linalg.solve(cov,realized))
                if sign<=0:raise ValueError('non-positive covariance')
                self.scores.append({'date':p['signal'],'available':p['end'],'half_life':half,'score':float((ld+quad)/len(realized)),'variance_ratio':quad/len(realized)})
    def predict(self,history,codes):
        covs={};corrs={};metas={}
        for half in self.s.covariance_half_lives_weeks:
            cov,corr,meta=covariance_estimate(history[codes],half,self.s.min_asset_sessions)
            covs[half]=cov;corrs[half]=corr;metas[half]=meta
        records=pd.DataFrame(self.scores)
        chosen=13. if 13. in covs else list(covs)[0];count=0;inflation=1.
        if not records.empty:
            records=records.loc[records.date.isin(sorted(records.date.unique())[-self.s.covariance_evidence_weeks:])]
            count=records.date.nunique()
            if count>=self.s.covariance_min_evidence_weeks:chosen=float(records.groupby('half_life').score.mean().idxmin())
            use=records.loc[records.half_life==chosen];inflation=max(1.,float(np.sqrt(use.variance_ratio.mean())))
        selected=covs[chosen]*inflation**2
        return selected,corrs[chosen],{'covariance_half_life_weeks':chosen,'covariance_evidence_weeks':count,'covariance_inflation':inflation,**metas[chosen]},covs
    def queue(self,signal,fill,end,codes,covs):
        if end is not None:self.pending.append({'signal':signal,'fill':fill,'end':end,'codes':codes,'covs':{k:v*5 for k,v in covs.items()}})

def make_scenarios(history,cov,cash_daily,mu_lower,s,date_seed):
    """Joint paths: synchronized block resampling + multivariate Student-t innovations.
    History has no invented pre-listing returns; use only complete common rows.
    Positive expected excess is excluded from the protective scenario drift.
    """
    rng=np.random.default_rng(s.seed+date_seed);n=len(cov);days=1+5*s.scenario_weeks;paths=s.scenario_paths
    sd=np.sqrt(np.diag(cov));drift=np.minimum(np.asarray(mu_lower),0)/5+cash_daily
    common=history.dropna().tail(s.scenario_history_sessions).to_numpy()
    if len(common)<s.min_asset_sessions:raise ValueError('scenario common history insufficient')
    centered=common-common.mean(axis=0);hsd=np.maximum(centered.std(axis=0,ddof=1),1e-5)
    standardized=centered/hsd
    block=s.scenario_block_sessions;starts=rng.integers(0,len(common),(paths//2,int(np.ceil(days/block))))
    idx=((starts[:,:,None]+np.arange(block))%len(common)).reshape(paths//2,-1)[:,:days]
    bootstrap=standardized[idx]*sd
    chol=np.linalg.cholesky(cov+np.eye(n)*1e-14)
    normals=rng.normal(size=(paths-paths//2,days,n))@chol.T
    mixing=np.sqrt((s.student_df-2)/rng.chisquare(s.student_df,size=(paths-paths//2,days,1)))
    logshocks=np.concatenate([bootstrap,normals*mixing],axis=0)+drift-.5*sd**2
    return np.expm1(logshocks)

def scenario_drawdowns(plan,paths,current,excluded,account,cash_daily,s,latency_growth=None,latency_excluded=None):
    """Compound wealth with drifting holdings, costs and original global account peak.
    First simulated close is execution latency on OLD positions. Future plans execute
    every five sessions; beyond the MPC horizon the last plan is maintained.
    """
    m,days,n=paths.shape;wealth=np.ones(m);peaks=np.full(m,account.peak/account.nav)
    holdings=np.broadcast_to(current,(m,n)).copy();worst=np.full(m,account.drawdown)
    def step(k):
        nonlocal wealth,peaks,holdings,worst
        v=holdings*(1+paths[:,k]);growth=v.sum(axis=1)+(1-holdings.sum(axis=1))*(1+cash_daily)
        wealth*=growth;holdings=v/growth[:,None];peaks=np.maximum(peaks,wealth);worst=np.maximum(worst,1-wealth/peaks)
    if latency_growth is None:step(0)
    else:
        wealth*=latency_growth;holdings=current*(1+paths[:,0,:])/latency_growth[:,None]
        peaks=np.maximum(peaks,wealth);worst=np.maximum(worst,1-wealth/peaks)
    for k in range(1,days):
        if (k-1)%5==0:
            stage=min((k-1)//5,len(plan)-1);target=plan[stage]
            # Approximation used only for scenario costs. The REAL backtest uses transact().
            cost=s.cost_rate*(np.abs(holdings-target).sum(axis=1)+((latency_excluded if latency_excluded is not None else excluded) if k==1 else 0.))
            wealth*=1-cost;holdings=np.broadcast_to(target,(m,n)).copy()
            worst=np.maximum(worst,1-wealth/peaks)
        step(k)
    return worst

def optimize_plan(mu,rho,cov,current,account,paths,cash_daily,s,liquidation=0.,latency_growth=None,latency_excluded=None):
    """Covariance/cost/uncertainty objective; actual account floor enters future loss CVaR.
    Returns weights over the same codes as paths; absent decision assets are handled outside.
    """
    n=len(mu);H=s.planning_weeks;w=cp.Variable((H,n),nonneg=True);cons=[cp.sum(w,axis=1)<=1]
    if account.drawdown>=s.max_drawdown:
        return np.zeros((H,n)),{'status':'account_limit_breached','gamma':dynamic_gamma(account,s),'scenario_scale':0.,'scenario_mdd_q95':account.drawdown}
    gamma=dynamic_gamma(account,s);covweek=cov*5
    objective=0.;previous=current
    for k in range(H):
        objective+=np.asarray(mu)@w[k]-np.asarray(rho)@w[k]-gamma*cp.quad_form(w[k],cp.psd_wrap(covweek))-s.cost_rate*cp.norm1(w[k]-previous)
        previous=w[k]
    # Actual current peak sets the floor, including old-position execution latency.
    oldgrowth=1+paths[:,0,:]@current+(1-current.sum())*cash_daily if latency_growth is None else latency_growth
    holding=np.prod(1+paths[:,1:6,:],axis=1)-1;cashweek=(1+cash_daily)**5-1
    loss=1-cp.multiply(oldgrowth,1+cashweek+(holding-cashweek)@w[0])
    fee_reserve=s.cost_rate*(cp.norm1(w[0]-current)+liquidation)
    loss+=fee_reserve*np.maximum(oldgrowth,1.)
    q=cp.Variable();u=cp.Variable(len(paths),nonneg=True)
    cons+=[u>=loss-q,q+cp.sum(u)/((1-s.scenario_confidence)*len(paths))<=account.cushion(s.max_drawdown)]
    problem=cp.Problem(cp.Maximize(objective),cons)
    problem.solve(solver='CLARABEL')
    if problem.status not in {'optimal','optimal_inaccurate'} or w.value is None:
        return np.zeros((H,n)),{'status':'latency_risk_infeasible_cash','gamma':gamma,'scenario_scale':0.,'scenario_mdd_q95':np.nan}
    plan=np.maximum(w.value,0)
    if plan.sum(axis=1).max()>1+1e-6:raise RuntimeError('MPC capital budget residual')
    plan/=np.maximum(1,plan.sum(axis=1))[:,None]
    # Re-evaluate full compounded paths; no linearized historical drawdown acceptance.
    # Grid search is explicit: no assumption that drawdown is monotone in scale.
    for scale in np.linspace(1,0,s.scale_grid_points):
        scaled=plan*scale;draws=scenario_drawdowns(scaled,paths,current,liquidation,account,cash_daily,s,latency_growth,latency_excluded)
        quantile=float(np.quantile(draws,s.scenario_confidence))
        if quantile<=s.max_drawdown+1e-8:
            return scaled,{'status':problem.status,'gamma':gamma,'scenario_scale':float(scale),'scenario_mdd_q95':quantile,'scenario_mean_mdd':float(draws.mean()),'forecast_weekly_excess':float(mu@scaled[0]),'robust_weekly_excess':float((mu-rho)@scaled[0]),'forecast_excess_volatility':float(np.sqrt(scaled[0]@covweek@scaled[0]*s.annual_weeks)),'remaining_account_loss_budget':account.cushion(s.max_drawdown)}
    return np.zeros((H,n)),{'status':'scenario_risk_infeasible_cash','gamma':gamma,'scenario_scale':0.,'scenario_mdd_q95':np.nan,'remaining_account_loss_budget':account.cushion(s.max_drawdown)}

def allocate(history,cash,forecast,current,account,risk,s):
    allcodes=list(history);zero=pd.Series(0.,index=allcodes)
    ret=history.pct_change(fill_method=None).sub(cash,axis=0)
    codes=[c for c in allcodes if ret[c].notna().sum()>=s.min_asset_sessions and np.isfinite(history.iloc[-1][c]) and c in forecast.index]
    if not codes:return zero,{'status':'insufficient_history'},zero,pd.DataFrame()
    cov,corr,audit,_=risk.predict(ret,codes)
    kept,dedup=representatives(corr,codes,s.correlation_threshold);selected=[codes[j] for j in kept]
    mu=forecast.loc[codes,'mu'].astype(float).to_numpy();rho=forecast.loc[codes,'rho'].astype(float).to_numpy()
    old=current.loc[codes].to_numpy();liquidation=float(current.drop(codes).sum())
    dailycash=float(cash.iloc[-1]);scenarios=make_scenarios(ret[codes],cov,dailycash,mu-rho,s,history.index[-1].toordinal())
    # Removed assets still contribute their OLD returns over execution latency.
    future=scenarios[:,:,kept].copy()
    oldgrowth=1+scenarios[:,0,:]@old+(1-old.sum())*dailycash
    solve_current=old[kept].copy()
    removed=[j for j in range(len(codes)) if j not in kept]
    liquidated=float(old[removed].sum())
    latency_excluded=(old[removed]*(1+scenarios[:,0,:][:,removed])).sum(axis=1)/oldgrowth if removed else np.zeros(len(scenarios))
    plan,info=optimize_plan(mu[kept],rho[kept],cov[np.ix_(kept,kept)],solve_current,account,future,dailycash,s,liquidated,oldgrowth,latency_excluded)
    zero.loc[selected]=plan[0]
    audit.update(info);audit.update(eligible_count=len(selected),mean_signal_strength=float(forecast.loc[selected,'signal_strength'].mean()),max_signal_strength=float(forecast.loc[selected,'signal_strength'].max()),selected_forecast_models=','.join(sorted(set(forecast.loc[selected,'selected']))),global_account_drawdown=account.drawdown,forecast_target_supported=bool(info.get('robust_weekly_excess',0)*s.annual_weeks>=2*s.max_drawdown),cash=float(1-zero.sum()))
    var=float(plan[0]@cov[np.ix_(kept,kept)]@plan[0]);rc=pd.Series(0.,index=allcodes)
    if var>1e-18:rc.loc[selected]=plan[0]*(cov[np.ix_(kept,kept)]@plan[0])/var
    return zero,audit,rc,dedup

def backtest(prices,cash,assets,forecasts,s,start=504,end=None,progress=False):
    if end is None:end=len(prices)
    if start<s.min_asset_sessions:raise ValueError('insufficient warmup')
    forecasts=forecasts.copy();forecasts['date']=pd.to_datetime(forecasts.date)
    bydate={date:frame.set_index('code') for date,frame in forecasts.groupby('date')}
    risk=RiskForecaster(s);weights=pd.Series(0.,index=prices.columns);pending=None;nav=peak=1.;rows=[];hold=[];signals=[];rcs=[];dedups=[]
    returns=prices.pct_change(fill_method=None);positions=signal_positions(prices.index);nextpos={int(a):int(b) for a,b in zip(positions[:-1],positions[1:])};sigset=set(positions)
    for i in range(start,end):
        date=prices.index[i];previous=nav;r=returns.iloc[i]
        if r.loc[weights>1e-12].isna().any():raise ValueError('missing held return')
        v=weights*(1+r.fillna(0));growth=v.sum()+(1-weights.sum())*(1+cash.iloc[i]);nav*=growth;weights=v/growth
        turnover=cost=0.;executed=None
        if pending is not None:
            target,executed=pending;nav,turnover,cost=transact(nav,weights,target,s.cost_rate);weights=target.copy();pending=None
        peak=max(peak,nav)
        if i in sigset:
            risk.update(prices,cash,date)
            forecast=bydate.get(date,pd.DataFrame(columns=['mu','rho','signal_strength','selected']))
            target,audit,rc,dedup=allocate(prices.iloc[:i+1],cash.iloc[:i+1],forecast,weights,Account(nav,peak),risk,s)
            pending=(target,date);signals.append({'date':date,**audit,**target.to_dict()});rcs.append({'date':date,**rc.to_dict()})
            if not dedup.empty:dedups.append(dedup.assign(date=date))
            # Store ORIGINAL risk forecasts for future evaluation, never recomputed retroactively.
            codes=[c for c in prices if returns.loc[:date,c].notna().sum()>=s.min_asset_sessions and c in forecast.index]
            if codes:
                _,_,_,covs=risk.predict(returns.loc[:date].sub(cash.loc[:date],axis=0),codes)
                nextfill=nextpos.get(i,len(prices))+1;last=prices.index[nextfill] if nextfill<len(prices) else None
                if i+1<len(prices):risk.queue(date,prices.index[i+1],last,codes,covs)
            if progress and len(signals)%100==0:print('MPC weeks',len(signals),str(date.date()),'dd',round(1-nav/peak,4),'exposure',round(target.sum(),3),flush=True)
        rows.append({'date':date,'nav':nav,'return':nav/previous-1,'drawdown':1-nav/peak,'turnover':turnover,'cost_fraction':cost,'exposure':weights.sum(),'cash_return':cash.iloc[i],'executed_signal_date':executed,'halted':False});hold.append({'date':date,**weights.to_dict(),'cash':1-weights.sum()})
    frames=[pd.DataFrame(x).set_index('date') for x in [rows,hold,signals,rcs]]
    return (*frames,pd.concat(dedups,ignore_index=True) if dedups else pd.DataFrame(),pd.DataFrame(risk.scores))

def evaluate(curve,s):
    m=metrics(curve,Settings(s.max_drawdown));cash_cagr=float(np.prod(1+curve.cash_return)**(s.annual_sessions/len(curve))-1)
    m.update(cash_cagr=cash_cagr,annual_excess=m['annual_return']-cash_cagr,required_annual_excess=2*s.max_drawdown,goal_passed=bool(m['max_drawdown']<=s.max_drawdown and m['annual_return']-cash_cagr>=2*s.max_drawdown))
    return m

def calibrate_gamma(prices,cash,assets,forecasts,s,start=130,end=504):
    """Initial, disjoint training period only. Never retune on evaluation NAV.
    Choose highest net cash excess among candidates that meet training drawdown input.
    If none is feasible select least drawdown and flag calibration failure.
    """
    rows=[]
    for g in [2.,5.,10.,20.]:
        ss=replace(s,gamma0=g,scenario_paths=96)
        curve,*_=backtest(prices.iloc[:end],cash.iloc[:end],assets,forecasts.loc[pd.to_datetime(forecasts.date)<prices.index[end]],ss,start=start,end=end)
        rows.append({'gamma0':g,**evaluate(curve,ss)})
    table=pd.DataFrame(rows);valid=table.loc[table.max_drawdown<=s.max_drawdown]
    selected=float(valid.sort_values(['annual_excess','max_drawdown'],ascending=[False,True]).iloc[0].gamma0) if not valid.empty else float(table.sort_values('max_drawdown').iloc[0].gamma0)
    return selected,table

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser();parser.add_argument('--market',required=True,choices=['CN','US']);parser.add_argument('--max-drawdown',required=True,type=float);parser.add_argument('--forecasts',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--gamma0',type=float);args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('refusing to overwrite results')
    args.out.mkdir(parents=True,exist_ok=True)
    assets=json.loads((root/'config.json').read_text())['markets'][args.market]['assets'];file=root/'data'/('cn_expanded28_20260924.csv' if args.market=='CN' else 'us_expanded_20260924.csv')
    prices=read_prices(file,assets);cash,cash_audit,cash_meta=lagged_cash(prices,args.market);forecasts=pd.read_csv(args.forecasts,parse_dates=['date','label_end','train_label_end','evidence_label_end']);s=AdaptiveSettings(args.max_drawdown)
    if args.gamma0 is None:
        selected,calibration=calibrate_gamma(prices,cash,assets,forecasts,s);calibration.to_csv(args.out/'gamma_calibration.csv',index=False);s=replace(s,gamma0=selected)
    else:s=replace(s,gamma0=args.gamma0)
    manifest={'settings':asdict(s),'market':args.market,'currency':cash_meta['currency'],'frequency':'weekly_only','execution':'first trading day of W-FRI week close signal; t+1 close fill; new return begins t+2','data_start':str(prices.index[0].date()),'evaluation_start':str(prices.index[504].date()),'data_end':str(prices.index[-1].date()),'cash_meta':cash_meta,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'forecast_code_sha256':hashlib.sha256((root/'forecast_adaptive.py').read_bytes()).hexdigest(),'prices_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'forecast_sha256':hashlib.sha256(args.forecasts.read_bytes()).hexdigest(),'user_constraints':['adjustable drawdown; weekly; one variable per ETF; full correlation; remove extreme positive duplicates; local-market pool; cash/bond definitions'],'research_assumptions':['long only; no leverage','MPC horizon 4 weeks; risk path horizon 13 weeks','192 forward paths; 95th percentile maximum drawdown stress acceptance; not a guarantee','Student-t log innovations df=5 + synchronized five-session block bootstrap','mean uncertainty is conservative HAC-based radius, not known true expectation','gamma0 calibrated on initial separate period; covariance memory selected using matured forecast scores','no class/asset cap, breadth, trend gate, top-k, cooldown, peak reset or historical 252-day drawdown budget'],'validation':'prequential forecasts and initial training separation, but historical data were previously inspected; not an untouched holdout','paper_status':'adaptation of robust MPC and account drawdown feedback; not a full HMM paper reproduction'}
    (args.out/'run.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(args.out/'rotation_adaptive_snapshot.py').write_bytes(Path(__file__).read_bytes());cash_audit.to_csv(args.out/'cash_audit.csv',index=False)
    result=backtest(prices,cash,assets,forecasts,s,progress=True)
    for name,frame in zip(['curve','holdings','signals','risk_contributions','dedup','covariance_scores'],result):frame.to_csv(args.out/(name+'.csv'),index=name not in ['dedup','covariance_scores'])
    stats=evaluate(result[0],s);(args.out/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
