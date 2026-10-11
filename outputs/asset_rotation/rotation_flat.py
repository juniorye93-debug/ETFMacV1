"""Weekly instrument-level allocation; class labels are reporting metadata only.
Research model: causal covariance + historical linear drawdown constraints, not a
future drawdown guarantee. No breadth, top-k, class/asset caps, trend gate or lockout.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from rotation import read_prices, transact, metrics, Settings, tail_mean
from rotation_v2 import correlation_representatives, lagged_cash
ROOT=Path(__file__).resolve().parent

@dataclass(frozen=True)
class FlatModel:
    max_drawdown: float=.08
    excess_return_multiple: float=2.
    lookback: int=252
    correlation_threshold: float=.90
    forecast_shrinkage: float=.5
    risk_aversion: float=1.
    fee_bps: float=3.
    slippage_bps: float=5.
    cost_horizon_sessions: int=63
    def __post_init__(self):
        if not 0<self.max_drawdown<1 or self.excess_return_multiple<=0:raise ValueError('invalid goals')
        if self.lookback!=252 or not 0<self.correlation_threshold<=1:raise ValueError('invalid estimation settings')
        if not 0<=self.forecast_shrinkage<=1 or self.risk_aversion<=0:raise ValueError('invalid objective settings')
        if min(self.fee_bps,self.slippage_bps)<0 or self.cost_horizon_sessions<=0:raise ValueError('invalid costs')
    @property
    def target(self):return self.max_drawdown*self.excess_return_multiple
    @property
    def cost_rate(self):return (self.fee_bps+self.slippage_bps)/10000


def forecast_excess(history,cash):
    forecasts=[]
    for n in (63,126,252):
        forecasts.append((history.iloc[-1]/history.iloc[-1-n])**(252/n)-np.prod(1+cash.iloc[-n:].to_numpy())**(252/n))
    return .5*forecasts[0]+.3*forecasts[1]+.2*forecasts[2]


def optimize_weights(returns,cash,mu,current,s):
    """Input scenarios are point-in-time; weights independent of asset class labels."""
    codes=list(returns);cr=cash.to_numpy();R=returns.to_numpy();n=len(codes);t=len(R)
    if not n:raise ValueError('no eligible assets')
    if not np.isfinite(R).all() or not np.isfinite(cr).all():raise ValueError('missing risk inputs')
    covs=[LedoitWolf().fit((R-cr[:,None])[-k:]).covariance_*252+np.eye(n)*1e-10 for k in (252,63)]
    w=cp.Variable(n,nonneg=True);dd=cp.Variable(t+1,nonneg=True)
    path=(R-cr[:,None])@w+cr
    cons=[cp.sum(w)<=1,dd[0]==0,dd[1:]>=dd[:-1]-path,dd[1:]<=s.max_drawdown]
    forecast=mu.reindex(codes).to_numpy();risk=cp.maximum(*[cp.quad_form(w,cp.psd_wrap(c)) for c in covs])
    # Include required liquidation of excluded assets in audit, though constant in this solve.
    liquidation=float(current.drop(labels=codes,errors='ignore').abs().sum())
    penalty=s.cost_rate*(252/s.cost_horizon_sessions)*(cp.norm1(w-current.reindex(codes,fill_value=0).to_numpy())+liquidation)
    feasibility=cp.Problem(cp.Maximize(forecast@w),cons)
    feasibility.solve(solver='CLARABEL')
    if feasibility.status not in {'optimal','optimal_inaccurate'} or w.value is None:raise RuntimeError('risk feasibility solve: '+str(feasibility.status))
    best=float(forecast@w.value);feasible=best>=s.target+1e-6
    if feasible:
        cons.append(forecast@w>=s.target)
        obj=cp.Minimize(s.risk_aversion*risk+penalty)
    else:obj=cp.Maximize(forecast@w-s.risk_aversion*risk-penalty)
    problem=cp.Problem(obj,cons);problem.solve(solver='CLARABEL')
    if problem.status not in {'optimal','optimal_inaccurate'} or w.value is None:raise RuntimeError('weight solve: '+str(problem.status))
    weights=np.maximum(w.value,0)
    if weights.sum()>1+1e-6:raise RuntimeError('budget residual')
    if weights.sum()>1:weights/=weights.sum()
    simulated=(R-cr[:,None])@weights+cr;cumulative=np.r_[0,np.cumsum(simulated)]
    draws=np.maximum.accumulate(cumulative)[1:]-cumulative[1:]
    if draws.max()>s.max_drawdown+1e-6:raise RuntimeError('drawdown feasibility residual')
    var=float(weights@covs[0]@weights)
    rc=weights*(covs[0]@weights)/var if var>1e-16 else np.zeros(n)
    realized_forecast=float(forecast@weights)
    if feasible and realized_forecast<s.target-1e-6:raise RuntimeError('return target residual')
    audit={'status':problem.status,'target_feasible':feasible,'max_forecast_excess':best,
           'forecast_excess':realized_forecast,'target_annual_excess':s.target,
           'estimated_excess_volatility':float(np.sqrt(max(weights@c@weights for c in covs))),
           'fitted_linear_mdd':float(draws.max()),'fitted_linear_cdar95':tail_mean(draws,.95),
           'eligible_count':n,'forced_liquidation_weight':liquidation}
    return pd.Series(weights,index=codes),audit,pd.Series(rc,index=codes)


def allocate(history,cash,assets,current,s):
    r=history.pct_change(fill_method=None).tail(s.lookback)
    eligible,dedup=correlation_representatives(r,assets,s.correlation_threshold)
    target=pd.Series(0.,index=history.columns);rc=target.copy()
    if not eligible:return target,{'status':'no_eligible_assets','target_feasible':False},rc,dedup
    mu=forecast_excess(history[eligible],cash).clip(-.60,.60)*s.forecast_shrinkage
    weights,audit,risk=optimize_weights(r[eligible],cash.tail(s.lookback),mu,current,s)
    target.loc[eligible]=weights;rc.loc[eligible]=risk
    return target,audit,rc,dedup


def backtest(prices,cash,assets,s,start=504,progress=False):
    if start<s.lookback or len(prices)<start+3:raise ValueError('insufficient history')
    returns=prices.pct_change(fill_method=None);periods=prices.index.to_period('W-FRI')
    weights=pd.Series(0.,index=prices.columns);pending=None;nav=peak=1.
    rows=[];holdings=[];signals=[];contributions=[];dedups=[]
    for i in range(start,len(prices)):
        date=prices.index[i];prior=nav;r=returns.iloc[i]
        if r[weights>1e-12].isna().any():raise ValueError('held asset return missing')
        values=weights*(1+r.fillna(0));growth=values.sum()+(1-weights.sum())*(1+cash.iloc[i]);nav*=growth;weights=values/growth
        fee=turnover=0.;executed=None
        if pending is not None:
            target,executed=pending
            nav,turnover,fee=transact(nav,weights,target,s.cost_rate);weights=target.copy();pending=None
        peak=max(peak,nav)
        if i==start or periods[i]!=periods[i-1]:
            target,audit,rc,dedup=allocate(prices.iloc[i-s.lookback:i+1],cash.iloc[i-s.lookback:i+1],assets,weights,s)
            pending=(target,date)
            signals.append({'date':date,**audit,**target.to_dict(),'cash':1-target.sum()})
            contributions.append({'date':date,**rc.to_dict()})
            if not dedup.empty:dedups.append(dedup.assign(date=date))
            if progress and len(signals)%100==0:print('weekly signals',len(signals),str(date.date()),flush=True)
        rows.append({'date':date,'nav':nav,'return':nav/prior-1,'drawdown':1-nav/peak,'turnover':turnover,'cost_fraction':fee,'exposure':weights.sum(),'halted':False,'cash_return':cash.iloc[i],'executed_signal_date':executed})
        holdings.append({'date':date,**weights.to_dict(),'cash':1-weights.sum()})
    frames=[pd.DataFrame(x).set_index('date') for x in (rows,holdings,signals,contributions)]
    return (*frames,pd.concat(dedups,ignore_index=True) if dedups else pd.DataFrame())


def evaluate(curve,s):
    m=metrics(curve,Settings(s.max_drawdown));cash_cagr=float(np.prod(1+curve.cash_return)**(252/len(curve))-1)
    m.update(cash_cagr=cash_cagr,annual_excess=m['annual_return']-cash_cagr,required_annual_excess=s.target,
             goal_passed=bool(m['max_drawdown']<=s.max_drawdown and m['annual_return']-cash_cagr>=s.target))
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--market',choices=['CN','US'],default='CN');p.add_argument('--max-drawdown',type=float,default=.08)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--correlation-threshold',type=float,default=.90)
    a=p.parse_args();s=FlatModel(max_drawdown=a.max_drawdown,correlation_threshold=a.correlation_threshold)
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Do not overwrite research results')
    a.out.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'config.json').read_text());assets=cfg['markets'][a.market]['assets']
    f=ROOT/'data'/('cn_expanded28_20260924.csv' if a.market=='CN' else 'us_expanded_20260924.csv')
    meta=json.loads(f.with_suffix('.meta.json').read_text())
    if meta['market']!=a.market or meta['currency']!=cfg['markets'][a.market]['currency'] or meta['price_type']!='adjusted_close':raise ValueError('price metadata mismatch')
    prices=read_prices(f,assets);cash,ca,cm=lagged_cash(prices,a.market)
    manifest={'settings':asdict(s),'market':a.market,'frequency':'weekly','start_index_for_comparison':504,'last_price_date':str(prices.index[-1].date()),
              'cash_meta':cm,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prices_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
              'model_assumptions':['long only, no leverage','252-day linear historical drawdown <= input D; not future realized drawdown guarantee','risk penalty is max of 63/252-day cash-relative covariance variances','historical momentum expected return, unvalidated','no trend gate or positive-forecast gate','no class caps, individual caps, top-k, breadth, cooldown or extra volatility ceiling','weekly first trading day close signal, next close fill; first new return on following session','every week resolves weights, including when previously all cash'],
              'validation':'previously viewed historical data, not an untouched holdout; multiple architecture changes, not a single-factor ablation'}
    (a.out/'run.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(a.out/'rotation_flat_snapshot.py').write_bytes(Path(__file__).read_bytes())
    result=backtest(prices,cash,assets,s,progress=True)
    for name,frame in zip(['curve','holdings','signals','risk_contributions','dedup'],result):frame.to_csv(a.out/(name+'.csv'),index=name!='dedup')
    ca.to_csv(a.out/'cash_audit.csv',index=False)
    stats=evaluate(result[0],s);(a.out/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
