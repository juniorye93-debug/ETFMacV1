"""Two-stage, cash-relative ETF allocation with finite protection and explicit goal evaluation.

Research implementation inspired by the source-reviewed modules in references_v2.
It is not a reproduction of skfolio NCO or Freqtrade's trading engine.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from pypfopt import HRPOpt
from rotation import read_prices, transact, metrics, Settings
from cash_rates import cash_accrual

ROOT=Path(__file__).resolve().parent
CLASSES=('equity','bond','gold','oil')

@dataclass(frozen=True)
class Model:
    max_drawdown: float
    excess_return_multiple: float=2.
    frequency: str='weekly'
    method: str='nested_target'
    lookback: int=252
    volatility_limit: float=.10
    max_weight: float=.35
    class_cap_equity: float=.55
    class_cap_bond: float=.80
    class_cap_gold: float=.30
    class_cap_oil: float=.15
    cdar_fraction: float=.6
    historical_mdd_fraction: float=.9
    confidence: float=.95
    forecast_shrinkage: float=.5
    risk_aversion: float=1.
    fee_bps: float=3.
    slippage_bps: float=5.
    cost_horizon_sessions: int=63
    correlation_threshold: float=.90
    correlation_policy: str='global_config_order'
    risk_check_frequency: str='weekly'
    soft_stop_fraction: float=.5
    hard_stop_fraction: float=.9
    cooldown_sessions: int=20
    recovery_loss_fraction: float=.25
    recovery_increment: float=.25
    recovery_confirmations: int=2

    def __post_init__(self):
        if not 0 < self.max_drawdown < 1 or self.excess_return_multiple <= 0:
            raise ValueError('回撤及超额收益倍数无效')
        if self.frequency not in {'daily','weekly'} or self.method not in {'nested_target','protective_nested'}:
            raise ValueError('未知频率或模型')
        if self.lookback != 252 or not 0 < self.volatility_limit < 1:
            raise ValueError('本研究固定252日估计窗，波动上限独立于回撤目标')
        if not 0 < self.soft_stop_fraction < self.hard_stop_fraction < 1:
            raise ValueError('软/硬风控阈值无效')
        if self.cooldown_sessions < 1 or not 0 < self.recovery_increment <= 1 or self.recovery_confirmations < 1:
            raise ValueError('恢复参数无效')
        if not 0 < self.max_weight <= 1 or any(not 0 <= self.cap(c) <= 1 for c in CLASSES):
            raise ValueError('权重上限无效')
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError('交易费用无效')
        if not 0 < self.correlation_threshold <= 1 or self.correlation_policy not in {'global_config_order','legacy_in_class'}:
            raise ValueError('相关性去重参数无效')
        if self.risk_check_frequency not in {'daily','weekly'}:
            raise ValueError('风险检查频率无效')

    @property
    def target(self):return self.excess_return_multiple*self.max_drawdown
    def cap(self,c):return getattr(self,'class_cap_'+c)


def capped_simplex(reference,cap,total=1.):
    """Euclidean projection on a capped simplex; preserves feasible invested budget."""
    a=np.asarray(reference,dtype=float);caps=np.broadcast_to(cap,a.shape).astype(float)
    if np.any(caps<0) or total<0 or caps.sum()<total-1e-10:
        raise ValueError('权重上限无法容纳要求的投资总额')
    lo=float(np.min(a-caps))-total;hi=float(np.max(a))+total
    for _ in range(80):
        mid=(lo+hi)/2;w=np.clip(a-mid,0,caps)
        if w.sum()>total:lo=mid
        else:hi=mid
    return np.clip(a-(lo+hi)/2,0,caps)


def lagged_cash(prices,market,file=None):
    file=Path(file) if file is not None else ROOT/'data'/('aaa_ncd_1y.csv' if market=='CN' else 'us_dgs3mo.csv')
    y=pd.read_csv(file,index_col=0,parse_dates=True).annual_yield
    meta=json.loads(file.with_suffix('.meta.json').read_text())
    if meta.get('currency') != ('CNY' if market=='CN' else 'USD') or meta.get('unit')!='decimal':
        raise ValueError('现金利率币种或单位不正确')
    observation_dates=None
    if market=='US':
        # H.15 is published after its observation date. Conservatively delay two business days.
        observation_dates=pd.Series(y.index,index=y.index+pd.offsets.BDay(2)).groupby(level=0).last()
        y.index=y.index+pd.offsets.BDay(2)
        y=y.groupby(level=0).last()
    r,audit=cash_accrual(prices.index,y)
    if observation_dates is not None:
        audit['availability_proxy_date']=audit['last_observation_date']
        audit['last_observation_date']=audit['availability_proxy_date'].map(observation_dates)
    return r,audit,meta


def correlation_representatives(ret,assets,threshold=.90):
    """Causal, market-wide positive-correlation pruning in explicit config order.

    Every removed asset has a DIRECT retained representative. Negative correlation
    is never pruned. No connected-component chaining, forecasts or future data.
    """
    if not 0 < threshold <= 1:raise ValueError('invalid correlation threshold')
    eligible=[c for c in assets if c in ret and ret[c].notna().all() and ret[c].std()>1e-8]
    corr=ret[eligible].corr();kept=[];audit=[]
    for c in eligible:
        matches=[k for k in kept if corr.loc[c,k]>=threshold]
        representative=max(matches,key=lambda k:corr.loc[c,k]) if matches else c
        audit.append({'code':c,'representative':representative,'kept':not bool(matches),
                      'correlation':float(corr.loc[c,representative]),'threshold':threshold})
        if not matches:kept.append(c)
    return kept,pd.DataFrame(audit)


def class_snapshot(history,cash_history,assets,s):
    codes=list(history);ret=history.pct_change(fill_method=None).iloc[-252:]
    eligible=[c for c in codes if ret[c].notna().all() and ret[c].std()>1e-8]
    if s.correlation_policy=='global_config_order':
        eligible,_=correlation_representatives(ret,assets,s.correlation_threshold)
    B=pd.DataFrame(0.,index=codes,columns=CLASSES)
    scores=pd.Series(np.nan,index=codes);passed=pd.Series(False,index=codes)
    if not eligible:return B,scores,0.
    p=history[eligible]
    # Same-period annualized excess return: compare to the currency-matched cash path.
    forecasts=[]
    for n in (63,126,252):
        asset_annual=(p.iloc[-1]/p.iloc[-1-n])**(252/n)-1
        cash_annual=np.prod(1+cash_history.iloc[-n:].to_numpy())**(252/n)-1
        forecasts.append(asset_annual-cash_annual)
    excess=.5*forecasts[0]+.3*forecasts[1]+.2*forecasts[2]
    # Shrink forecasts rather than force their values up to the user's goal.
    scores.loc[eligible]=excess.clip(-.60,.60)*s.forecast_shrinkage
    passed.loc[eligible]=(excess>0)&(p.iloc[-1]>p.tail(126).mean())
    corr=ret[eligible].corr()
    for cls in CLASSES:
        ranked=scores[[c for c in eligible if assets[c]['class']==cls and passed[c]]].sort_values(ascending=False,kind='stable')
        chosen=[];bucket_counts={}
        for c in ranked.index:
            bucket=assets[c].get('bucket',cls)
            if cls=='equity' and bucket_counts.get(bucket,0)>=2:continue
            if s.correlation_policy=='legacy_in_class' and any(corr.loc[c,k]>s.correlation_threshold for k in chosen):continue
            chosen.append(c);bucket_counts[bucket]=bucket_counts.get(bucket,0)+1
            if len(chosen)>= (4 if cls=='equity' else 3):break
        if not chosen:continue
        if len(chosen)==1:w=np.ones(1)
        else:
            cov=LedoitWolf().fit(ret[chosen].to_numpy()).covariance_+np.eye(len(chosen))*1e-12
            w=pd.Series(HRPOpt(cov_matrix=pd.DataFrame(cov,index=chosen,columns=chosen)).optimize()).reindex(chosen).to_numpy()
            w=capped_simplex(w,max(.5,1/len(chosen)))
        B.loc[chosen,cls]=w
    # Equal bucket weights stop numerous sector ETFs from dominating breadth.
    groups={}
    for c in eligible:
        if assets[c]['class']=='equity':groups.setdefault(assets[c].get('bucket','equity'),[]).append(float(passed[c]))
    breadth=float(np.mean([np.mean(v) for v in groups.values()])) if groups else 0.
    return B,scores,breadth


def build_sleeves(prices,cash,assets,s):
    """Causal inner strategies: t signal, t+1 close fill, t+2 first new return.

    The outer model later receives these realized synthetic sleeve returns,
    never a current inner weight applied backwards over its fitting period.
    """
    codes=list(prices);periods=prices.index.to_period('W-FRI')
    live=pd.DataFrame(0.,index=codes,columns=CLASSES);pending=None;latest=None
    R=pd.DataFrame(np.nan,index=prices.index,columns=CLASSES);snapshots={}
    for i in range(s.lookback,len(prices)):
        r=prices.iloc[i]/prices.iloc[i-1]-1
        held=live.sum(axis=1)>1e-12
        if r[held].isna().any():raise ValueError('类内持仓收益缺失')
        v=live.mul(1+r.fillna(0),axis=0)
        growth=v.sum(axis=0)+(1-live.sum(axis=0))*(1+cash.iloc[i])
        R.iloc[i]=growth-1
        live=v.div(growth,axis=1)
        if pending is not None:live=pending.copy();pending=None
        scheduled=s.frequency=='daily' or i==s.lookback or periods[i]!=periods[i-1]
        if scheduled:
            latest=class_snapshot(prices.iloc[i-s.lookback:i+1],cash.iloc[i-s.lookback:i+1],assets,s)
            pending=latest[0].copy();snapshots[i]=latest
    return R,snapshots


def constrained_outer(B,forecast,sleeve_returns,asset_returns,cash_history,current,s,breadth):
    codes=list(B.index);active=B.sum(axis=0)>0
    zero=pd.Series(0.,index=codes)
    if not active.any():return zero,{'status':'no_positive_excess_trend','target_feasible':False,'max_forecast_excess':0.,'forecast_excess':0.,'equity_breadth':breadth}
    z=cp.Variable(len(CLASSES),nonneg=True);w=B.to_numpy()@z
    selected=np.where(B.sum(axis=1).to_numpy()>0)[0]
    mu=forecast.fillna(0).to_numpy()
    constraints=[cp.sum(z)<=1,w<=s.max_weight]
    for j,c in enumerate(CLASSES):
        cap=s.cap(c)
        if c=='equity' and s.method=='protective_nested':cap*=float(np.clip(2*breadth-1,0,1))
        constraints.append(z[j]<=cap if active.iloc[j] else z[j]==0)
    covs=[]
    for window in (252,63):
        x=asset_returns.iloc[-window:,selected].to_numpy()
        cov=LedoitWolf().fit(x).covariance_*252+np.eye(len(selected))*1e-10
        covs.append(cov)
        constraints.append(cp.quad_form(w[selected],cp.psd_wrap(cov))<=s.volatility_limit**2)
    # Walk-forward sleeve risk includes class rotation, unlike static backfilling of current weights.
    R=sleeve_returns.to_numpy();cr=cash_history.to_numpy();t=len(R)
    cv=LedoitWolf().fit(R-cr[:,None]).covariance_*252+np.eye(len(CLASSES))*1e-10
    constraints.append(cp.quad_form(z,cp.psd_wrap(cv))<=s.volatility_limit**2)
    path=(R-cr[:,None])@z+cr
    dd=cp.Variable(t+1,nonneg=True);q=cp.Variable(nonneg=True);u=cp.Variable(t,nonneg=True)
    constraints += [dd[0]==0,dd[1:]>=dd[:-1]-path,dd[1:]<=s.max_drawdown*s.historical_mdd_fraction,
                    u>=dd[1:]-q,q+cp.sum(u)/((1-s.confidence)*t)<=s.max_drawdown*s.cdar_fraction]
    # Target feasibility is evaluated separately; infeasibility is never concealed by inventing returns.
    try:
        feasibility=cp.Problem(cp.Maximize(mu@w),constraints);feasibility.solve(solver='CLARABEL')
        if feasibility.status not in {'optimal','optimal_inaccurate'} or z.value is None:raise ValueError('infeasible risk constraints')
        best=float(mu@(B.to_numpy()@z.value));feasible=best>=s.target+1e-6
        transaction_cost=(s.fee_bps+s.slippage_bps)/10000*(252/s.cost_horizon_sessions)*cp.norm1(w-current.to_numpy())
        variance=cp.quad_form(w[selected],cp.psd_wrap(covs[0]))
        if feasible:
            constraints.append(mu@w>=s.target)
            objective=cp.Minimize(s.risk_aversion*variance+transaction_cost)
        else:objective=cp.Maximize(mu@w-s.risk_aversion*variance-transaction_cost)
        problem=cp.Problem(objective,constraints);problem.solve(solver='CLARABEL')
        if problem.status not in {'optimal','optimal_inaccurate'} or z.value is None:raise ValueError('allocation solve failed')
        zz=np.maximum(z.value,0);zz[~active.to_numpy()]=0
        target=pd.Series(B.to_numpy()@zz,index=codes)
        if target.sum()>1+1e-6 or target.max()>s.max_weight+1e-6:raise ValueError('weight feasibility residual')
        # Numerical residual only. Economically meaningful reallocation happens inside the optimizer.
        if target.sum()>1:target/=target.sum()
        risk=float(np.sqrt(target.iloc[selected].to_numpy()@covs[0]@target.iloc[selected].to_numpy()))
        return target,{'status':problem.status,'target_feasible':bool(feasible),'max_forecast_excess':best,
                       'forecast_excess':float(mu@target.to_numpy()),'estimated_volatility':risk,'equity_breadth':breadth,
                       **{'class_'+c:float(zz[j]) for j,c in enumerate(CLASSES)}}
    except (cp.error.SolverError,ValueError) as e:
        return zero,{'status':'solver_failed_cash:'+str(e),'target_feasible':False,'max_forecast_excess':np.nan,'forecast_excess':0.,'equity_breadth':breadth}


class RecoveryController:
    """Finite episode controls; NEVER modifies the global performance high-water mark."""
    def __init__(self,s):
        self.s=s;self.phase='normal';self.scale=1.;self.episode_peak=1.;self.until=-1;self.confirmations=0
    def update(self,i,nav,ready,scheduled):
        old=(self.phase,self.scale);s=self.s;event=''
        if self.phase=='cooldown':
            if i>=self.until and scheduled and ready:
                self.phase='recovery';self.scale=s.recovery_increment;self.episode_peak=nav;event='resume_after_cooldown'
            return self.scale,old!=(self.phase,self.scale),event
        self.episode_peak=max(self.episode_peak,nav);dd=1-nav/self.episode_peak
        stop=s.max_drawdown*(s.recovery_loss_fraction if self.phase=='recovery' else s.hard_stop_fraction)
        if dd>=stop:
            self.phase='cooldown';self.scale=0.;self.until=i+s.cooldown_sessions;self.confirmations=0;event='finite_stop'
        elif self.phase=='normal' and dd>=s.max_drawdown*s.soft_stop_fraction:
            self.phase='defensive';self.scale=.5;self.confirmations=0;event='single_defensive_cut'
        elif self.phase=='defensive' and scheduled:
            self.confirmations=self.confirmations+1 if ready else 0
            if self.confirmations>=s.recovery_confirmations:
                self.phase='recovery';self.scale=.5;self.episode_peak=nav;event='resume_after_confirmation'
        elif self.phase=='recovery' and scheduled and ready:
            self.scale=min(1.,self.scale+s.recovery_increment);event='recovery_step'
            if self.scale>=1:self.phase='normal'
        return self.scale,old!=(self.phase,self.scale),event


def backtest_v2(prices,cash,assets,s,sleeves=None):
    if sleeves is None:sleeves=build_sleeves(prices,cash,assets,s)
    R,snapshots=sleeves;codes=list(prices);start=2*s.lookback
    if len(prices)<=start+2:raise ValueError('需要至少507行价格：252日类内预热＋252日因果类间估计')
    returns=prices.pct_change(fill_method=None);weights=pd.Series(0.,index=codes);pending=None
    shadow_weights=weights.copy();shadow_pending=None;shadow=1.;shadow_cash=1.;shadow_history=[]
    base=weights.copy();nav=peak=1.;controller=RecoveryController(s);rows=[];holdings=[];signals=[];events=[]
    last_audit={};allocation_date=None
    available=sorted(snapshots);latest_i=max(k for k in available if k<=start)
    for i in range(start,len(prices)):
        date=prices.index[i];prior=nav;r=returns.iloc[i]
        if r[weights>1e-12].isna().any():raise ValueError('实际持仓收益缺失')
        r=r.fillna(0);v=weights*(1+r);growth=v.sum()+(1-weights.sum())*(1+cash.iloc[i]);nav*=growth;weights=v/growth
        sv=shadow_weights*(1+r);sg=sv.sum()+(1-shadow_weights.sum())*(1+cash.iloc[i]);shadow*=sg;shadow_weights=sv/sg
        fee=turnover=0.;executed=None
        if pending is not None:
            target,signal_date=pending;nav,turnover,fee=transact(nav,weights,target,(s.fee_bps+s.slippage_bps)/10000);weights=target.copy();executed=signal_date;pending=None
        if shadow_pending is not None:
            shadow,_,_=transact(shadow,shadow_weights,shadow_pending,(s.fee_bps+s.slippage_bps)/10000);shadow_weights=shadow_pending.copy();shadow_pending=None
        shadow_cash*=1+cash.iloc[i]
        shadow_history.append(shadow/shadow_cash);peak=max(peak,nav);global_dd=1-nav/peak
        scheduled=i in snapshots or i==start
        audit={}
        if scheduled:
            if i in snapshots:latest_i=i
            B,forecast,breadth=snapshots[latest_i]
            # All fitting rows ended by the current close. Every sleeve return was generated causally.
            base,audit=constrained_outer(B,forecast,R.iloc[i-251:i+1],returns.iloc[i-251:i+1],cash.iloc[i-251:i+1],weights,s,breadth)
            last_audit=audit.copy();allocation_date=date
            shadow_pending=base.copy()
        ready=(base.sum()>.05 and len(shadow_history)>=22 and shadow_history[-1]>np.mean(shadow_history[-21:]) and shadow_history[-1]>shadow_history[-22])
        if scheduled or s.risk_check_frequency=='daily':
            scale,changed,event=controller.update(i,nav,ready,scheduled)
        else:
            # Daily valuation still updates the episode peak; orders only follow weekly signals.
            controller.episode_peak=max(controller.episode_peak,nav)
            scale,changed,event=controller.scale,False,''
        if scheduled or changed:
            target=base*scale;pending=(target,date)
            reported=last_audit.copy()
            for key in ('target_feasible','forecast_excess','max_forecast_excess'):
                reported['base_'+key]=reported.pop(key,np.nan)
            reported['forecast_excess']=reported['base_forecast_excess']*scale
            reported['max_forecast_excess']=reported['base_max_forecast_excess']*scale
            reported['target_feasible']=bool(reported['forecast_excess']>=s.target-1e-6)
            signals.append({'date':date,'phase':controller.phase,'risk_scale':scale,'global_drawdown':global_dd,
                            'target_annual_excess':s.target,'recovery_ready':ready,'allocation_date':allocation_date,**reported,
                            **target.to_dict(),'cash':1-target.sum()})
        if event:events.append({'date':date,'event':event,'phase':controller.phase,'risk_scale':scale,'global_drawdown':global_dd,'episode_peak_for_control':controller.episode_peak,'global_peak_for_performance':peak})
        rows.append({'date':date,'nav':nav,'return':nav/prior-1,'drawdown':global_dd,'turnover':turnover,'cost_fraction':fee,
                     'exposure':weights.sum(),'halted':controller.phase=='cooldown','cash_return':cash.iloc[i],'executed_signal_date':executed,
                     'phase':controller.phase,'risk_scale':scale,'shadow_nav':shadow,'shadow_excess_nav':shadow/shadow_cash,'global_peak':peak})
        holdings.append({'date':date,**weights.to_dict(),'cash':1-weights.sum()})
    return pd.DataFrame(rows).set_index('date'),pd.DataFrame(holdings).set_index('date'),pd.DataFrame(signals).set_index('date'),pd.DataFrame(events)


def evaluate(curve,s):
    out=metrics(curve,Settings(max_drawdown=s.max_drawdown))
    cash_cagr=float(np.prod(1+curve.cash_return)**(252/len(curve))-1)
    out.update(cash_cagr=cash_cagr,annual_excess=out['annual_return']-cash_cagr,required_annual_excess=s.target,
               excess_shortfall=max(0.,s.target-(out['annual_return']-cash_cagr)),
               goal_passed=bool(out['max_drawdown']<=s.max_drawdown and out['annual_return']-cash_cagr>=s.target),
               cooldown_fraction=float(curve.phase.eq('cooldown').mean()) if 'phase' in curve else 0.)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--market',choices=['CN','US'],required=True)
    p.add_argument('--max-drawdown',type=float,required=True);p.add_argument('--excess-return-multiple',type=float,default=2.)
    p.add_argument('--frequency',choices=['weekly'],default='weekly');p.add_argument('--method',choices=['nested_target','protective_nested'],default='nested_target')
    p.add_argument('--prices',type=Path);p.add_argument('--cash-yields',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--volatility-limit',type=float,default=.10)
    p.add_argument('--correlation-threshold',type=float,default=.90)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('输出目录非空，禁止覆盖研究记录')
    a.out.mkdir(parents=True,exist_ok=True)
    s=Model(a.max_drawdown,excess_return_multiple=a.excess_return_multiple,frequency=a.frequency,method=a.method,volatility_limit=a.volatility_limit,correlation_threshold=a.correlation_threshold)
    cfg=json.loads(a.config.read_text());assets=cfg['markets'][a.market]['assets']
    file=a.prices if a.prices is not None else ROOT/'data'/('cn_expanded28_20260924.csv' if a.market=='CN' else 'us_expanded_20260924.csv');prices=read_prices(file,assets)
    price_meta=json.loads(file.with_suffix('.meta.json').read_text())
    if price_meta.get('market')!=a.market or price_meta.get('currency')!=cfg['markets'][a.market]['currency'] or price_meta.get('price_type')!='adjusted_close':
        raise ValueError('价格元数据市场/币种/复权口径不匹配')
    cash,cash_audit,cash_meta=lagged_cash(prices,a.market,a.cash_yields)
    manifest={'settings':asdict(s),'target_definition':'strategy CAGR minus same-period cash CAGR >= 2 * input maximum drawdown; realized full-history maximum drawdown <= input limit',
              'market':a.market,'cash_meta':cash_meta,'prices_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'validation':'fixed design research; previously viewed history, not a new untouched holdout','execution':'t close signal; t+1 close fill; t+2 first return; global performance peak never reset'}
    (a.out/'run.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    result=backtest_v2(prices,cash,assets,s)
    for name,data in zip(['curve','holdings','signals','events'],result):data.to_csv(a.out/(name+'.csv'))
    rows=[]
    for name,sub in [('full',result[0]),('last252_descriptive',result[0].iloc[-252:]),*[(str(y),g) for y,g in result[0].groupby(result[0].index.year)]]:
        rows.append({'segment':name,**evaluate(sub,s)})
    pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);cash_audit.to_csv(a.out/'cash_audit.csv',index=False)
    print(a.market,a.method,a.frequency,json.dumps(rows[0],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
