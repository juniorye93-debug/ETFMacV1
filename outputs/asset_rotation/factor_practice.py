"""Fast, reproducible weekly factor/strategy experiments on frozen real Wind data.
All inspected history is exploratory. No brokerage connection or qualified model.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import argparse,hashlib,json,pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rotation import read_prices,transact,metrics,Settings,tail_mean
from rotation_v2 import lagged_cash
from rotation_adaptive import covariance_estimate,representatives
from forecast_adaptive import signal_positions
from weekly_review.run_pure_risk_parity import erc_weights
from platform_factors import build_factors,factor_manifest,BASE_FACTORS,NEW_PRICE_FACTORS,VALUATION_FACTORS,RSRS_FACTORS,FUNDAMENTAL_FACTORS,SHARE_FACTORS

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'factor_practice'

@dataclass(frozen=True)
class PracticeSettings:
    max_drawdown:float=.08
    start:str='2017-01-01'
    cost_rate:float=.0008
    covariance_history:int=1040
    covariance_half_life_weeks:float=13.
    min_asset_days:int=252
    positive_correlation_threshold:float=.90
    factor_top_k:int=3
    rsrs_entry:float=.7
    rsrs_exit:float=-.7
    rank_keep_margin:float=.05
    execution_buffer_l1:float=.05
    scenario_paths:int=64
    scenario_days:int=65
    scenario_block_days:int=5
    scenario_quantile:float=.95
    scenario_risk_measure:str='quantile'
    scenario_vol_multiplier:float=1.
    recovery_max_weekly_increase:float=.10
    recovery_confirmation_weeks:int=2
    seed:int=7107
    def __post_init__(self):
        if not 0<self.max_drawdown<1:raise ValueError('max_drawdown must lie in(0,1)')
        if not 0<=self.cost_rate<.1:raise ValueError('invalid cost')
        if self.scenario_risk_measure not in ['quantile','cvar']:raise ValueError('invalid scenario risk measure')
        if self.scenario_paths<16 or self.scenario_days<5 or self.scenario_vol_multiplier<=0:raise ValueError('invalid scenarios')

def load_market(market,s=PracticeSettings(),rebuild=False):
    cfg=json.loads((ROOT/'config.json').read_text());assets=cfg['markets'][market]['assets']
    prices=read_prices(ROOT/'data'/('cn_expanded28_20260924.csv' if market=='CN' else 'us_expanded_20260924.csv'),assets)
    cash,ca,cm=lagged_cash(prices,market);benchmark='510300.SH' if market=='CN' else 'SPY.P'
    dest=OUT/market;dest.mkdir(parents=True,exist_ok=True);cache=dest/'factors.pkl'
    if cache.exists() and not rebuild:
        with cache.open('rb') as f:factors=pickle.load(f)
        # Deterministic schema extension; do not change cached historical inputs
        # or silently refetch prices when adding the predeclared reversal factor.
        if 'short_reversal_1m' not in factors:factors['short_reversal_1m']=-factors['mom_1m']
    else:
        ohlcv=pd.read_csv(OUT/'data'/(market+'_ohlcv.csv'),parse_dates=['date'])
        valuation=pd.read_csv(ROOT/'signal_research/wind_valuation/index_valuation_history.csv',parse_dates=['date']) if market=='CN' else None
        fp=OUT/'data'/'CN_fundamentals.csv';sp=OUT/'data'/'ETF_shares_correct_calendar.csv'
        if not sp.exists():sp=OUT/'data'/'ETF_shares.csv'
        financial=pd.read_csv(fp,parse_dates=['date']) if market=='CN' and fp.exists() else None
        shares=pd.read_csv(sp,parse_dates=['date']) if sp.exists() else None
        if shares is not None:
            shares['date']=pd.to_datetime(shares['date'],format='mixed').dt.normalize()
            shares=shares.loc[shares.market==market]
        factors=build_factors(prices,cash,benchmark,ohlcv,valuation,financial=financial,shares=shares)
        with cache.open('wb') as f:pickle.dump(factors,f)
    return {'market':market,'assets':assets,'prices':prices,'cash':cash,'cash_audit':ca,'cash_meta':cm,'factors':factors,'benchmark':benchmark,'positions':signal_positions(prices.index),'returns':prices.pct_change(fill_method=None),'dest':dest}

def prepare_weeks(d,s):
    packages={};p=d['prices'];r=d['returns'];rows=[]
    for pos in d['positions']:
        if p.index[pos]<pd.Timestamp(s.start) or pos<252:continue
        history=r.iloc[max(0,pos-s.covariance_history+1):pos+1]
        available=[c for c in p if history[c].notna().sum()>=s.min_asset_days and np.isfinite(p.iloc[pos][c])]
        if not available:continue
        cov,corr,meta=covariance_estimate(history[available],s.covariance_half_life_weeks,s.min_asset_days)
        keep,duplicates=representatives(corr,available,s.positive_correlation_threshold)
        for row in duplicates.to_dict('records'):rows.append({'date':p.index[pos],**row})
        packages[int(pos)]={'codes':available,'representatives':[available[j] for j in keep],'cov':cov,'corr':corr,'meta':meta,'history':history[available]}
    d['packages']=packages
    filename=f'dedup_hl{s.covariance_half_life_weeks:g}_threshold{s.positive_correlation_threshold:g}.csv'
    pd.DataFrame(rows).to_csv(d['dest']/filename,index=False)
    if s.covariance_half_life_weeks==13.:pd.DataFrame(rows).to_csv(d['dest']/'dedup.csv',index=False)

def factor_diagnostics(d,s):
    p=d['prices'];f=d['factors'];stock=[c for c,a in d['assets'].items() if a['class']=='equity'];positions=d['positions']
    names=[n for n in BASE_FACTORS+NEW_PRICE_FACTORS+RSRS_FACTORS+VALUATION_FACTORS+FUNDAMENTAL_FACTORS+SHARE_FACTORS if n in f and f[n].notna().any().any()]
    obs=[];latest=[]
    for j,pos in enumerate(positions):
        if pos not in d['packages']:continue
        codes=[c for c in d['packages'][pos]['representatives'] if c in stock]
        if pos==max(d['packages']):
            for c in codes:latest.append({'date':p.index[pos],'code':c,'name':d['assets'][c]['name'],**{n:f[n].iloc[pos][c] for n in names}})
        for h in [4,13,26]:
            if j+h>=len(positions):continue
            fill=pos+1;end=int(positions[j+h])+1
            if end>=len(p):continue
            y=p.iloc[end]/p.iloc[fill]/np.prod(1+d['cash'].iloc[fill+1:end+1])-1
            for n in names:
                pair=pd.DataFrame({'x':f[n].iloc[pos][codes],'y':y[codes]}).dropna()
                if len(pair)<5 or pair.x.nunique()<2 or pair.y.nunique()<2:continue
                obs.append({'date':p.index[pos],'label_end':p.index[end],'factor':n,'horizon_weeks':h,'assets':len(pair),'rank_ic':float(spearmanr(pair.x,pair.y).statistic)})
    a=pd.DataFrame(obs);a.to_csv(d['dest']/'factor_ic_observations.csv',index=False)
    summaries=[]
    for segment,mask in [('all_explored',a.date.notna()),('2017_2021_explored',a.date<pd.Timestamp('2022-01-01')),('2022_2026_explored',a.date>=pd.Timestamp('2022-01-01'))]:
        for (name,h),g in a.loc[mask].groupby(['factor','horizon_weeks']):
            x=g.rank_ic.to_numpy();n=len(x);rng=np.random.default_rng(s.seed+h)
            starts=rng.integers(0,n,(512,int(np.ceil(n/h))));idx=((starts[:,:,None]+np.arange(h))%n).reshape(512,-1)[:,:n]
            lo,hi=np.quantile(x[idx].mean(axis=1),[.025,.975])
            summaries.append({'market':d['market'],'segment':segment,'factor':name,'horizon_weeks':h,'weeks':n,'mean_ic':float(x.mean()),'block_ci_low':float(lo),'block_ci_high':float(hi),'mean_assets':g.assets.mean(),'inference':'exploratory95 interval; many comparisons, not acceptance'})
    pd.DataFrame(summaries).to_csv(d['dest']/'factor_ic_summary.csv',index=False);pd.DataFrame(latest).to_csv(d['dest']/'latest_factors.csv',index=False)

def weighted_target(d,pos,codes,method='erc',raw=None):
    z=pd.Series(0.,index=d['prices'].columns)
    if not codes:return z
    package=d['packages'][pos];ix=[package['codes'].index(c) for c in codes];cov=package['cov'][np.ix_(ix,ix)]
    if raw is not None:w=np.asarray(raw,float);w=w/w.sum()
    elif method=='equal':w=np.ones(len(codes))/len(codes)
    elif method=='inverse_vol':w=1/np.sqrt(np.diag(cov));w/=w.sum()
    else:w,_=erc_weights(cov)
    z.loc[codes]=w;return z

def ranked_rule(factor,top=3,absolute=True,stock_only=False,method='erc',hysteresis=False):
    def rule(d,pos,current,state):
        codes=d['packages'][pos]['representatives']
        if stock_only:codes=[c for c in codes if d['assets'][c]['class']=='equity']
        score=d['factors'][factor].iloc[pos].reindex(codes).dropna()
        if absolute:score=score.loc[d['factors']['cash_relative_13612U'].iloc[pos].reindex(score.index)>0]
        if score.empty:return pd.Series(0.,index=current.index)
        ranks=score.rank(pct=True);sorted_codes=score.sort_values(ascending=False,kind='stable').index.tolist();chosen=sorted_codes[:top]
        if hysteresis and len(chosen)==top:
            cutoff=float(ranks[chosen[-1]])
            incumbents=[c for c in score.index if current[c]>1e-8 and ranks[c]>=cutoff-.05]
            chosen=sorted(incumbents,key=lambda c:-ranks[c])[:top]
            chosen+= [c for c in sorted_codes if c not in chosen][:top-len(chosen)]
        return weighted_target(d,pos,chosen,method)
    return rule

def scenario_growth(d,pos,s):
    pkg=d['packages'][pos]
    if 'scenario_growth' in pkg:return
    rng=np.random.default_rng(s.seed+pos);cov=pkg['cov'];sd=np.sqrt(np.diag(cov));n=len(sd);P=s.scenario_paths;days=s.scenario_days+1
    common=pkg['history'].dropna().to_numpy();center=common-common.mean(axis=0);std=np.maximum(center.std(axis=0,ddof=1),1e-8)
    starts=rng.integers(0,len(common),(P//2,int(np.ceil(days/s.scenario_block_days))))
    ix=((starts[:,:,None]+np.arange(s.scenario_block_days))%len(common)).reshape(P//2,-1)[:,:days]
    historical=center[ix]/std*sd
    z=rng.normal(size=(P-P//2,days,n))@np.linalg.cholesky(cov+np.eye(n)*1e-14).T
    tscale=np.sqrt(3/rng.chisquare(5,size=(P-P//2,days,1)))
    cash_daily=float(d['cash'].iloc[pos]);shocks=np.concatenate([historical,z*tscale],axis=0)
    # No assumed positive alpha in protective scenarios. Variance correction in log growth.
    shocks*=s.scenario_vol_multiplier
    logs=shocks+cash_daily-.5*(sd*s.scenario_vol_multiplier)**2
    pkg['latency_returns']=np.expm1(logs[:,0]).astype('float32')
    pkg['scenario_growth']=np.exp(np.cumsum(logs[:,1:],axis=1)).astype('float32')
    pkg['cash_growth']=(1+cash_daily)**np.arange(1,s.scenario_days+1)

def prospective_risk(d,pos,target,current,nav,peak,s):
    scenario_growth(d,pos,s);pkg=d['packages'][pos];codes=pkg['codes'];w=target.reindex(codes).to_numpy();old=current.reindex(codes).to_numpy()
    extra=float(current.sum()-old.sum());cash_daily=float(d['cash'].iloc[pos])
    g=1+pkg['latency_returns']@old+(1-old.sum()-extra)*cash_daily+extra*cash_daily
    old_drift=old[None,:]*(1+pkg['latency_returns'])/g[:,None]
    fee=s.cost_rate*(np.abs(old_drift-w[None,:]).sum(axis=1)+extra/g)
    wealth=g[:,None]*(1-fee[:,None])*(pkg['scenario_growth']@w+(1-w.sum())*pkg['cash_growth'][None,:])
    path=np.concatenate([np.full((len(g),1),1.),g[:,None],(g*(1-fee))[:,None],wealth],axis=1)
    path_peak=np.maximum(peak/nav,np.maximum.accumulate(path,axis=1))
    dd=np.max(1-path/path_peak,axis=1)
    return tail_mean(dd,s.scenario_quantile) if s.scenario_risk_measure=='cvar' else float(np.quantile(dd,s.scenario_quantile))

def guard_target(d,pos,target,current,nav,peak,state,s,recovery=False):
    scale=0.;risk=np.nan;breached=1-nav/peak>=s.max_drawdown
    # Positive signal for two weekly decisions is a research recovery rule, not a user cap.
    if target.sum()>0:state['positive_weeks']=state.get('positive_weeks',0)+1
    else:state['positive_weeks']=0
    if not breached:
        for trial in np.linspace(1,0,21):
            candidate=target*trial
            if recovery and state.get('risk_state') in ['DERISK','PROBE','RESTORE']:
                max_exposure=current.sum()+(s.recovery_max_weekly_increase if state['positive_weeks']>=s.recovery_confirmation_weeks else 0)
                if candidate.sum()>max_exposure+1e-10:continue
            q=prospective_risk(d,pos,candidate,current,nav,peak,s)
            if q<=s.max_drawdown+1e-10:scale=float(trial);risk=q;break
    candidate=target*scale
    # Re-evaluate the FINAL weights, including all overlays and recovery changes.
    if not np.isfinite(risk):risk=prospective_risk(d,pos,candidate,current,nav,peak,s)
    prev=state.get('risk_state','NORMAL')
    if candidate.sum()<current.sum()-1e-5:status='DERISK'
    elif current.sum()<1e-8 and candidate.sum()>0:status='PROBE'
    elif candidate.sum()>current.sum()+1e-5 and prev!='NORMAL':status='RESTORE'
    elif scale>=1-1e-8:status='NORMAL'
    else:status=prev if prev in ['DERISK','PROBE','RESTORE'] else 'DERISK'
    state['risk_state']=status
    return candidate,{'scenario_scale':scale,'scenario_mdd_estimate':risk,'scenario_risk_measure':s.scenario_risk_measure,'scenario_mdd_q95':risk if s.scenario_risk_measure=='quantile' else np.nan,'risk_state':status,'prospective_budget_failed':risk>s.max_drawdown+1e-10,'account_already_breached':breached}

def simulate(d,name,rule,s,guard=False,recovery=False,buffer=False,blend4=False):
    p=d['prices'];cash=d['cash'];weights=pd.Series(0.,index=p.columns);nav=peak=1.;pending=None;pending_signal=None;state={};records=[];holdings=[];signals=[];history_targets=[]
    for i,date in enumerate(p.index):
        if date<pd.Timestamp(s.start):continue
        prior=nav;r=d['returns'].iloc[i]
        if r[weights>1e-12].isna().any():raise ValueError('missing held price')
        values=weights*(1+r.fillna(0));growth=float(values.sum()+(1-weights.sum())*(1+cash.iloc[i]));nav*=growth;weights=values/growth
        cost=turnover=0.;executed=None
        if pending is not None:
            nav,turnover,cost=transact(nav,weights,pending,s.cost_rate);weights=pending.copy();executed=pending_signal;pending=None
        peak=max(peak,nav);dd=1-nav/peak
        if i in d['packages']:
            target=rule(d,i,weights,state)
            if target.isna().any() or (target<-1e-10).any() or target.sum()>1+1e-8:raise ValueError('invalid target '+name)
            if blend4:
                history_targets.append(target.copy());target=sum(history_targets[-4:])/4 # initially unused batches stay in cash
            if buffer and float(np.abs(target-weights).sum())<s.execution_buffer_l1:target=weights.copy()
            audit={}
            if guard:target,audit=guard_target(d,i,target,weights,nav,peak,state,s,recovery)
            pkg=d['packages'][i];w=target.reindex(pkg['codes']).to_numpy()
            audit['final_annual_vol']=float(np.sqrt(max(0,w@pkg['cov']@w)*252))
            audit['account_drawdown']=dd;audit['peak']=peak
            pending=target.copy();pending_signal=date
            signals.append({'date':date,**target.to_dict(),'cash':1-target.sum(),**audit})
        records.append({'date':date,'nav':nav,'return':nav/prior-1,'drawdown':dd,'turnover':turnover,'cost_fraction':cost,'exposure':weights.sum(),'cash_return':cash.iloc[i],'halted':False,'executed_signal_date':executed})
        holdings.append({'date':date,**weights.to_dict(),'cash':1-weights.sum()})
    curve=pd.DataFrame(records).set_index('date');positions=pd.DataFrame(holdings).set_index('date');sig=pd.DataFrame(signals).set_index('date')
    dest=d['dest']/name;dest.mkdir(exist_ok=True)
    curve.to_csv(dest/'curve.csv');positions.to_csv(dest/'holdings.csv');sig.to_csv(dest/'signals.csv')
    results=[]
    for segment,part in [('all_explored',curve),('2017_2021_explored',curve.loc[:'2021-12-31']),('2022_2026_explored',curve.loc['2022-01-01':])]:
        if part.empty:continue
        m=metrics(part,Settings(s.max_drawdown));cn=(1+part.cash_return).prod()**(252/len(part))-1
        m.update(market=d['market'],model=name,segment=segment,cash_cagr=cn,annual_cash_excess=m['annual_return']-cn,excess_target=2*s.max_drawdown,goal_met=(m['max_drawdown']<=s.max_drawdown and m['annual_return']-cn>=2*s.max_drawdown),start=str(part.index[0].date()),end=str(part.index[-1].date()),guarded=guard)
        results.append(m)
    pd.DataFrame(results).to_csv(dest/'metrics.csv',index=False)
    return results

def fixed_rule(codes,weights):
    def rule(d,pos,current,state):
        z=pd.Series(0.,index=current.index)
        for c,w in zip(codes,weights):
            if c in d['packages'][pos]['codes']:z[c]=w
        return z
    return rule

def trend_rule(days=210):
    def rule(d,pos,current,state):
        codes=d['packages'][pos]['representatives'];p=d['prices']
        selected=[c for c in codes if p.iloc[pos][c]>p.iloc[max(0,pos-days+1):pos+1][c].mean() and p.iloc[max(0,pos-days+1):pos+1][c].notna().sum()>=days]
        return weighted_target(d,pos,selected,'equal')
    return rule

def dual_rule(d,pos,current,state):
    """Legacy two-asset stock/bond adaptation, NOT Antonacci's GEM replication.

    Compares stock to bond momentum; absolute cash hurdle only applies to stock.
    Always invests 100% in the fallback bond, including negative bond trends.
    This unguarded diagnostic does not implement any account drawdown budget.
    Preserve it for reproducing prior results; audited alternatives are separate.
    """
    stock='510300.SH' if d['market']=='CN' else 'SPY.P';bond='CBA00601.CS' if d['market']=='CN' else 'TLT.O'
    x=d['factors']['mom_12m'].iloc[pos];ab=d['factors']['cash_relative_12m'].iloc[pos]
    chosen=stock if np.isfinite(x[stock]) and x[stock]>x[bond] and ab[stock]>0 else bond
    return weighted_target(d,pos,[chosen])

def audited_stock_bond_rule(*,compare_bond=True,bond_cash_hurdle=True,cash_only=False):
    """Explicit ablation of the legacy rule, not a GEM or 8% risk-control claim.

    Missing momentum is never treated as a positive signal. No fitted parameters
    or portfolio caps; additions change signal selection, not position sizing.
    """
    def rule(d,pos,current,state):
        stock='510300.SH' if d['market']=='CN' else 'SPY.P'
        bond='CBA00601.CS' if d['market']=='CN' else 'TLT.O'
        x=d['factors']['mom_12m'].iloc[pos]
        ab=d['factors']['cash_relative_12m'].iloc[pos]
        available=d['packages'][pos]['codes']
        stock_ok=stock in available and np.isfinite(ab[stock]) and ab[stock]>0
        if compare_bond:
            stock_ok=stock_ok and np.isfinite(x[stock]) and np.isfinite(x[bond]) and x[stock]>x[bond]
        if stock_ok:return weighted_target(d,pos,[stock])
        bond_ok=bond in available and not cash_only
        if bond_cash_hurdle:bond_ok=bond_ok and np.isfinite(ab[bond]) and ab[bond]>0
        return weighted_target(d,pos,[bond]) if bond_ok else pd.Series(0.,index=current.index)
    return rule

def rsrs_rule(d,pos,current,state,dampened=True):
    codes=d['packages'][pos]['representatives'];key='rsrs_dampened' if dampened else 'rsrs'
    score=d['factors'][key].iloc[pos];old=state.setdefault('asset_state',{})
    for c in codes:
        if pd.notna(score[c]):
            if score[c]>.7:old[c]=True
            elif score[c]<-.7:old[c]=False
    allowed=[c for c in codes if old.get(c,False) and pd.notna(d['factors']['bias_velocity'].iloc[pos][c])]
    if not allowed:return pd.Series(0.,index=current.index)
    chosen=d['factors']['bias_velocity'].iloc[pos][allowed].nlargest(3).index.tolist()
    return weighted_target(d,pos,chosen)

def fantastic_rule(d,pos,current,state):
    codes=['512100.SH','CBA00601.CS','518880.SH'] if d['market']=='CN' else ['IWM.P','EEM.P','TLT.O','GLD.P']
    x=d['prices'].iloc[pos]/d['prices'].iloc[max(0,pos-75)]-1
    use=[c for c in codes if c in d['packages'][pos]['codes'] and pd.notna(x[c]) and x[c]>0]
    return weighted_target(d,pos,use,raw=x[use].to_numpy())

def haa_rule(d,pos,current,state):
    if d['market']!='US':raise ValueError('TIP canary unavailable in CN; no fabricated canary')
    mom=d['factors']['mom_13612U'].iloc[pos];codes=d['packages'][pos]['representatives']
    defensive=[c for c in codes if d['assets'][c]['class']=='bond' and mom[c]>0]
    safe=max(defensive,key=lambda c:mom[c]) if defensive else None
    slots=[]
    if mom['TIP.P']>0:
        attack=[c for c in codes if d['assets'][c]['class']!='bond'];attack=sorted(attack,key=lambda c:-mom[c])[:4]
        slots=[c if mom[c]>0 else safe for c in attack]
    else:slots=[safe]*4
    z=pd.Series(0.,index=current.index)
    for c in slots:
        if c is not None:z[c]+=.25
    return z

def allweather_rule(d,pos,current,state,boost=False):
    codes=d['packages'][pos]['representatives'];z=pd.Series(0.,index=current.index)
    buckets={a:[c for c in codes if d['assets'][c]['class']==a] for a in ['equity','bond','gold','oil']}
    buckets={a:c for a,c in buckets.items() if c}
    for a,c in buckets.items():z+=weighted_target(d,pos,c,'inverse_vol')/len(buckets)
    if boost:
        stock='510300.SH' if d['market']=='CN' else 'SPY.P';p=d['prices'][stock].iloc[:pos+1]
        draw=1-p.iloc[-1]/p.max();repair=p.iloc[-1]>p.tail(120).mean()
        donor=max(buckets.get('bond',[]),key=lambda c:z[c],default=None)
        if draw>.25 and repair and donor:
            amount=min(.8*z[stock],z[donor]);z[stock]+=amount;z[donor]-=amount
    return z

def rsrs_v21_rule(d,pos,current,state):
    b=d['benchmark'];rs=d['factors']['rsrs'][b];recent=rs.iloc[max(0,pos-7):pos+1].to_numpy()
    if not np.isfinite(recent).all():return pd.Series(0.,index=current.index)
    slope=np.polyfit(np.arange(len(recent)),recent,1)[0]
    sell=(slope<0 and rs.iloc[pos]>0) or rs.iloc[pos]<=-.7
    if sell:return pd.Series(0.,index=current.index)
    return ranked_rule('bias_velocity',1,False,True)(d,pos,current,state)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['factors','strategies','all'],default='all');parser.add_argument('--markets',nargs='+',default=['CN','US']);parser.add_argument('--max-drawdown',type=float,default=.08);parser.add_argument('--rebuild-factors',action='store_true');args=parser.parse_args();s=PracticeSettings(max_drawdown=args.max_drawdown)
    OUT.mkdir(exist_ok=True);all_results=[];experiments=[]
    for market in args.markets:
        print('prepare',market,flush=True);d=load_market(market,s,args.rebuild_factors);prepare_weeks(d,s)
        factor_diagnostics(d,s)
        if args.stage=='factors':print('factors done',market,len(d['factors']),flush=True);continue
        factors=[n for n in BASE_FACTORS+NEW_PRICE_FACTORS+RSRS_FACTORS+VALUATION_FACTORS+FUNDAMENTAL_FACTORS+SHARE_FACTORS if n in d['factors'] and d['factors'][n].notna().any().any()]
        models=[]
        for n in factors:
            models.append(('factor_'+n,ranked_rule(n,3,False,True),{},'Equity-only Top3/ERC factor diagnostic; not an 8% constrained portfolio'))
        benchmark=['510300.SH','CBA00101.CS','518880.SH'] if market=='CN' else ['SPY.P','IEF.O','GLD.P']
        models += [('cash',fixed_rule([],[]),{},'Cash baseline'),('bond_only',fixed_rule([benchmark[1]],[1]),{},'Bond baseline'),('fixed_30_50_20',fixed_rule(benchmark,[.3,.5,.2]),{},'Fixed stock/bond/gold benchmark'),('pure_erc',lambda d,p,c,t:weighted_target(d,p,d['packages'][p]['representatives']),{},'Full eligible pool covariance ERC'),('trend_10month',trend_rule(210),{},'QuantConnect trend rule weekly adaptation'),('dual_momentum',dual_rule,{},'Legacy stock/bond momentum adaptation; NOT GEM; 100% fallback bond; no drawdown guard'),('gtaa_equal',ranked_rule('mom_13612U',6,True,method='equal'),{},'GTAA multi-horizon weekly Top6/equal adaptation'),('gtaa_erc',ranked_rule('mom_13612U',6,True),{},'Same GTAA selection; complete covariance ERC'),('weekly_top2',ranked_rule('mom_1m',2,False,method='equal'),{},'Minimal JoinQuant weekly template adaptation'),('wufu_weighted_trend',ranked_rule('weighted_trend25',1,True),{},'Weighted trend core; intraday stop/premium filters not reproduced'),('rsrs_tristate',lambda d,p,c,t:rsrs_rule(d,p,c,t,False),{},'Per-asset RSRS entry/exit/keep instead of whole-stock gate'),('rsrs_dampened',rsrs_rule,{},'Vol-percentile dampened RSRS three-state weekly adaptation'),('fantastic_rotation',fantastic_rule,{},'US original four categories; CN three available categories'),('momentum_erc',ranked_rule('mom_13612W',3,True),{},'Common cross-asset momentum/ERC prototype'),('momentum_hysteresis',ranked_rule('mom_13612W',3,True,hysteresis=True),{'buffer':True},'Only keep margin and execution buffer added'),('momentum_four_signals',ranked_rule('mom_13612W',3,True),{'blend4':True},'Four weekly target snapshots averaged; not a replication of cohort subaccounts'),('momentum_guard',ranked_rule('mom_13612W',3,True),{'guard':True},'Prospective compounded joint-path risk after final weights'),('momentum_guard_recovery',ranked_rule('mom_13612W',3,True,hysteresis=True),{'guard':True,'recovery':True,'buffer':True},'Add explicit DERISK/PROBE/RESTORE without resetting global peak'),('fft_guard',ranked_rule('fft_trend',3,True),{'guard':True},'Windowed FFT signal plus same forward risk guard'),('kalman_guard',ranked_rule('kalman_filter_trend',3,True),{'guard':True},'Causal Kalman signal plus same forward risk guard'),('relative_strength_guard',ranked_rule('relative_strength',3,True),{'guard':True},'Relative trend plus same forward risk guard')]
        if market=='US':models.append(('haa_weekly',haa_rule,{},'HAA TIP canary is an original model assumption; weekly adaptation'))
        models += [('allweather_bucket',allweather_rule,{},'Source-like class equal/within inverse-vol baseline; not HRP or ERC'),('allweather_boost_guard',lambda d,p,c,t:allweather_rule(d,p,c,t,True),{'guard':True},'Dip-repair overlay first; final covariance/path risk checked afterwards'),('rsrs_v21_market_gate',rsrs_v21_rule,{},'Original V2.1 market gate core adapted to weekly; diagnostic, not user stock cap')]
        for name,rule,kwargs,description in models:
            result=simulate(d,name,rule,s,**kwargs);all_results+=result
            experiments.append({'market':market,'model':name,'description':description,'kwargs':kwargs,'status':'executed','production_accepted':False})
            m=result[0];print(market,name,'CAGR',round(m['annual_return']*100,2),'DD',round(m['max_drawdown']*100,2),'excess',round(m['annual_cash_excess']*100,2),flush=True)
        pd.DataFrame([r for r in all_results if r['market']==market]).to_csv(d['dest']/'all_metrics.csv',index=False)
        del d
    if all_results:pd.DataFrame(all_results).to_csv(OUT/'all_metrics.csv',index=False)
    if experiments:(OUT/'experiments.json').write_text(json.dumps(experiments,ensure_ascii=False,indent=2))
    (OUT/'method.json').write_text(json.dumps({'settings':asdict(s),'factor_manifest':factor_manifest(),'date_local':'2026-10-07','data_asof':'2026-09-24','execution':'First observed session of W-FRI week close signal; next trading close execution; return begins after execution; actual drifting holdings and net trades charged8bps','cash':'CN M1010885 lagged ACT/365; US DGS3MO conservative publication lag','drawdown':'Observed from original all-account daily high-water mark. Forward simulation is a risk estimate, not a guarantee or a past252 fixed-weight drawdown constraint. Segment metrics reset only for describing segments; whole-account drawdown also reported.','parameters_origin':'Top3/6, windows, .7 thresholds,64 paths,65 days,95% quantile and10pp restoration pace are experiment assumptions, not user constraints','blend4':'Four signal snapshots netted; not four independent sleeve account histories','limitations':['All historical data previously inspected; no untouched test claim','Current surviving universe, incomplete historic membership','CN bond wealth indices are nontradable theory proxies; no index trade liquidity','Forward adjustment/dividends and close-fill transaction costs are approximations','Valuation diagnostics lack PIT vintages; excluded from prediction models','Many trials; best historical row is not an accepted strategy']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
