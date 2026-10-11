"""Single-asset daily research engine. No order routing. Original implementation."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, hashlib, json, math
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

@dataclass
class Config:
    estimator: str = "logistic"
    horizon: int = 20
    stop_atr: float = 2.
    take_atr: float = 3.
    fee_bps: float = 3.
    slippage_bps: float = 5.
    min_probability: float = .55
    risk_budget: float = .005
    vol_target: float = .10
    max_notional: float = 1.
    equity: float = 1_000_000.
    multiplier: float = 1.
    lot: int = 1
    margin_rate: float = 1.
    margin_cap: float = .30
    short: bool = False
    min_hold: int = 0
    gap_atr: float = .75
    retrain: int = 21
    calibration_bars: int = 756
    train_bars: int = 2520
    min_train: int = 250
    min_calibration: int = 30
    embargo: int = 20
    def validate(self):
        if self.estimator not in {"logistic","random_forest"}: raise ValueError("Unknown probability estimator")
        if min(self.horizon,self.retrain,self.lot,self.multiplier,self.equity,self.stop_atr,self.take_atr)<=0: raise ValueError('positive parameters required')
        if self.min_hold not in (0,1) or self.min_hold>=self.horizon: raise ValueError('min_hold must be 0 or 1 and less than horizon')
        if not (0<self.risk_budget<1 and 0<self.vol_target and 0<self.max_notional<=1 and 0<self.margin_rate<=1 and 0<self.margin_cap<=1 and 0<self.min_probability<1): raise ValueError('invalid risk settings')
        if min(self.fee_bps,self.slippage_bps,self.gap_atr,self.embargo)<0: raise ValueError('negative costs/gap/embargo')

def read_data(path):
    d=pd.read_csv(path,parse_dates=['date']).set_index('date')
    if d.index.has_duplicates or d.index.hasnans or not d.index.is_monotonic_increasing: raise ValueError('dates must be unique and increasing')
    cols=['open','high','low','close','volume'];d=d[cols].astype(float)
    if not np.isfinite(d.values).all() or (d[cols[:4]]<=0).any().any() or (d.volume<0).any(): raise ValueError('invalid OHLCV')
    if ((d.high<d[['open','close','low']].max(axis=1)) | (d.low>d[['open','close','high']].min(axis=1))).any(): raise ValueError('inconsistent OHLC')
    return d

def features(d,c):
    close=d.close;ret=close.pct_change();tr=pd.concat([d.high-d.low,(d.high-close.shift()).abs(),(d.low-close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    fast=close.ewm(span=20,adjust=False).mean();slow=close.ewm(span=60,adjust=False).mean()
    side=pd.Series(np.where((fast>slow)&(close.pct_change(63)>0),1,np.where((fast<slow)&(close.pct_change(63)<0),-1,0)),index=d.index)
    if not c.short: side=side.clip(lower=0)
    x=pd.DataFrame(index=d.index)
    for n in [5,21,63,126]: x[f'momentum_{n}']=close.pct_change(n)*side/(ret.rolling(63).std()*np.sqrt(n))
    x['ema_gap']=(fast-slow)/atr*side;x['distance']=(close-slow)/atr*side
    delta=close.diff();up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rsi=up/(up+down);x['rsi']=(rsi-.5)*side
    x['volatility']=ret.rolling(21).std()*np.sqrt(252)
    x['vol_ratio']=ret.rolling(21).std()/ret.rolling(126).std()
    x['range']=atr/close
    x['breakout']=(close-d.high.rolling(20).max().shift())/atr*side
    x['volume_ratio']=d.volume/d.volume.rolling(20).mean().replace(0,np.nan)
    x['side']=side
    return x.replace([np.inf,-np.inf],np.nan),atr,side

def barrier(d,t,side,atr,c,partial=False):
    """Signal at close t; entry next open, fixed ATR barriers, gap-aware exits."""
    start=t+1;end=start+c.horizon-1
    if end>=len(d) and not partial: return None
    end=min(end,len(d)-1)
    if float(d.volume.iloc[start])<=0: return None
    scheduled_end=start+c.horizon-1
    if float(d.volume.iloc[end])<=0:
        # A suspended bar cannot execute a timed exit. Keep holding until trading resumes.
        tradable=np.flatnonzero(d.volume.iloc[end:].to_numpy()>0)
        if len(tradable): end+=int(tradable[0])
        elif partial: end=len(d)-1
        else: return None
    entry=float(d.open.iloc[start]);reference=float(d.close.iloc[t])
    if abs(entry-reference)>c.gap_atr*atr: return None
    stop=entry-side*c.stop_atr*atr;take=entry+side*c.take_atr*atr
    if min(stop,take)<=0: return None
    exit_price=float(d.close.iloc[end]);reason='time'
    for j in range(start+c.min_hold,end+1):
        if float(d.volume.iloc[j])<=0: continue
        o,h,l=float(d.open.iloc[j]),float(d.high.iloc[j]),float(d.low.iloc[j])
        stop_gap=(o<=stop if side==1 else o>=stop);take_gap=(o>=take if side==1 else o<=take)
        stop_hit=(l<=stop if side==1 else h>=stop);take_hit=(h>=take if side==1 else l<=take)
        if stop_gap: exit_price=o;end=j;reason='stop_gap';break
        if take_gap: exit_price=o;end=j;reason='take_gap';break
        # Daily OHLC cannot identify first hit: conservative stop-first policy.
        if stop_hit: exit_price=stop;end=j;reason='both_stop_first' if take_hit else 'stop';break
        if take_hit: exit_price=take;end=j;reason='take';break
    gross=side*(exit_price-entry)/entry
    cost=(c.fee_bps+c.slippage_bps)/1e4*(1+exit_price/entry)
    return dict(t=t,start=start,end=end,side=int(side),entry=entry,stop=stop,take=take,exit=exit_price,reason='time_deferred_suspension' if reason=='time' and end>scheduled_end else reason,net=gross-cost,win=int(gross>cost),atr=atr,censored=(reason=='time' and (end<scheduled_end or float(d.volume.iloc[end])<=0)))

def make_events(d,x,atr,side,c):
    rows=[]
    for t in range(126,len(d)-c.horizon):
        if side.iloc[t] and x.iloc[t].notna().all():
            e=barrier(d,t,int(side.iloc[t]),float(atr.iloc[t]),c)
            if e: rows.append(e)
    return pd.DataFrame(rows)

def nonoverlap(e):
    ids=[];last=-1
    for i,r in e.iterrows():
        if r.start>last: ids.append(i);last=r.end
    return e.loc[ids]

def uniqueness(e):
    lo=int(e.start.min());hi=int(e.end.max());count=np.zeros(hi-lo+1)
    for r in e.itertuples(): count[r.start-lo:r.end-lo+1]+=1
    return np.array([np.mean(1/count[r.start-lo:r.end-lo+1]) for r in e.itertuples()])

def fit_model(x,events,at,c):
    eligible=events[(events.end<at-c.embargo)&(events.t>=at-c.train_bars)]
    cal=nonoverlap(eligible[eligible.t>=at-c.calibration_bars])
    train=eligible[(eligible.t<at-c.calibration_bars)&(eligible.end<at-c.calibration_bars-c.embargo)]
    if len(train)<c.min_train or len(cal)<c.min_calibration or train.win.nunique()<2 or cal.win.nunique()<2: return None
    estimator=LogisticRegression(C=.1,max_iter=1000) if c.estimator=='logistic' else RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_leaf=30,random_state=20261005,n_jobs=1)
    base=make_pipeline(StandardScaler(),estimator)
    weights=uniqueness(train);weights=weights/weights.mean()
    base.fit(x.iloc[train.t],train.win,**{base.steps[-1][0]+'__sample_weight':weights})
    logits=base_logits(base,x.iloc[cal.t])
    calibrator=LogisticRegression(C=1.,max_iter=1000).fit(logits,cal.win)
    p=calibrator.predict_proba(logits)[:,1]
    # This is a calibration-fit diagnostic, not an out-of-sample claim.
    return dict(base=base,calibrator=calibrator,train_n=len(train),cal_n=len(cal),cal=cal,
                prior_probability=float(cal.win.mean()),avg_win=float(cal.loc[cal.win==1,'net'].mean()),avg_loss=float(-cal.loc[cal.win==0,'net'].mean()),
                trained_at=at,latest_label_end=int(eligible.end.max()),calibration_fit_brier=float(brier_score_loss(cal.win,p)))

def base_logits(base,x):
    if hasattr(base,'decision_function'): return base.decision_function(x).reshape(-1,1)
    p=np.clip(base.predict_proba(x)[:,1],1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def predict(m,row):
    logit=base_logits(m['base'],pd.DataFrame([row]))
    return float(m['calibrator'].predict_proba(logit)[0,1])

def size(entry,atr,vol,equity,c):
    weight=min(c.max_notional,c.risk_budget/(c.stop_atr*atr/entry),c.vol_target/max(vol,.001),c.margin_cap/c.margin_rate if c.margin_rate<1 else 1.)
    units=math.floor(weight*equity/(entry*c.multiplier*c.lot))*c.lot
    actual=units*entry*c.multiplier/equity
    return units,actual

def plan(d,x,atr,side,t,m,c,equity):
    s=int(side.iloc[t]);ref=float(d.close.iloc[t]);a=float(atr.iloc[t])
    result=dict(date=str(d.index[t].date()),probability_estimator=c.estimator,direction={1:'LONG',-1:'SHORT',0:'FLAT'}[s],action='WAIT',win_probability=None,
        reference_price=ref,atr=a,stop_loss=None,take_profit=None,max_holding_bars=c.horizon,target_weight=0.,units=0,
        expected_net_return=None,reason='no_trend',model_trained_at=None)
    if not s: return result
    result.update(stop_loss=ref-s*c.stop_atr*a,take_profit=ref+s*c.take_atr*a,entry_min=ref-c.gap_atr*a,entry_max=ref+c.gap_atr*a,
                  price_status='provisional; reanchor to actual next open',reason='insufficient_training_or_calibration')
    if m is None or x.iloc[t].isna().any(): return result
    p=predict(m,x.iloc[t]);ev=p*m['avg_win']-(1-p)*m['avg_loss']
    units,w=size(ref,a,float(x.volatility.iloc[t]),equity,c)
    result.update(win_probability=p,prior_probability=m['prior_probability'],expected_net_return=ev,training_events=m['train_n'],calibration_independent_events=m['cal_n'],
        model_trained_at=str(d.index[m['trained_at']].date()),latest_training_label=str(d.index[m['latest_label_end']].date()),reason='probability_or_expectancy_filter')
    if p>=c.min_probability and ev>0 and units>0:
        result.update(action='ENTER_NEXT_OPEN_IF_GAP_OK',target_weight=s*w,units=units,reason='passed')
    return result

def run(d,c,out,symbol='asset'):
    c.validate();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x,atr,side=features(d,c);events=make_events(d,x,atr,side,c)
    if events.empty: raise ValueError('no usable labeled events')
    m=None;signals=[];trades=[];equity=c.equity;curve=[];busy_until=-1;first=None;open_position=None;active_position=None
    for t in range(126,len(d)-1):
        if t% c.retrain==0 or (m is None and t%c.retrain==1): m=fit_model(x,events,t,c)
        if m is None:
            if first is not None and t+1>busy_until: curve.append(dict(date=str(d.index[t+1].date()),equity=equity))
            continue
        if first is None: first=t;curve.append(dict(date=str(d.index[t].date()),equity=equity))
        row=plan(d,x,atr,side,t,m,c,equity)
        if t<busy_until:
            row.update(action='HOLD',reason='position_active',trend_direction=row['direction'],direction='LONG' if active_position['side']==1 else 'SHORT',
                       stop_loss=active_position['stop'],take_profit=active_position['take'],units=active_position['units'],
                       position_entry_win_probability=active_position['win_probability'],candidate_win_probability=row['win_probability'],
                       win_probability=active_position['win_probability'],probability_timestamp=active_position['signal_date'],
                       target_weight=active_position['side']*active_position['units']*float(d.close.iloc[t])*c.multiplier/
                           (active_position['entry_equity']+active_position['side']*active_position['units']*c.multiplier*(float(d.close.iloc[t])-active_position['entry'])-active_position['units']*active_position['entry']*c.multiplier*(c.fee_bps+c.slippage_bps)/1e4));signals.append(row);continue
        signals.append(row)
        if row['action']!='ENTER_NEXT_OPEN_IF_GAP_OK':
            if t+1>busy_until: curve.append(dict(date=str(d.index[t+1].date()),equity=equity))
            continue
        # Partial tails may close early; otherwise retain an open marked-to-market position.
        e=barrier(d,t,int(side.iloc[t]),float(atr.iloc[t]),c,partial=True)
        if e is None:
            row.update(action='WAIT',reason='entry_gap_or_invalid_barriers')
            curve.append(dict(date=str(d.index[t+1].date()),equity=equity));continue
        units,w=size(e['entry'],e['atr'],float(x.volatility.iloc[t]),equity,c)
        if not units:
            row.update(action='WAIT',reason='below_one_lot')
            curve.append(dict(date=str(d.index[t+1].date()),equity=equity));continue
        active_position={**e,'units':units,'win_probability':row['win_probability'],'signal_date':row['date'],'entry_equity':equity}
        row.update(stop_loss=e['stop'],take_profit=e['take'],units=units,target_weight=w*e['side'],actual_entry=e['entry'],price_status='historical next-open execution')
        capital=equity;notional=units*e['entry']*c.multiplier
        one_cost=(c.fee_bps+c.slippage_bps)/1e4
        for j in range(e['start'],e['end']+1):
            px=e['exit'] if j==e['end'] else float(d.close.iloc[j])
            pnl=units*c.multiplier*e['side']*(px-e['entry'])-notional*one_cost
            if j==e['end'] and not e['censored']: pnl-=units*px*c.multiplier*one_cost
            curve.append(dict(date=str(d.index[j].date()),equity=capital+pnl))
        if e['censored']:
            open_position={**e,'units':units,'weight':w*e['side'],'win_probability':row['win_probability'],'signal_date':row['date']}
            equity=curve[-1]['equity'];busy_until=len(d);continue
        equity=capital+notional*e['net'];busy_until=e['end']
        trades.append({**e,'signal_date':str(d.index[t].date()),'entry_date':str(d.index[e['start']].date()),'exit_date':str(d.index[e['end']].date()),
                       'win_probability':row['win_probability'],'prior_probability':row['prior_probability'],'units':units,'weight':w*e['side'],'pnl':equity-capital,'equity_after':equity})
    sig=pd.DataFrame(signals);tr=pd.DataFrame(trades);cv=pd.DataFrame(curve)
    if cv.empty: raise ValueError('insufficient history: no calibrated model; supply longer OHLCV')
    cv=cv.drop_duplicates('date',keep='last').sort_values('date');returns=cv.equity.pct_change().dropna()
    total=float(cv.equity.iloc[-1]/c.equity-1);years=(pd.Timestamp(cv.date.iloc[-1])-pd.Timestamp(cv.date.iloc[0])).days/365.25
    metric=dict(symbol=symbol,probability_estimator=c.estimator,start=cv.date.iloc[0],end=cv.date.iloc[-1],closed_trades=len(tr),total_return=total,
                cagr=(1+total)**(1/years)-1 if years else None,max_drawdown=float((cv.equity/cv.equity.cummax()-1).min()),
                sharpe=float(returns.mean()/returns.std()*np.sqrt(252)) if returns.std()>0 else None,
                realized_win_rate=float(tr.win.mean()) if len(tr) else None,
                oos_trade_brier=float(brier_score_loss(tr.win,tr.win_probability)) if len(tr) else None)
    if len(tr):
        metric['oos_prior_brier']=float(brier_score_loss(tr.win,tr.prior_probability))
        metric['brier_skill_vs_prior']=1-metric['oos_trade_brier']/metric['oos_prior_brier'] if metric['oos_prior_brier']>0 else None
    # Full-notional buy-and-hold is a reference, not a risk-matched competitor.
    start=d.index.get_loc(pd.Timestamp(cv.date.iloc[0]));end=d.index.get_loc(pd.Timestamp(cv.date.iloc[-1]))
    metric['buy_hold_return']=float(d.close.iloc[end]/d.close.iloc[start]-1-2*(c.fee_bps+c.slippage_bps)/1e4)
    baseline=simulate_baseline(d,x,atr,side,start,end,c)
    metric['trend_without_probability']=baseline
    reliability=[]
    if len(tr):
        for left in np.arange(0,1,.1):
            group=tr[(tr.win_probability>=left)&(tr.win_probability<left+.1)]
            if len(group):
                n=len(group);ph=float(group.win.mean());z=1.96;den=1+z*z/n
                center=(ph+z*z/(2*n))/den;half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
                reliability.append(dict(bin_left=float(left),n=n,predicted=float(group.win_probability.mean()),observed=ph,wilson_low=center-half,wilson_high=center+half))
    last=len(d)-1;latest_model=fit_model(x,events,last,c);latest=plan(d,x,atr,side,last,latest_model,c,equity)
    if open_position:
        latest.update(action='HOLD',reason='position_active',trend_direction=latest['direction'],direction='LONG' if open_position['side']==1 else 'SHORT',stop_loss=open_position['stop'],take_profit=open_position['take'],position_direction='LONG' if open_position['side']==1 else 'SHORT',
                      position_entry=open_position['entry'],position_stop=open_position['stop'],position_take=open_position['take'],
                      position_signal_date=open_position['signal_date'],position_units=open_position['units'],
                      position_entry_win_probability=open_position['win_probability'],candidate_win_probability=latest['win_probability'],
                      win_probability=open_position['win_probability'],probability_timestamp=open_position['signal_date'],units=open_position['units'],
                      target_weight=open_position['side']*open_position['units']*float(d.close.iloc[-1])*c.multiplier/equity)
    (out/'open_position.json').write_text(json.dumps(open_position,indent=2))
    age=(dt.datetime.now(ZoneInfo('Asia/Shanghai')).date()-d.index[-1].date()).days
    validation=[]
    if len(tr)<100: validation.append('fewer_than_100_closed_oos_trades')
    if total<=0: validation.append('nonpositive_oos_account_return')
    if metric.get('brier_skill_vs_prior',-1)<=0: validation.append('probability_model_does_not_beat_past_prior')
    latest.update(symbol=symbol,data_asof=str(d.index[-1].date()),research_only=True,data_age_calendar_days=age,deployment_status='NOT_VALIDATED' if validation else 'REQUIRES_PAPER_TRADING',validation_gaps=validation)
    if age>7 and latest['action']!='HOLD':
        latest.update(action='WAIT',reason='stale_data',target_weight=0.,units=0)

    sig.to_csv(out/'decisions.csv',index=False);tr.to_csv(out/'trades.csv',index=False);cv.to_csv(out/'equity.csv',index=False)
    pd.DataFrame(reliability).to_csv(out/'reliability.csv',index=False)
    for name,obj in [('metrics',metric),('latest_signal',latest),('config',asdict(c))]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    return metric,latest

def simulate_baseline(d,x,atr,side,start,end,c):
    equity=c.equity;count=0;t=start
    while t<end:
        if side.iloc[t] and x.iloc[t].notna().all():
            e=barrier(d,t,int(side.iloc[t]),float(atr.iloc[t]),c)
            if e and e['end']<=end:
                units,w=size(e['entry'],e['atr'],float(x.volatility.iloc[t]),equity,c)
                if units:
                    equity*=1+w*e['net'];count+=1;t=e['end'];continue
        t+=1
    return dict(total_return=equity/c.equity-1,closed_trades=count)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--symbol',default='asset');p.add_argument('--config');p.add_argument('--out',required=True)
    a=p.parse_args();c=Config(**json.loads(Path(a.config).read_text())) if a.config else Config()
    meta_path=Path(a.data).with_suffix('.meta.json')
    if not meta_path.exists(): raise ValueError('Formal runs require verified provider source metadata')
    source_meta=json.loads(meta_path.read_text())
    if source_meta.get('source') not in {'Wind WSD','AKShare Eastmoney','AKShare Sina Futures','Yahoo Finance'} or source_meta.get('synthetic') is not False:
        raise ValueError('Missing verified real-market data provenance')
    d=read_data(a.data);metric,latest=run(d,c,a.out,a.symbol)
    (Path(a.out)/'provenance.json').write_text(json.dumps({'input':str(Path(a.data).resolve()),'sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':len(d),'data_source':source_meta},indent=2))
    print(json.dumps({'metrics':metric,'latest':latest},ensure_ascii=False,indent=2))
