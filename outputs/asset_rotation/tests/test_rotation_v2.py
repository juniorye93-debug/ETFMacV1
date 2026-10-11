import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import rotation_v2 as v

ASSETS={'a':{'class':'equity','bucket':'broad'},'b':{'class':'bond','bucket':'bond'},'c':{'class':'gold','bucket':'gold'}}

def panel(n=565):
 rng=np.random.default_rng(18);rr=rng.normal([.0007,.0002,.0004],[.006,.0005,.004],(n,3))
 return pd.DataFrame(100*np.cumprod(1+rr,axis=0),columns=ASSETS,index=pd.bdate_range('2020-01-01',periods=n))

def test_capped_weights_reallocate_budget():
 w=v.capped_simplex([.8,.1,.1],.4)
 np.testing.assert_allclose(w,[.4,.3,.3],atol=1e-9)
 assert w.sum()==pytest.approx(1.)
 with pytest.raises(ValueError):v.capped_simplex([.8,.1,.1],.2)


def test_finite_stop_recovers_without_waiting_for_old_high():
 s=v.Model(.08);c=v.RecoveryController(s)
 c.update(0,1.,True,True)
 scale,_,_=c.update(1,.92,False,False)
 assert scale==0 and c.phase=='cooldown'
 expiry=c.until
 for i in range(2,expiry):
  c.update(i,.92,True,True)
  assert c.phase=='cooldown' and c.until==expiry
 scale,_,event=c.update(expiry,.92,True,True)
 assert event=='resume_after_cooldown' and scale==.25
 for i,expected in enumerate([.5,.75,1.],expiry+1):
  scale,_,_=c.update(i,.921,True,True)
  assert scale==expected
 assert c.phase=='normal'


def test_soft_cut_not_repeated_multiplication():
 c=v.RecoveryController(v.Model(.08));c.update(0,1.,True,True)
 assert c.update(1,.95,False,False)[0]==.5
 for i in range(2,10):
  scale,changed,_=c.update(i,.95,False,False)
  assert scale==.5 and not changed


def test_sleeves_are_causal():
 p=panel();cash=pd.Series(.0001,index=p.index);s=v.Model(.08)
 R,sn=v.build_sleeves(p,cash,ASSETS,s)
 p2=p.copy();p2.iloc[535:]*=np.linspace(1.,1.8,len(p2)-535)[:,None]
 R2,sn2=v.build_sleeves(p2,cash,ASSETS,s)
 pd.testing.assert_frame_equal(R.iloc[:535],R2.iloc[:535])
 for i in sn:
  if i<535:
   pd.testing.assert_frame_equal(sn[i][0],sn2[i][0]);pd.testing.assert_series_equal(sn[i][1],sn2[i][1])


def test_two_stage_caps_and_explicit_target_feasibility():
 rng=np.random.default_rng(90);r=pd.DataFrame(rng.normal(.0007,.001,(252,3)),columns=ASSETS)
 B=pd.DataFrame(0.,index=ASSETS,columns=v.CLASSES)
 for c,cls in [('a','equity'),('b','bond'),('c','gold')]:B.loc[c,cls]=1.
 sleeves=r@B;cash=pd.Series(.0001,index=r.index);current=pd.Series(0.,index=ASSETS)
 w,a=v.constrained_outer(B,pd.Series([.40,.35,.30],index=ASSETS),sleeves,r,cash,current,v.Model(.08),1.)
 assert a['target_feasible']
 assert w.max()<=.35+1e-6 and w.sum()<=1+1e-6 and w['c']<=.30+1e-6
 assert a['forecast_excess']>=.16-1e-6
 w,a=v.constrained_outer(B,pd.Series([.02,.01,.015],index=ASSETS),sleeves,r,cash,current,v.Model(.08),1.)
 assert not a['target_feasible'] and a['max_forecast_excess']<.16


def test_goal_uses_annual_excess_not_zero_rate_return():
 r=np.full(252,.0007);cash=np.full(252,.0003)
 c=pd.DataFrame({'return':r,'cash_return':cash,'turnover':0.,'exposure':1.,'halted':False})
 result=v.evaluate(c,v.Model(.08))
 assert result['annual_return']>.16
 assert result['annual_excess']<.16
 assert not result['goal_passed']


def test_execution_and_global_drawdown_survive_recovery(monkeypatch):
 p=panel(560);p.loc[:,:]=100.;p.iloc[506:,0]=80.;p.iloc[530:,0]=85.
 cash=pd.Series(0.,index=p.index);B=pd.DataFrame(0.,index=p.columns,columns=v.CLASSES);B.loc['a','equity']=1.
 snapshots={i:(B,pd.Series(.2,index=p.columns),1.) for i in range(252,len(p))}
 R=pd.DataFrame(0.,index=p.index,columns=v.CLASSES)
 def allocate(*args):return pd.Series([.5,0.,0.],index=p.columns),{'status':'test','target_feasible':True,'forecast_excess':.2,'max_forecast_excess':.25}
 monkeypatch.setattr(v,'constrained_outer',allocate)
 c,h,signals,events=v.backtest_v2(p,cash,ASSETS,v.Model(.08,frequency='daily',fee_bps=0,slippage_bps=0),sleeves=(R,snapshots))
 assert c.iloc[0]['return']==0 and c.iloc[1]['return']==0
 assert c.iloc[2]['return']==pytest.approx(-.1)
 assert c.global_peak.eq(1.).all()  # Accounting peak never follows episode resets.
 assert c.drawdown.max()==pytest.approx(.1)
 assert v.evaluate(c,v.Model(.08))['target_breached']
 assert signals.iloc[0].target_feasible
 stopped=signals[signals.risk_scale==0]
 assert not stopped.target_feasible.any()
 assert stopped.forecast_excess.eq(0).all()


def test_outer_backtest_has_no_future_leakage():
 p=panel(538);cash=pd.Series(.0001,index=p.index);s=v.Model(.08)
 c,h,sg,_=v.backtest_v2(p,cash,ASSETS,s)
 changed=p.copy();changed.iloc[525:]*=np.linspace(1.,1.6,len(p)-525)[:,None]
 cc,hh,ss,_=v.backtest_v2(changed,cash,ASSETS,s)
 for a,b in [(c,cc),(h,hh),(sg,ss)]:pd.testing.assert_frame_equal(a.loc[a.index<p.index[525]],b.loc[b.index<p.index[525]])


def test_global_dedup_direct_edges_negative_correlation_and_cross_class():
 # A--B--C forms a chain: B is redundant with A, but C must survive.
 t=np.linspace(0,2*np.pi,252,endpoint=False)
 a=np.sin(t);b=np.sin(t+np.deg2rad(20));c=np.sin(t+np.deg2rad(40))
 r=pd.DataFrame({'a':a,'b':b,'c':c,'negative':-a})
 assets={k:{'class':v} for k,v in zip(r,['equity','bond','equity','gold'])}
 kept,audit=v.correlation_representatives(r,assets,.90)
 assert kept==['a','c','negative']
 assert audit.set_index('code').loc['b','representative']=='a'
 corr=r[kept].corr().to_numpy()
 assert np.all(corr[np.triu_indices(len(kept),1)]<.90)


def test_global_dedup_excludes_incomplete_history_and_uses_config_priority():
 r=pd.DataFrame({'a':np.arange(252.),'b':np.arange(252.),'missing':np.arange(252.)})
 r.loc[0,'missing']=np.nan
 kept,_=v.correlation_representatives(r,{'b':{},'a':{},'missing':{}},.90)
 assert kept==['b']
 with pytest.raises(ValueError):v.correlation_representatives(r,{'a':{}},1.1)


def test_strict_weekly_signals_include_account_protection():
 p=panel(585);p.iloc[530:540,0]*=.70
 cash=pd.Series(.0001,index=p.index);s=v.Model(.08)
 curve,holdings,signals,events=v.backtest_v2(p,cash,ASSETS,s)
 periods=p.index.to_period('W-FRI')
 allowed={p.index[i] for i in range(504,len(p)) if i==504 or periods[i]!=periods[i-1]}
 assert set(signals.index)<=allowed
 if not events.empty:assert set(pd.to_datetime(events.date))<=allowed
