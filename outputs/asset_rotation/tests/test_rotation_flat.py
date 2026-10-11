import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import rotation_flat as f


def test_single_asset_can_exceed_old_35_and_55_percent_caps():
 rng=np.random.default_rng(7);r=pd.DataFrame({'equity':rng.normal(.0006,.003,252)})
 cash=pd.Series(.00004,index=r.index)
 w,a,rc=f.optimize_weights(r,cash,pd.Series({'equity':.20}),pd.Series({'equity':0.}),f.FlatModel())
 assert w.equity>.79 and w.sum()<=1+1e-7
 assert a['target_feasible'] and a['forecast_excess']>=.16-1e-6
 assert rc.sum()==pytest.approx(1.)
 assert a['fitted_linear_mdd']<=.08+1e-6


def test_negative_forecast_hedger_is_not_filtered():
 x=np.tile([.02,-.02],126)
 r=pd.DataFrame({'growth':x+.0005,'hedge':-x+.0001});cash=pd.Series(.00004,index=r.index)
 w,a,rc=f.optimize_weights(r,cash,pd.Series({'growth':.30,'hedge':-.01}),pd.Series(0.,index=r.columns),f.FlatModel())
 assert w.hedge>.20
 assert a['forecast_excess']>=.16-1e-6


def panel():
 rng=np.random.default_rng(11);r=rng.normal([.0005,.0002,.0004],[.008,.002,.006],(545,3))
 return pd.DataFrame(100*np.cumprod(1+r,axis=0),columns=['a','b','c'],index=pd.bdate_range('2020-01-01',periods=len(r)))


def test_class_labels_do_not_affect_allocation():
 p=panel().iloc[:505];cash=pd.Series(.00005,index=p.index);current=pd.Series(0.,index=p.columns)
 assets={c:{'class':'equity'} for c in p};other={c:{'class':'bond','bucket':'anything'} for c in p}
 w,a,rc,d=f.allocate(p.tail(253),cash.tail(253),assets,current,f.FlatModel())
 w2,a2,rc2,d2=f.allocate(p.tail(253),cash.tail(253),other,current,f.FlatModel())
 pd.testing.assert_series_equal(w,w2)
 assert len(d)==3


def test_causal_weekly_execution_and_accounting():
 p=panel();cash=pd.Series(.00005,index=p.index);assets={c:{} for c in p};s=f.FlatModel()
 curve,h,sig,rc,d=f.backtest(p,cash,assets,s)
 p2=p.copy();p2.iloc[530:]*=np.linspace(1,1.5,len(p2)-530)[:,None]
 c2,h2,s2,_,_=f.backtest(p2,cash,assets,s)
 pd.testing.assert_frame_equal(curve.loc[:p.index[529]],c2.loc[:p.index[529]])
 pd.testing.assert_frame_equal(sig.loc[:p.index[529]],s2.loc[:p.index[529]])
 periods=p.index.to_period('W-FRI');allowed={p.index[i] for i in range(504,len(p)) if i==504 or periods[i]!=periods[i-1]}
 assert set(sig.index)<=allowed
 rr=p.pct_change(fill_method=None).fillna(0)
 prev=pd.Series(0.,index=p.columns)
 for date in curve.index:
  expected=(prev*rr.loc[date]).sum()+(1-prev.sum())*cash.loc[date]
  assert curve.loc[date,'return']==pytest.approx((1+expected)*(1-curve.loc[date,'cost_fraction'])-1,abs=1e-12)
  assert h.loc[date].sum()==pytest.approx(1.)
  source=curve.loc[date,'executed_signal_date']
  if pd.notna(source):
   assert p.index.get_loc(date)==p.index.get_loc(source)+1
   np.testing.assert_allclose(h.loc[date,p.columns],sig.loc[source,p.columns],atol=1e-10)
  prev=h.loc[date,p.columns]
