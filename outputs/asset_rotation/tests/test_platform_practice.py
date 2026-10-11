import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pytest
from platform_factors import build_factors,regression_score,FactorSettings
from factor_practice import simulate,PracticeSettings,guard_target,audited_stock_bond_rule,dual_rule
from platform_forecast import FEATURES,build_panel

def test_all_expanded_factors_are_prefix_invariant():
    rng=np.random.default_rng(718);ix=pd.bdate_range('2019-01-01',periods=680)
    p=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0002,.01,(680,4)),axis=0)),index=ix,columns=list('abcd'));cash=pd.Series(.0001,index=ix)
    o=p.stack().rename('close').reset_index();o.columns=['date','code','close'];o['high']=o.close*1.01;o['low']=o.close*.99;o['volume']=1000.
    a=build_factors(p,cash,'a',o);cut=ix[645];changed=p.copy();changed.loc[changed.index>cut]*=2;cc=cash.copy();cc.loc[cc.index>cut]=.01;oo=o.copy();oo.loc[oo.date>cut,['close','high','low']]*=2
    b=build_factors(changed,cc,'a',oo)
    for k in a:pd.testing.assert_frame_equal(a[k].loc[:cut],b[k].loc[:cut])

def test_weighted_regression_agrees_with_wls_definition():
    x=np.arange(25);y=.003*x+np.sin(x)*.01;w=np.linspace(1,2,25)
    score,beta,r2=regression_score(y,w);b,intercept=np.polyfit(x,y,1,w=w)
    ww=w*w;mean=np.average(y,weights=ww);expected=1-np.sum(ww*(y-(b*x+intercept))**2)/np.sum(ww*(y-mean)**2)
    assert beta==pytest.approx(b);assert r2==pytest.approx(expected);assert score==pytest.approx(b*expected)

def test_next_close_execution_does_not_earn_execution_day_return(tmp_path):
    ix=pd.bdate_range('2020-01-06',periods=4);p=pd.DataFrame({'a':[100,110,121,121]},index=ix)
    d={'market':'CN','prices':p,'returns':p.pct_change(fill_method=None),'cash':pd.Series(0.,index=ix),'dest':tmp_path,'packages':{0:{'codes':['a'],'cov':np.array([[.0001]])}}}
    s=PracticeSettings(start='2020-01-06',cost_rate=0)
    simulate(d,'timing',lambda d,p,c,t:pd.Series(1.,index=c.index),s)
    curve=pd.read_csv(tmp_path/'timing/curve.csv');assert curve.nav.iloc[1]==pytest.approx(1);assert curve.nav.iloc[2]==pytest.approx(1.1)
    assert pd.to_datetime(curve.executed_signal_date.iloc[1])==ix[0]

def test_restore_keeps_original_peak_and_respects_remaining_budget():
    p=pd.DataFrame({'a':[100.]},index=pd.to_datetime(['2020-01-06']));pkg={'codes':['a'],'latency_returns':np.zeros((64,1)),'scenario_growth':np.full((64,2,1),.8),'cash_growth':np.ones(2)}
    d={'prices':p,'cash':pd.Series(0.,index=p.index),'packages':{0:pkg}};s=PracticeSettings();current=pd.Series(0.,index=['a']);target=pd.Series(1.,index=['a']);state={'risk_state':'DERISK','positive_weeks':1}
    out,audit=guard_target(d,0,target,current,.95,1.,state,s,recovery=True)
    assert 0<out.sum()<=.1+1e-10;assert audit['scenario_mdd_q95']<=.08+1e-10;assert state['risk_state']=='PROBE'
    out,audit=guard_target(d,0,target,current,.91,1.,state,s,recovery=True)
    assert out.sum()==0;assert audit['account_already_breached'];assert audit['scenario_mdd_q95']>=.09-1e-10

def test_valuation_and_financial_vintages_are_excluded_from_learners():
    assert not any('cheapness' in k or 'yield' in k or 'consensus' in k or 'roe' in k or 'flow' in k for k in FEATURES)
    ix=pd.bdate_range('2020-01-06',periods=400);p=pd.DataFrame({'a':np.arange(400)+100.},index=ix);cash=pd.Series(.0001,index=ix)
    from forecast_adaptive import signal_positions
    factors={k:p*.001 for k in FEATURES};d={'prices':p,'returns':p.pct_change(fill_method=None),'cash':cash,'positions':signal_positions(ix),'factors':factors}
    panel=build_panel(d,13);row=panel.dropna(subset=['target']).iloc[0];fill=ix.get_loc(row.fill_date);end=ix.get_loc(row.label_end)
    expected=p.iloc[end,0]/p.iloc[fill,0]/np.prod(1+cash.iloc[fill+1:end+1])-1
    assert row.target==pytest.approx(expected);assert row.label_end>row.date

def test_mixed_share_date_formats_align_and_use_prior_day():
    ix=pd.bdate_range('2020-01-06',periods=320);p=pd.DataFrame({'a':100+np.arange(320),'b':150+np.arange(320)},index=ix)
    shares=pd.DataFrame({'date':[str(d.date()) if i%2 else str(d) for i,d in enumerate(ix)],'code':'a','shares':1000+np.arange(320)})
    f=build_factors(p,pd.Series(0.,index=ix),'b',shares=shares)
    assert f['share_flow5'].iloc[30]['a']==pytest.approx(1029/1024-1)
    shares.loc[30,'shares']=999999
    g=build_factors(p,pd.Series(0.,index=ix),'b',shares=shares)
    assert g['share_flow5'].iloc[30]['a']==f['share_flow5'].iloc[30]['a']

def test_audited_bond_hurdle_does_not_buy_losing_defensive_bond():
    codes=['510300.SH','CBA00601.CS'];ix=pd.to_datetime(['2020-01-06'])
    d={'market':'CN','prices':pd.DataFrame([[100,100]],index=ix,columns=codes),
       'factors':{'mom_12m':pd.DataFrame([[-.2,-.1]],index=ix,columns=codes),
                  'cash_relative_12m':pd.DataFrame([[-.22,-.12]],index=ix,columns=codes)},
       'packages':{0:{'codes':codes,'cov':np.diag([.01,.001])}}}
    current=pd.Series(0.,index=codes)
    assert dual_rule(d,0,current,{})['CBA00601.CS']==pytest.approx(1)
    assert audited_stock_bond_rule()(d,0,current,{}).sum()==pytest.approx(0)
    d['factors']['cash_relative_12m'].iloc[0,1]=np.nan
    assert audited_stock_bond_rule()(d,0,current,{}).sum()==pytest.approx(0)
    d['factors']['cash_relative_12m'].iloc[0,1]=.02
    assert audited_stock_bond_rule()(d,0,current,{})['CBA00601.CS']==pytest.approx(1)

def test_equity_absolute_ablation_does_not_compare_equity_to_bond():
    codes=['SPY.P','TLT.O'];ix=pd.to_datetime(['2020-01-06'])
    d={'market':'US','prices':pd.DataFrame([[100,100]],index=ix,columns=codes),
       'factors':{'mom_12m':pd.DataFrame([[.05,.1]],index=ix,columns=codes),
                  'cash_relative_12m':pd.DataFrame([[.02,.07]],index=ix,columns=codes)},
       'packages':{0:{'codes':codes,'cov':np.diag([.01,.001])}}}
    current=pd.Series(0.,index=codes)
    assert audited_stock_bond_rule()(d,0,current,{})['TLT.O']==pytest.approx(1)
    assert audited_stock_bond_rule(compare_bond=False)(d,0,current,{})['SPY.P']==pytest.approx(1)
    d['factors']['cash_relative_12m'].iloc[0,0]=-.02
    assert audited_stock_bond_rule(compare_bond=False,cash_only=True)(d,0,current,{}).sum()==pytest.approx(0)
