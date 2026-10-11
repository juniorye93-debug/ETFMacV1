import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
import forecast_adaptive as f
import rotation_adaptive as a


def synthetic_prices(n=280):
    rng=np.random.default_rng(50)
    r=rng.normal([.0003,.00012,.0002],[.011,.001,.007],(n,3))
    return pd.DataFrame(100*np.cumprod(1+r,axis=0),columns=['equity','bond','gold'],index=pd.bdate_range('2020-01-01',periods=n))


def test_future_prices_cannot_change_prior_model_fit_selection_or_forecast():
    p=synthetic_prices();cash=pd.Series(.00004,index=p.index)
    settings=f.ForecastSettings(min_asset_sessions=20,min_train_weeks=8,min_evidence_weeks=8,evidence_weeks=12,train_weeks=26,refit_weeks=2,bootstrap_draws=128)
    first=f.prequential_forecasts(f.build_panel(p,cash,settings),settings)
    cut=p.index[205];changed=p.copy();changed.loc[changed.index>cut]*=np.linspace(1,2,len(changed.loc[changed.index>cut]))[:,None]
    second=f.prequential_forecasts(f.build_panel(changed,cash,settings),settings)
    cols=['date','code','prior','prior_se','mu','rho','selected','signal_strength',*f.MODEL_NAMES,'train_label_end','evidence_label_end']
    pd.testing.assert_frame_equal(first.loc[first.date<=cut,cols].reset_index(drop=True),second.loc[second.date<=cut,cols].reset_index(drop=True))
    for col in ['train_label_end','evidence_label_end']:
        valid=first[col].notna();assert (pd.to_datetime(first.loc[valid,col])<=first.loc[valid,'date']).all()


def test_bad_prediction_or_many_correlated_assets_does_not_earn_confidence():
    dates=pd.date_range('2020-01-01',periods=60,freq='W');rng=np.random.default_rng(8)
    y=rng.normal(size=len(dates));frame=pd.DataFrame({'date':dates,'target':y,'prior':0.,'scale':1.,'tree':-y})
    strength,info=f.evidence_gate(frame,'tree',f.ForecastSettings())
    assert strength==0 and info['r2']<0
    repeated=pd.concat([frame.iloc[:20]]*100,ignore_index=True)
    strength,info=f.evidence_gate(repeated,'tree',f.ForecastSettings())
    assert info['weeks']==20 and strength==0


def test_covariance_preserves_bond_vs_equity_volatility_and_positive_dedup():
    rng=np.random.default_rng(2);x=rng.normal(0,.03,500)
    r=pd.DataFrame({'stock':x,'duplicate':x,'bond':rng.normal(0,.001,500),'hedge':-x})
    cov,corr,_=a.covariance_estimate(r,13)
    assert np.linalg.eigvalsh(cov).min()>0
    assert 20<np.sqrt(cov[0,0]/cov[2,2])<40
    kept,log=a.representatives(corr,list(r),.90)
    assert 1 not in kept and 3 in kept
    assert log.loc[log.code=='duplicate','representative'].iloc[0]=='stock'


def test_account_cushion_gamma_and_compounding_do_not_reset_peak():
    s=a.AdaptiveSettings(.08)
    account=a.Account(.9313,1.)
    assert account.cushion(.08)==pytest.approx((.08-.0687)/.9313)
    assert a.dynamic_gamma(account,s)/s.gamma0==pytest.approx(.08/(.08-.0687))
    paths=np.zeros((8,66,1));plan=np.zeros((4,1))
    dd=a.scenario_drawdowns(plan,paths,np.zeros(1),0.,account,.00005,s)
    assert (dd>=account.drawdown-1e-12).all()


def test_uncertainty_and_actual_drawdown_reduce_positions_without_asset_caps():
    s=replace(a.AdaptiveSettings(.08),scenario_paths=64,scenario_weeks=4)
    paths=np.zeros((64,21,1));cov=np.array([[.000025]])
    normal,info=a.optimize_plan(np.array([.003]),np.array([0.]),cov,np.zeros(1),a.Account(1.,1.),paths,.00005,s)
    assert normal[0,0]>.95
    doubtful,_=a.optimize_plan(np.array([.003]),np.array([.004]),cov,np.zeros(1),a.Account(1.,1.),paths,.00005,s)
    assert doubtful[0,0]<1e-4
    loss,_=a.optimize_plan(np.array([.003]),np.array([0.]),cov,np.zeros(1),a.Account(.925,1.),paths,.00005,s)
    assert loss[0,0]<normal[0,0]/2


def test_removed_old_asset_is_still_exposed_during_execution_latency():
    s=replace(a.AdaptiveSettings(.08),scenario_paths=64,scenario_weeks=4)
    paths=np.zeros((64,21,1));growth=np.full(64,.90)
    plan,info=a.optimize_plan(np.array([.003]),np.array([0.]),np.array([[.00001]]),np.zeros(1),a.Account(1.,1.),paths,0.,s,1.,growth,np.ones(64))
    assert plan.sum()==0 and info['status']=='latency_risk_infeasible_cash'


def forecasts_for(prices):
    rows=[]
    for pos in f.signal_positions(prices.index):
        for code in prices:
            rows.append({'date':prices.index[pos],'code':code,'mu':.001,'rho':.0002,'signal_strength':0.,'selected':'prior'})
    return pd.DataFrame(rows)


def test_weekly_backtest_is_causal_uses_t_plus_one_and_exact_fee_accounting():
    p=synthetic_prices(140);cash=pd.Series(.00004,index=p.index);forecast=forecasts_for(p)
    s=replace(a.AdaptiveSettings(.08),scenario_paths=64,scenario_weeks=4)
    c,h,sig,*_=a.backtest(p,cash,{k:{} for k in p},forecast,s,start=70)
    altered=p.copy();altered.iloc[120:]*=np.linspace(1,1.3,len(altered)-120)[:,None]
    c2,h2,sig2,*_=a.backtest(altered,cash,{k:{'class':'different'} for k in p},forecast,s,start=70)
    pd.testing.assert_frame_equal(c.loc[:p.index[119]],c2.loc[:p.index[119]])
    pd.testing.assert_frame_equal(sig.loc[:p.index[119]],sig2.loc[:p.index[119]])
    allowed=set(p.index[f.signal_positions(p.index)]);assert set(sig.index)<=allowed
    prev=h[list(p)].shift(1).fillna(0);rr=p.pct_change(fill_method=None).reindex(c.index)
    expected=(prev*rr).sum(axis=1)+(1-prev.sum(axis=1))*cash.reindex(c.index)
    np.testing.assert_allclose((1+expected)*(1-c.cost_fraction)-1,c['return'],atol=1e-12)
    for date,row in c.iterrows():
        if pd.notna(row.executed_signal_date):
            source=pd.Timestamp(row.executed_signal_date);assert p.index.get_loc(date)==p.index.get_loc(source)+1
            np.testing.assert_allclose(h.loc[date,list(p)],sig.loc[source,list(p)],atol=1e-10)
