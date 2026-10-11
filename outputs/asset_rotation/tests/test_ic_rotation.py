"""Synthetic examples are software tests, never investment performance data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pytest
from ic_rotation import FEATURES,ICSettings,walk_forward,cross_rule,build_panel

def test_future_labels_and_features_cannot_change_past_scores():
    rng=np.random.default_rng(839);dates=pd.date_range('2018-01-01',periods=75,freq='W-MON');rows=[]
    for j,date in enumerate(dates):
        for c in range(8):
            x=rng.normal(size=len(FEATURES));target=.01*x[0]+rng.normal(0,.02)
            rows.append({'date':date,'code':str(c),'scope':'equity','horizon_weeks':4,
                         'fill_date':date+pd.Timedelta(days=1),'label_end':date+pd.Timedelta(weeks=4,days=1),
                         'target':target,'target_rank':target*10,'raw_12m':x[0],
                         **dict(zip(FEATURES,x))})
    p=pd.DataFrame(rows);s=ICSettings(min_train_weeks=16,train_weeks=40,evidence_weeks=40,min_gate_weeks=12,refit_weeks=8)
    a,_,audit=walk_forward(p,4,s);cut=dates[51];changed=p.copy()
    changed.loc[changed.date>cut,FEATURES]=999.
    changed.loc[changed.label_end>cut,['target','target_rank']]*=-100
    b,_,_=walk_forward(changed,4,s)
    columns=['ic_shrink','ridge_rank','ic_gate','ridge_gate','ic_gate_active','ridge_gate_active']
    pd.testing.assert_frame_equal(a.loc[a.date<=cut,columns],b.loc[b.date<=cut,columns])
    for key in ['train_label_end','feature_evidence_label_end','ridge_train_label_end','ic_gate_evidence_label_end','ridge_gate_evidence_label_end']:
        v=audit.dropna(subset=[key]);assert (v[key]<=v.date).all()

def test_cross_section_selects_multiple_etfs_and_respects_dedup():
    codes=list('abcdefgh');date=pd.Timestamp('2020-01-06')
    panel=pd.DataFrame({'date':date,'code':codes,'ic_gate':np.arange(8,0,-1)})
    d={'prices':pd.DataFrame([np.ones(8)],index=[date],columns=codes),
       'assets':{c:{'class':'equity'} for c in codes},
       'packages':{0:{'codes':codes,'representatives':codes,'cov':np.eye(8)*.001}}}
    current=pd.Series(0.,index=codes)
    target=cross_rule(panel,'ic_gate','all')(d,0,current,{})
    assert list(target[target>1e-7].index)==list('abc');assert target.sum()==pytest.approx(1)
    d['packages'][0]['representatives']=list('cdefgh')
    target=cross_rule(panel,'ic_gate','all')(d,0,current,{})
    assert target['a']==0 and target['b']==0
    assert list(target[target>1e-7].index)==list('cd')

def test_no_ic_evidence_uses_explicit_same_pool_covariance_baseline():
    codes=list('abcdef');date=pd.Timestamp('2020-01-06')
    panel=pd.DataFrame({'date':date,'code':codes,'ic_gate':0.})
    d={'prices':pd.DataFrame([np.ones(6)],index=[date],columns=codes),
       'assets':{c:{'class':'equity'} for c in codes},
       'packages':{0:{'codes':codes,'representatives':codes,'cov':np.eye(6)*.001}}}
    target=cross_rule(panel,'ic_gate','all')(d,0,pd.Series(0.,index=codes),{})
    assert np.allclose(target,1/6)

def test_absolute_cash_gate_is_separate_from_centered_rank_score():
    codes=list('abcdef');date=pd.Timestamp('2020-01-06')
    panel=pd.DataFrame({'date':date,'code':codes,'skip_12_1':[6,5,4,3,2,1]})
    d={'prices':pd.DataFrame([np.ones(6)],index=[date],columns=codes),
       'assets':{c:{'class':'equity'} for c in codes},
       'factors':{'cash_relative_13612U':pd.DataFrame([[-.1,.1,.1,.1,.1,.1]],index=[date],columns=codes)},
       'packages':{0:{'codes':codes,'representatives':codes,'cov':np.eye(6)*.001}}}
    current=pd.Series(0.,index=codes)
    target=cross_rule(panel,'skip_12_1','all',absolute=True)(d,0,current,{})
    assert target['a']==0 and target['b']==pytest.approx(1)

def test_ic_forward_label_starts_after_execution_close_and_includes_cash():
    ix=pd.bdate_range('2020-01-06',periods=25);codes=list('abcdef')
    p=pd.DataFrame({c:100*np.exp(np.arange(25)*(.001+.0001*j)) for j,c in enumerate(codes)},index=ix)
    factor={f:p.copy() for f in FEATURES if f!='short_reversal_1m'};factor['mom_12m']=p.copy();factor['mom_1m']=p.copy()
    d={'prices':p,'positions':np.array([0,5,10,15,20]),'cash':pd.Series(.0002,index=ix),'factors':factor,
       'assets':{c:{'class':'equity'} for c in codes},'packages':{j:{'representatives':codes} for j in [0,5,10,15,20]}}
    panel=build_panel(d,1,'equity');row=panel.iloc[0]
    assert row.fill_date==ix[1] and row.label_end==ix[6]
    assert row.target==pytest.approx(p.iloc[6,0]/p.iloc[1,0]/1.0002**5-1)
