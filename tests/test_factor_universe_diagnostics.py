import numpy as np
import pandas as pd
from riskparity.factor_universe_diagnostics import hac_mean,bh_fdr,quintile_returns,partial_rank_ic


def test_fdr_known_adjusted_probabilities():
    q=bh_fdr([.001,.01,.03,.2,np.nan])
    np.testing.assert_allclose(q.iloc[:4],[.004,.02,.04,.2])
    assert np.isnan(q.iloc[4])


def test_hac_recognizes_serial_dependence():
    # Long runs increase uncertainty compared with the independence assumption.
    x=np.tile(np.repeat([-1.,1.],25),4)+.1
    se0,_,_=hac_mean(x,0)
    se20,_,_=hac_mean(x,20)
    assert se20>2*se0


def test_quintiles_use_groups_not_single_extreme_stocks():
    v=pd.DataFrame({'score':np.arange(100),'return':np.arange(100)/100})
    q=quintile_returns(v,'score','return')
    np.testing.assert_allclose(q,[.095,.295,.495,.695,.895])
    v['score']=1
    assert all(np.isnan(x) for x in quintile_returns(v,'score','return'))


def test_control_removes_pure_size_rank_signal():
    v=pd.DataFrame({'score':np.arange(100),'return':np.arange(100),'market_cap':np.exp(np.arange(100)/20), 'industry':['same']*100})
    # All score information is explained by log size, so a residual IC is undefined.
    assert np.isnan(partial_rank_ic(v,'score','return'))
