import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import equity_signals as e

def prices(n=320):
    rng=np.random.default_rng(19)
    return pd.DataFrame(100*np.cumprod(1+rng.normal(.0004,.01,(n,4)),axis=0),
                        index=pd.bdate_range('2020-01-01',periods=n),columns=list('abcd'))

def test_no_future_price_or_cash_can_change_past_signal():
    p=prices();cash=pd.Series(.0001,index=p.index);cut=p.index[290]
    a=e.price_signals(p,cash);p.loc[p.index>cut]*=2;cash.loc[cash.index>cut]=.01
    b=e.price_signals(p,cash)
    for k in a:pd.testing.assert_frame_equal(a[k].loc[:cut],b[k].loc[:cut])

def test_skip_month_uses_ratio_and_cash_compounds():
    p=prices();cash=pd.Series(.0001,index=p.index);f=e.price_signals(p,cash);i=280
    assert f['skip_12_1'].iloc[i,0]==pytest.approx(p.iloc[i-21,0]/p.iloc[i-252,0]-1)
    assert f['cash_relative_12m'].iloc[i,0]==pytest.approx((p.iloc[i,0]/p.iloc[i-252,0])/(1.0001**252)-1)
    assert f['mom_13612U'].iloc[i,0]==pytest.approx(np.mean([p.iloc[i,0]/p.iloc[i-n,0]-1 for n in [21,63,126,252]]))

def test_new_etfs_and_missing_price_do_not_receive_invented_momentum():
    p=prices();p.loc[p.index[:100],'a']=np.nan;p.loc[p.index[200],'b']=np.nan
    f=e.price_signals(p,pd.Series(0.,index=p.index))
    assert f['mom_12m']['a'].isna().all()
    assert np.isnan(f['mom_3m'].loc[p.index[210],'b'])
    assert np.isnan(f['mom_13612U'].loc[p.index[280],'b'])

def test_nonpositive_pe_and_unverified_valuation_blocked():
    p=prices();p.iloc[250,0]=-5;s=e.SignalSettings(valuation_sessions=100,valuation_min_sessions=30)
    with pytest.raises(ValueError,match='PIT'):e.valuation_signals(p,p,s)
    f=e.valuation_signals(p,p,s,allow_unverified_diagnostic=True)
    assert np.isnan(f['pe_cheapness'].iloc[251,0])
    changed=p.copy();changed.iloc[290:]*=9
    g=e.valuation_signals(changed,changed,s,allow_unverified_diagnostic=True)
    pd.testing.assert_frame_equal(f['pe_cheapness'].iloc[:290],g['pe_cheapness'].iloc[:290])

def test_forward_label_starts_after_execution_close_and_needs_mature_endpoint():
    from signal_diagnostic import label_at
    p=prices();cash=pd.Series(.0001,index=p.index)
    period=p.index.to_period('W-FRI')
    pos=np.flatnonzero(np.r_[True,period[1:]!=period[:-1]])
    y,end=label_at(p,cash,pos,10,13)
    fill=pos[10]+1;last=pos[23]+1
    expected=(p.iloc[last]/p.iloc[fill])/(1.0001**(last-fill))-1
    pd.testing.assert_series_equal(y,expected)
    assert end==p.index[last]
    assert label_at(p,cash,pos,len(pos)-2,13)==(None,None)
