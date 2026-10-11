import numpy as np
import pandas as pd
from riskparity.factor_library import daily_factors, intraday_factors, quote_factors, validate_daily


def prices():
    idx = pd.bdate_range('2026-01-01', periods=100)
    c = 100 * np.exp(np.cumsum(np.sin(np.arange(100)) * .01))
    return pd.DataFrame({'open':c,'high':c*1.01,'low':c*.99,'close':c,'volume':1000.,'amt':100000.,'turn':2.},index=idx)


def test_volatility_and_market_residual():
    d=prices(); f=daily_factors(d,d.close)
    expected=np.std(np.diff(np.log(d.close))[-20:],ddof=1)*np.sqrt(252)
    assert np.isclose(f.vol_20.iloc[-1],expected)
    assert np.isclose(f.beta_60.iloc[-1],1)
    assert f.idiosyncratic_vol_60.iloc[-1] < 1e-7
    assert np.isclose(f.turnover_20.iloc[-1],.02)


def test_no_future_leakage_and_zero_volume():
    d=prices(); before=daily_factors(d)
    changed=d.copy(); changed.loc[changed.index[-1],'close'] *= 2
    pd.testing.assert_frame_equal(before.iloc[:-1],daily_factors(changed).iloc[:-1])
    d.loc[d.index[-1],['volume','amt']]=0
    assert np.isnan(daily_factors(d).amihud_20.iloc[-1])
    assert daily_factors(d).zero_volume_fraction_20.iloc[-1] == .05


def test_minute_aggregation_excludes_gaps_and_masks_partial_day():
    idx=pd.date_range('2026-09-28 09:35',periods=24,freq='5min').append(pd.date_range('2026-09-28 13:05',periods=24,freq='5min'))
    bars=pd.DataFrame({'open':100.,'close':101.,'amt':1.},index=idx)
    f=intraday_factors(bars)
    assert np.isclose(f.realized_variance.iloc[0],48*np.log(1.01)**2)
    assert np.isclose(f.tail_amount_fraction.iloc[0],6/48)
    assert np.isclose(f.amount_concentration.iloc[0],1/48)
    assert np.isnan(intraday_factors(bars.iloc[:-1]).realized_variance.iloc[0])


def test_quotes_and_invalid_ohlc():
    q=pd.DataFrame({'bid':[99.],'ask':[101.],'bid_depth':[300.],'ask_depth':[100.]},index=pd.to_datetime(['2026-09-28 10:00']))
    f=quote_factors(q)
    assert f.relative_spread.iloc[0] == .02
    assert f.book_imbalance.iloc[0] == .5
    d=prices(); d.loc[d.index[-1],'high']=1
    import pytest
    with pytest.raises(ValueError): validate_daily(d)


def test_wind_minute_amount_alias():
    from types import SimpleNamespace
    from riskparity.factor_library import wind_frame
    result=SimpleNamespace(ErrorCode=0,Fields=['close','amount'],Data=[[100.],[1000.]],Times=[pd.Timestamp('2026-09-28 10:00')])
    f=wind_frame(result)
    assert f.amt.iloc[0] == 1000.
    assert 'amount' not in f
