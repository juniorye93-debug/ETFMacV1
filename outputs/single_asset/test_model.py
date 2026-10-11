import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
from model import Config, barrier, features, make_events, fit_model, predict, size, read_data

def bars(rows):
    return pd.DataFrame(rows,columns=['open','high','low','close','volume'],index=pd.date_range('2000-01-01',periods=len(rows)))

def test_ambiguous_bar_stop_first():
    d=bars([[100,101,99,100,1],[100,104,97,101,1],[101,102,100,101,1]])
    e=barrier(d,0,1,1,Config(horizon=2))
    assert e['exit']==98 and e['reason']=='both_stop_first' and e['win']==0

def test_stop_gap_fills_open():
    d=bars([[100,101,99,100,1],[100,101,99,100,1],[90,92,88,91,1]])
    e=barrier(d,0,1,1,Config(horizon=2))
    assert e['exit']==90 and e['reason']=='stop_gap'

def test_short_barriers_and_cost():
    d=bars([[100,101,99,100,1],[100,101,96,97,1],[97,98,96,97,1]])
    e=barrier(d,0,-1,1,Config(horizon=2,short=True))
    assert e['take']==97 and e['stop']==102 and 0<e['net']<.03

def test_t_plus_one_and_gap_cancellation():
    d=bars([[100,101,99,100,1],[100,104,97,101,1],[101,102,100,101,1]])
    assert barrier(d,0,1,1,Config(horizon=2,min_hold=1))['reason']=='time'
    d.iloc[1,0]=102
    assert barrier(d,0,1,1,Config(horizon=2)) is None

def test_partial_position_not_counted_closed():
    d=bars([[100,101,99,100,1],[100,101,99,100,1]])
    assert barrier(d,0,1,1,Config()) is None
    assert barrier(d,0,1,1,Config(),partial=True)['censored']

def test_position_risk_and_margin():
    c=Config(multiplier=300,lot=1,margin_rate=.12)
    n,w=size(4000,50,.2,1e6,c)
    assert n==0 and w==0
    n,w=size(100,2,.2,1e6,Config(lot=100))
    assert n%100==0 and w*4/100<=.005+1e-10

def sample(n=4000):
    rng=np.random.default_rng(20261005);close=100*np.exp(np.cumsum(rng.normal(.0003,.012,n)))
    o=np.r_[close[0],close[:-1]]
    return pd.DataFrame({'open':o,'high':np.maximum(o,close)*1.005,'low':np.minimum(o,close)*.995,'close':close,'volume':rng.uniform(100,200,n)},index=pd.bdate_range('2000-01-01',periods=n))

@pytest.mark.parametrize("estimator",["logistic","random_forest"])
def test_future_changes_cannot_change_features_or_probability(estimator):
    d=sample();c=Config(min_calibration=15,estimator=estimator);at=3300
    x,a,s=features(d,c);events=make_events(d,x,a,s,c);m=fit_model(x,events,at,c)
    changed=d.copy();changed.iloc[at+1:,:4]*=1.5
    xx,aa,ss=features(changed,c);ee=make_events(changed,xx,aa,ss,c);mm=fit_model(xx,ee,at,c)
    pd.testing.assert_frame_equal(x.iloc[:at+1],xx.iloc[:at+1])
    assert m is not None and mm is not None
    assert m['latest_label_end']<at-c.embargo
    assert abs(predict(m,x.iloc[at])-predict(mm,xx.iloc[at]))<1e-12

def test_invalid_ohlc_rejected(tmp_path):
    d=bars([[100,99,101,100,1]]);d.index.name='date';p=tmp_path/'bad.csv';d.to_csv(p)
    import pytest
    with pytest.raises(ValueError): read_data(p)

def test_end_to_end_accounting_and_single_position(tmp_path):
    from model import run
    d=sample(2200);c=Config(min_calibration=15,calibration_bars=504,retrain=63,min_probability=.50)
    metrics,latest=run(d,c,tmp_path,'SYNTHETIC_SOFTWARE_TEST')
    trades=pd.read_csv(tmp_path/'trades.csv');curve=pd.read_csv(tmp_path/'equity.csv')
    assert len(trades)>0
    assert (trades.start.iloc[1:].to_numpy()>trades.end.iloc[:-1].to_numpy()).all()
    assert np.allclose(trades.pnl,trades.net*trades.entry*trades.units*c.multiplier)
    assert curve.date.is_unique
    expected=d.loc[pd.Timestamp(curve.date.iloc[0]):pd.Timestamp(curve.date.iloc[-1])].index
    assert list(pd.to_datetime(curve.date))==list(expected)
    assert latest['data_asof']==str(d.index[-1].date())
    if pd.isna(pd.read_json(tmp_path/'open_position.json',typ='series').get('entry',np.nan)):
        assert abs(curve.equity.iloc[-1]-c.equity-trades.pnl.sum())<1e-6

def test_wind_ohlcv_adapter_and_error_propagation():
    from fetch import fetch
    from types import SimpleNamespace
    import pytest
    class FakeWind:
        def wsd(self,code,fields,start,end,options):
            assert code=='510300.SH' and fields=='open,high,low,close,volume'
            assert 'PriceAdj=F' in options and 'Fill=Blank' in options
            return SimpleNamespace(ErrorCode=0,Fields=['OPEN','HIGH','LOW','CLOSE','VOLUME'],Times=[pd.Timestamp('2026-09-30')],Data=[[100],[102],[99],[101],[1000]])
    d,opts=fetch(FakeWind(),'510300.SH','2020-01-01','2026-10-05','SSE','F')
    assert d.close.iloc[0]==101 and list(d.columns)==['open','high','low','close','volume']
    class FailedWind:
        def wsd(self,*args): return SimpleNamespace(ErrorCode=-40520007)
    with pytest.raises(RuntimeError,match='-40520007'):
        fetch(FailedWind(),'510300.SH','2020-01-01','2026-10-05','SSE','F')

def test_fallback_only_after_wind_failure_and_records_source(tmp_path):
    from fetch import orchestrate
    from types import SimpleNamespace
    import json
    args=SimpleNamespace(symbol='510300.SH',instrument='etf',provider='auto',out=str(tmp_path))
    calls=[]
    def fake_attempt(a,provider):
        calls.append(provider)
        if provider=='wind': return {'provider':provider,'success':False,'reason':'test connection failure'}
        (tmp_path/'510300.SH.meta.json').write_text(json.dumps({'source':'AKShare Eastmoney','synthetic':False}))
        return {'provider':provider,'success':True,'reason':None}
    meta=orchestrate(args,fake_attempt)
    assert calls==['wind','akshare'] and meta['fallback_used']
    assert meta['attempts'][0]['reason']=='test connection failure'
    calls.clear()
    def successful_wind(a,provider):
        calls.append(provider);(tmp_path/'510300.SH.meta.json').write_text(json.dumps({'source':'Wind WSD','synthetic':False}))
        return {'provider':provider,'success':True,'reason':None}
    assert not orchestrate(args,successful_wind)['fallback_used']
    assert calls==['wind']

def test_fallback_normalization_and_symbol_mapping():
    from fetch import normalize_cn,yahoo_symbol,fallback_providers
    from types import SimpleNamespace
    d=normalize_cn(pd.DataFrame({'日期':['2026-09-30'],'开盘':[100],'最高':[102],'最低':[99],'收盘':[101],'成交量':[500]}))
    assert d.volume.iloc[0]==500 and d.close.iloc[0]==101
    assert yahoo_symbol('510300.SH')=='510300.SS'
    assert yahoo_symbol('SPY.P')=='SPY'
    assert fallback_providers('IF2610.CFE','future')==['akshare']

def test_suspended_entry_and_delayed_time_exit():
    d=bars([[100,101,99,100,10],[100,101,99,100,0],[100,101,99,100,10]])
    assert barrier(d,0,1,1,Config(horizon=2)) is None
    d=bars([[100,101,99,100,10],[100,101,99,100,10],[100,101,99,100,0],[101,102,100,101,10]])
    e=barrier(d,0,1,1,Config(horizon=2))
    assert e['end']==3 and e['reason']=='time_deferred_suspension' and e['exit']==101
    e=barrier(d.iloc[:3],0,1,1,Config(horizon=2),partial=True)
    assert e['censored']
