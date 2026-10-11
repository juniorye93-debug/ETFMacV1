import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import rotation as m
from wind_fetch import fetch

ASSETS = {"a": {"class": "equity"}, "b": {"class": "bond"}, "c": {"class": "gold"}}
CAPS = {"equity": .55, "bond": .8, "gold": .3}


def prices(n=310):
    rng = np.random.default_rng(71)
    r = rng.normal(.0005, .008, (n, 3))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), columns=ASSETS,
                        index=pd.bdate_range("2020-01-01", periods=n))


def test_no_future_leakage():
    p = prices()
    s = m.Settings(.1, lookback=126, method="hrp")
    first = m.backtest(p, ASSETS, CAPS, s)
    changed = p.copy()
    changed.iloc[220:] *= np.linspace(1., 4., len(p) - 220)[:, None]
    second = m.backtest(changed, ASSETS, CAPS, s)
    for x, y in zip(first, second):
        pd.testing.assert_frame_equal(x.loc[x.index < p.index[220]], y.loc[y.index < p.index[220]])


def test_execution_is_next_close(monkeypatch):
    p = prices(132)
    p.iloc[:, :] = 100.
    p.iloc[127, 0] = 200.
    p.iloc[128:, 0] = 400.
    monkeypatch.setattr(m, "allocate", lambda *args: (pd.Series([1., 0., 0.], index=p.columns), {"status": "test"}))
    curve, holdings, signals = m.backtest(p, ASSETS, CAPS, m.Settings(.5, lookback=126, fee_bps=0, slippage_bps=0))
    assert curve.iloc[0]["return"] == 0
    assert curve.iloc[1]["return"] == 0  # fill occurs after first jump
    assert curve.iloc[2]["return"] == pytest.approx(1.)
    assert curve.iloc[1].executed_signal_date == signals.index[0]


def test_fee_self_financing():
    nav, turnover, fraction = m.transact(1, [0, 0], [.6, .4], .001)
    assert nav == pytest.approx(1 / 1.001)
    assert fraction == pytest.approx(.001 * turnover)
    nav2, _, fee2 = m.transact(1, [.6, .4], [0, 0], .001)
    assert nav2 == pytest.approx(.999)
    assert fee2 == pytest.approx(.001)


def test_holdings_drift(monkeypatch):
    p = prices(133)
    p.iloc[:, :] = 100
    p.iloc[128:, 0] = 110
    monkeypatch.setattr(m, "allocate", lambda *args: (pd.Series([.5, 0, 0], index=p.columns), {"status": "test"}))
    curve, holdings, _ = m.backtest(p, ASSETS, CAPS, m.Settings(.5, lookback=126, fee_bps=0, slippage_bps=0))
    assert holdings.iloc[2]["a"] == pytest.approx(.55 / 1.05)
    assert curve.iloc[2]["return"] == pytest.approx(.05)


def test_constraints():
    p = prices()
    s = m.Settings(.08, lookback=126)
    w, audit = m.allocate(p, pd.Series(0., index=p.columns), ASSETS, CAPS, s)
    assert w.sum() <= 1 + 1e-9
    assert w.min() >= 0
    assert w.max() <= s.max_weight + 1e-9
    assert audit["status"] in {"optimal", "optimal_inaccurate"}
    assert audit["cdar"] <= .08 * .6 + 1e-8
    assert audit["mdd"] <= .08 * .8 + 1e-8
    assert audit["vol"] <= .08 * .5 + 1e-8
    for cls, cap in CAPS.items():
        assert w[[a for a in ASSETS if ASSETS[a]["class"] == cls]].sum() <= cap + 1e-9


def test_hard_stop_and_gap_breach(monkeypatch):
    p = prices(140)
    p.iloc[:, :] = 100
    p.iloc[129:, 0] = 60
    monkeypatch.setattr(m, "allocate", lambda *args: (pd.Series([1., 0, 0], index=p.columns), {"status": "test"}))
    curve, holdings, signals = m.backtest(p, ASSETS, CAPS, m.Settings(.1, lookback=126, fee_bps=0, slippage_bps=0))
    assert curve.loc[p.index[129], "halted"]
    assert holdings.loc[p.index[130]:, "cash"].eq(1.).all()
    assert m.metrics(curve, m.Settings(.1))["target_breached"]


def test_missing_prices_rejected(tmp_path):
    p = prices()
    p.iloc[40, 0] = np.nan
    path = tmp_path / "p.csv"
    p.to_csv(path)
    with pytest.raises(ValueError, match="上市后"):
        m.read_prices(path, ASSETS)


def test_pre_listing_not_backfilled():
    p = prices()
    p.iloc[:250, 0] = np.nan
    w, _ = m.allocate(p, pd.Series(0., index=p.columns), ASSETS, CAPS, m.Settings(.1, lookback=126))
    assert w["a"] == 0


def test_initial_loss_in_drawdown():
    curve = pd.DataFrame({"return": [-.1, 0], "turnover": [0, 0], "exposure": [1, 1], "halted": [False, False]})
    assert m.metrics(curve, m.Settings(.2))["max_drawdown"] == pytest.approx(.1)


def test_wind_errors_and_layout():
    class Result:
        ErrorCode = 0
        Times = ["2020-01-01", "2020-01-02"]
        Data = [[100., 101.]]
    class Wind:
        def wsd(self, *args):
            return Result()
    frame, options = fetch(Wind(), ["a", "b"], "2020-01-01", "2020-01-02", "US")
    assert frame.shape == (2, 2)
    assert "NYSE" in options and "Fill=Blank" in options
    Result.ErrorCode = -1
    with pytest.raises(RuntimeError, match="WSD失败"):
        fetch(Wind(), ["a"], "2020-01-01", "2020-01-02", "US")


def test_all_downtrend_cash():
    p = prices()
    p.iloc[:, :] = np.linspace(100, 30, len(p))[:, None]
    w, audit = m.allocate(p, pd.Series(0., index=p.columns), ASSETS, CAPS, m.Settings(.1, lookback=126))
    assert w.sum() == 0
    assert audit["status"] == "no_positive_trend"


def test_solver_failure_cash(monkeypatch):
    def fail(*args, **kwargs):
        raise m.cp.error.SolverError("test")
    monkeypatch.setattr(m.cp.Problem, "solve", fail)
    p = prices()
    w, audit = m.allocate(p, pd.Series(0., index=p.columns), ASSETS, CAPS, m.Settings(.1, lookback=126))
    assert w.sum() == 0
    assert audit["status"] == "solver_error_cash"


def test_baseline_matches_engine_for_constant_allocation(monkeypatch):
    from benchmarks import run_baseline
    p = prices(170)
    s = m.Settings(.9, lookback=126, fee_bps=3, slippage_bps=5)
    target = pd.Series([.2,.2,.2],index=p.columns)
    monkeypatch.setattr(m, 'allocate', lambda *args: (target.copy(), {'status':'test'}))
    curve, _, _ = m.backtest(p, ASSETS, CAPS, s)
    reference = run_baseline(p,target,s)
    np.testing.assert_allclose(curve['nav'],reference['nav'],rtol=1e-13)
    np.testing.assert_allclose(curve['turnover'],reference['turnover'],rtol=1e-13)


def test_cash_accrues_weekends_and_lags_yield():
    from cash_rates import cash_accrual
    idx=pd.to_datetime(['2026-09-18','2026-09-21','2026-09-22'])
    annual=pd.Series([.0365,.073], index=pd.to_datetime(['2026-09-17','2026-09-21']))
    r,_=cash_accrual(idx,annual)
    assert r.iloc[1]==pytest.approx(1.0001**3-1)
    assert r.iloc[2]==pytest.approx(.0002)
    changed=annual.copy(); changed.iloc[-1]=.146
    other,_=cash_accrual(idx,changed)
    assert other.iloc[1]==pytest.approx(r.iloc[1])


def test_cash_missing_history_fails():
    from cash_rates import cash_accrual
    idx=pd.bdate_range('2026-09-21',periods=3)
    y=pd.Series([.02], index=pd.to_datetime(['2026-09-22']))
    with pytest.raises(ValueError,match='利率历史不足'):
        cash_accrual(idx,y)


def test_cash_only_engine_earns_yield(monkeypatch):
    p=prices(140)
    zero=pd.Series(0.,index=p.columns)
    monkeypatch.setattr(m,'allocate',lambda *args:(zero.copy(),{'status':'test'}))
    cash=pd.Series(.0001,index=p.index)
    curve,_,_=m.backtest(p,ASSETS,CAPS,m.Settings(.1,lookback=126),cash_returns=cash)
    assert curve.iloc[-1].nav==pytest.approx(1.0001**len(curve))
    assert m.metrics(curve,m.Settings(.1))['sharpe_cash_benchmark'] is None


def test_bounded_history_equals_full_allocation():
    p=prices(400)
    s=m.Settings(.1,lookback=126)
    current=pd.Series(0.,index=p.columns)
    full,_=m.allocate(p,current,ASSETS,CAPS,s)
    short,_=m.allocate(p.tail(127),current,ASSETS,CAPS,s)
    np.testing.assert_allclose(full,short,atol=1e-10)


def test_hrp_reacts_to_correlation():
    rng=np.random.default_rng(127)
    x=rng.normal(size=(253,3))
    def panel(z):
        z=(z-z.mean(axis=0))/z.std(axis=0)*.004+.001
        return pd.DataFrame(100*np.cumprod(1+z,axis=0),columns=ASSETS,index=pd.bdate_range('2020-01-01',periods=253))
    a=panel(x.copy())
    y=x.copy();y[:,1]=.95*x[:,0]+.05*x[:,1]
    b=panel(y)
    s=m.Settings(.5,lookback=252,method='hrp',max_weight=1)
    assets={k:{'class':'equity'} for k in ASSETS}
    current=pd.Series(0.,index=a.columns)
    wa,_=m.allocate(a,current,assets,{'equity':1},s)
    wb,_=m.allocate(b,current,assets,{'equity':1},s)
    assert np.abs(wa-wb).max()>.02


def test_audit_overlay_disabled_can_reinvest_after_loss(monkeypatch):
    p = prices(145)
    p.iloc[:, :] = 100.
    p.iloc[129:, 0] = 85.
    monkeypatch.setattr(m, 'allocate', lambda *args: (pd.Series([1., 0., 0.], index=p.columns), {'status':'test'}))
    settings=m.Settings(.08,lookback=126,fee_bps=0,slippage_bps=0,overlay_mode='none')
    c,h,_=m.backtest(p,ASSETS,CAPS,settings)
    assert not c.halted.any()
    assert h.iloc[-1]['a']==pytest.approx(1.)
    assert m.metrics(c,settings)['target_breached']  # No hidden reset of global high-water mark.


def test_scheduled_only_does_not_multiply_daily_cuts(monkeypatch):
    p=prices(145)
    p.iloc[:,:]=100.
    p.iloc[129:,0]=94.
    monkeypatch.setattr(m,'allocate',lambda *args:(pd.Series([1.,0.,0.],index=p.columns),{'status':'test'}))
    settings=m.Settings(.08,lookback=126,fee_bps=0,slippage_bps=0,overlay_mode='scheduled_only')
    _,_,signals=m.backtest(p,ASSETS,CAPS,settings)
    assert 'daily_defensive_cut' not in set(signals.status)
    assert signals.index.to_period('W-FRI').is_unique


def test_balanced_selection_limits_duplicate_bond_slots():
    n=260;t=np.arange(n);idx=pd.bdate_range('2020-01-01',periods=n)
    rr=.0003+.0001*np.sin(t)
    p=pd.DataFrame({c:100*np.cumprod(1+rr) for c in ['bond1','bond2','bond3']},index=idx)
    p['equity']=100*np.cumprod(1+.0005+.002*np.sin(t*.37))
    assets={c:{'class':'bond' if c.startswith('bond') else 'equity'} for c in p}
    _,a=m.allocate(p,p.iloc[-1]*0,assets,{'bond':.8,'equity':.55},m.Settings(.08,method='hrp',selection_mode='class_balanced'))
    selected=a['selected'].split(',')
    assert sum(c.startswith('bond') for c in selected)==1
    assert 'equity' in selected
