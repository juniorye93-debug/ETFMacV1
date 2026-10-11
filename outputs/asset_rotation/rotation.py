"""Research-only daily ETF rotation. No brokerage/order interfaces."""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import HRPOpt
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Settings:
    max_drawdown: float
    frequency: str = "weekly"
    method: str = "cdar"
    lookback: int = 252
    top_k: int = 5
    max_weight: float = .35
    cdar_fraction: float = .6
    historical_mdd_fraction: float = .8
    vol_fraction: float = .5
    confidence: float = .95
    soft_stop_fraction: float = .5
    hard_stop_fraction: float = .9
    fee_bps: float = 3
    slippage_bps: float = 5
    cash_rate: float = 0
    risk_free_rate: float = 0
    reference_penalty: float = .05
    turnover_penalty: float = .002
    overlay_mode: str = "legacy"
    selection_mode: str = "legacy"

    def __post_init__(self):
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown 必须是 (0,1) 内的小数，例如 0.10")
        if self.frequency not in {"daily", "weekly"} or self.method not in {"cdar", "hrp", "inverse_vol"}:
            raise ValueError("频率或算法无效")
        if self.lookback < 126 or self.top_k < 1 or not 0 < self.max_weight <= 1:
            raise ValueError("lookback>=126, top_k>=1, 0<max_weight<=1")
        if not 0 < self.soft_stop_fraction < self.hard_stop_fraction < 1:
            raise ValueError("回撤软阈值 < 硬阈值 < 1")
        if not 0 < self.confidence < 1 or not 0 < self.vol_fraction <= 2:
            raise ValueError("风险参数无效")
        if not 0 < self.cdar_fraction <= self.historical_mdd_fraction < 1:
            raise ValueError("0 < CDaR比例 <= 历史MDD比例 < 1")
        if min(self.fee_bps, self.slippage_bps, self.reference_penalty, self.turnover_penalty) < 0:
            raise ValueError("成本和惩罚项不能为负")
        if (self.fee_bps + self.slippage_bps) / 10000 >= .1:
            raise ValueError("交易成本异常")
        if min(self.cash_rate, self.risk_free_rate) <= -1:
            raise ValueError("年化利率必须大于-1")
        if self.overlay_mode not in {"legacy", "scheduled_only", "none"}:
            raise ValueError("overlay_mode必须为legacy/scheduled_only/none；后两者仅用于诊断")
        if self.selection_mode not in {"legacy", "class_balanced"}:
            raise ValueError("selection_mode必须为legacy/class_balanced")


def read_prices(path: Path, assets: dict) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("首列必须是有效交易日期")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("交易日必须唯一且升序")
    if set(prices.columns) != set(assets):
        raise ValueError(f"价格列必须与所选市场配置一致: {list(assets)}")
    prices = prices[list(assets)].astype(float)
    for code, asset in assets.items():
        listing = asset.get("listing_date")
        if listing and asset.get("instrument_type") != "index_proxy":
            prices.loc[prices.index < pd.Timestamp(listing), code] = np.nan
    if np.isinf(prices.to_numpy()).any() or (prices <= 0).any().any():
        raise ValueError("价格必须为正有限数，上市前允许空值")
    # Never backfill pre-listing history or silently invent tradable prices.
    for code in prices:
        valid = prices[code].first_valid_index()
        if valid is None or prices.loc[valid:, code].isna().any():
            raise ValueError(f"{code}: 上市后有缺失；先核对停牌/退市和交易日历，禁止自动填充")
    return prices


def cap_weights(w: pd.Series, assets: dict, caps: dict, limit: float):
    w = w.clip(lower=0, upper=limit)
    for cls, cap in caps.items():
        names = [a for a in w.index if assets[a]["class"] == cls]
        total = w[names].sum()
        if total > cap:
            w.loc[names] *= cap / total
    if w.sum() > 1:
        w /= w.sum()
    return w


def tail_mean(x, confidence):
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, confidence)
    return float(q + np.maximum(x - q, 0).mean() / (1 - confidence))


def historical_risk(returns, w, covariance, confidence):
    r = np.asarray(returns) @ np.asarray(w)
    cumulative = np.r_[0., np.cumsum(r)]
    dd = np.maximum.accumulate(cumulative)[1:] - cumulative[1:]
    return {"cdar": tail_mean(dd, confidence), "mdd": float(dd.max()),
            "vol": float(np.sqrt(max(0., w @ covariance @ w) * 252))}


def allocate(history, current, assets, caps, s: Settings):
    all_codes = history.columns
    zero = pd.Series(0., index=all_codes)
    returns = history.pct_change(fill_method=None).iloc[-s.lookback:]
    eligible = [c for c in all_codes if returns[c].notna().all() and returns[c].std() > 1e-8]
    if not eligible:
        return zero, {"status": "no_history"}
    p = history[eligible]
    momentum = sum(p.iloc[-1] / p.iloc[-1 - n] - 1 for n in (21, 63, 126)) / 3
    # Absolute trend and volatility-adjusted relative momentum; known history only.
    mask = (p.iloc[-1] > p.tail(126).mean()) & (momentum > 0)
    score = momentum / (returns[eligible].std() * np.sqrt(252))
    ranked = score[mask].sort_values(ascending=False, kind="stable").index.tolist()
    selected = ranked[:s.top_k]
    if s.selection_mode == "class_balanced":
        # Diagnostic ablation: at most two equity funds, one per other class.
        # Positively correlated duplicates are excluded using past returns only.
        selected = []
        counts = {}
        correlation = returns[eligible].corr()
        for code in ranked:
            cls = assets[code]["class"]
            if counts.get(cls, 0) >= (2 if cls == "equity" else 1):
                continue
            if any(correlation.loc[code, other] > .85 for other in selected):
                continue
            selected.append(code)
            counts[cls] = counts.get(cls, 0) + 1
            if len(selected) >= s.top_k:
                break
    if not selected:
        return zero, {"status": "no_positive_trend"}
    r = returns[selected]
    cov = LedoitWolf().fit(r.to_numpy()).covariance_
    cov += np.eye(len(selected)) * 1e-12
    cov_df = pd.DataFrame(cov, index=selected, columns=selected)
    if s.method == "inverse_vol":
        ref = 1 / np.sqrt(pd.Series(np.diag(cov), index=selected))
        ref /= ref.sum()
    elif len(selected) == 1:
        ref = pd.Series(1., index=selected)
    else:
        ref = pd.Series(HRPOpt(cov_matrix=cov_df).optimize(linkage_method="single")).reindex(selected)
    ref = cap_weights(ref, assets, caps, s.max_weight)
    budget = {"cdar": s.max_drawdown * s.cdar_fraction,
              "mdd": s.max_drawdown * s.historical_mdd_fraction,
              "vol": min(.15, s.max_drawdown * s.vol_fraction)}
    w_out = ref.to_numpy()
    status = "reference"
    if s.method == "cdar":
        t, n = r.shape
        w = cp.Variable(n, nonneg=True)
        dd = cp.Variable(t + 1, nonneg=True)
        threshold = cp.Variable(nonneg=True)
        excess = cp.Variable(t, nonneg=True)
        daily = r.to_numpy() @ w
        constraints = [cp.sum(w) <= 1, w <= s.max_weight, dd[0] == 0,
                       dd[1:] >= dd[:-1] - daily,
                       dd[1:] <= budget["mdd"], excess >= dd[1:] - threshold,
                       threshold + cp.sum(excess) / ((1 - s.confidence) * t) <= budget["cdar"],
                       cp.quad_form(w, cp.psd_wrap(cov)) <= budget["vol"] ** 2 / 252]
        for cls, cap in caps.items():
            ix = [i for i, a in enumerate(selected) if assets[a]["class"] == cls]
            if ix:
                constraints.append(cp.sum(w[ix]) <= cap)
        # Shrunk trend proxy, not a claim of accurate expected returns.
        mu = (momentum[selected] * 252 / 70).clip(-.2, .2).to_numpy() * .25
        objective = cp.Maximize(mu @ w - s.reference_penalty * cp.norm1(w - ref.to_numpy())
                               - s.turnover_penalty * cp.norm1(w - current[selected].to_numpy()))
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver="CLARABEL")
            status = str(problem.status)
            if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
                return zero, {"status": "solver_failed_cash:" + status}
            w_out = np.maximum(np.asarray(w.value), 0)
        except cp.error.SolverError:
            return zero, {"status": "solver_error_cash"}
    w_out = cap_weights(pd.Series(w_out, index=selected), assets, caps, s.max_weight).to_numpy()
    # Recheck after clipping, and conservatively scale down numerical residuals.
    # CDaR/MDD are homogeneous for this uncompounded historical approximation.
    risk = historical_risk(r, w_out, cov, s.confidence)
    keys = ("vol", "cdar", "mdd") if s.method == "cdar" else ("vol",)
    scale = min([1.] + [budget[k] / max(risk[k], 1e-12) for k in keys])
    w_out = w_out * scale
    zero.loc[selected] = w_out
    risk = historical_risk(r, w_out, cov, s.confidence)
    return zero, {"status": status, **risk, "selected": ",".join(selected)}


def transact(nav, weights, target, rate):
    """Solve fee = rate * gross traded notional; weights are post-cost weights."""
    before = nav * np.asarray(weights)
    target = np.asarray(target)
    lo, hi = 0., nav * min(2 * rate, .5)
    for _ in range(55):
        fee = (lo + hi) / 2
        required = rate * np.abs((nav - fee) * target - before).sum()
        if fee < required:
            lo = fee
        else:
            hi = fee
    fee = (lo + hi) / 2
    notional = np.abs((nav - fee) * target - before).sum()
    return nav - fee, notional / nav, fee / nav


def backtest(prices, assets, caps, s: Settings, cash_returns=None):
    if len(prices) <= s.lookback + 2:
        raise ValueError("交易日不足，至少需要lookback+3行")
    returns = prices.pct_change(fill_method=None)
    weights = pd.Series(0., index=prices.columns)
    pending = None
    pending_date = None
    nav = peak = 1.
    halted = False
    records, positions, signals = [], [], []
    cash_daily = (1 + s.cash_rate) ** (1 / 252) - 1
    if cash_returns is not None:
        cash_returns = cash_returns.reindex(prices.index)
        if cash_returns.isna().any() or not np.isfinite(cash_returns.to_numpy()).all() or (cash_returns <= -1).any():
            raise ValueError("现金收益缺失或无效")
    periods = prices.index.to_period("W-FRI")
    for i in range(s.lookback, len(prices)):
        date = prices.index[i]
        prior_nav = nav
        period_cash_return = cash_daily if cash_returns is None else float(cash_returns.iloc[i])
        r = returns.iloc[i]
        if r[weights > 1e-12].isna().any():
            raise ValueError(f"持仓标的收益缺失: {date}")
        r = r.fillna(0.)
        asset_values = weights * (1 + r)
        growth = float(asset_values.sum() + (1 - weights.sum()) * (1 + period_cash_return))
        if growth <= 0:
            raise ValueError("组合净值耗尽")
        nav *= growth
        weights = asset_values / growth
        turnover = fee = 0.
        executed_signal = None
        if pending is not None:
            nav, turnover, fee = transact(nav, weights, pending, (s.fee_bps + s.slippage_bps) / 10000)
            weights = pending.copy()
            executed_signal = pending_date
            pending = None
        peak = max(peak, nav)
        drawdown = 1 - nav / peak
        halted = (halted or drawdown >= s.hard_stop_fraction * s.max_drawdown) if s.overlay_mode != "none" else False
        soft = s.soft_stop_fraction * s.max_drawdown
        hard = s.hard_stop_fraction * s.max_drawdown
        exposure_scale = float(np.clip((hard - drawdown) / (hard - soft), 0, 1))
        if s.overlay_mode == "none":
            exposure_scale = 1.
        # Weekly decision on first observed session of a W-FRI week: no future holiday knowledge.
        scheduled = s.frequency == "daily" or i == s.lookback or periods[i] != periods[i - 1]
        audit = {}
        if halted:
            target = weights * 0
            audit = {"status": "hard_stop_latched"}
        elif scheduled:
            target, audit = allocate(prices.iloc[max(0, i-s.lookback):i + 1], weights, assets, caps, s)
            target *= exposure_scale
        elif drawdown > soft and s.overlay_mode == "legacy":
            # Daily risk overlay can only reduce between weekly allocation dates.
            target = weights * exposure_scale
            audit = {"status": "daily_defensive_cut"}
        else:
            target = None
        if target is not None:
            pending, pending_date = target, date
            signals.append({"date": date, "exposure_scale": exposure_scale, **audit,
                            **target.to_dict(), "cash": 1 - target.sum()})
        records.append({"date": date, "nav": nav, "return": nav / prior_nav - 1,
                        "drawdown": drawdown, "turnover": turnover, "cost_fraction": fee,
                        "exposure": weights.sum(), "halted": halted, "cash_return": period_cash_return,
                        "executed_signal_date": executed_signal})
        positions.append({"date": date, **weights.to_dict(), "cash": 1 - weights.sum()})
    return (pd.DataFrame(records).set_index("date"), pd.DataFrame(positions).set_index("date"),
            pd.DataFrame(signals).set_index("date"))


def metrics(curve, s):
    r = curve["return"].to_numpy()
    if not len(r):
        return {}
    nav = np.r_[1., np.cumprod(1 + r)]
    dd = 1 - nav / np.maximum.accumulate(nav)
    cagr = nav[-1] ** (252 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(252) if len(r) > 1 else 0.
    rf = (1 + s.risk_free_rate) ** (1 / 252) - 1
    mdd = float(dd.max())
    return {"sessions": len(r), "annual_return": float(cagr), "annual_volatility": float(vol),
            "sharpe": float((r.mean() - rf) * 252 / vol) if vol > 1e-12 else None,
            "sharpe_cash_benchmark": float((r - curve["cash_return"].to_numpy()).mean() * np.sqrt(252) / np.std(r - curve["cash_return"].to_numpy(), ddof=1)) if "cash_return" in curve and len(r)>1 and np.std(r - curve["cash_return"].to_numpy(), ddof=1)>1e-12 else None,
            "max_drawdown": mdd, "calmar": float(cagr / mdd) if mdd > 1e-12 else None,
            "realized_daily_cvar95": tail_mean(-r, .95),
            "annual_gross_turnover": float(curve.turnover.mean() * 252),
            "mean_exposure": float(curve.exposure.mean()),
            "target_breached": bool(mdd > s.max_drawdown),
            "halted": bool(curve.halted.any())}


def validate_config(config, market):
    entry = config["markets"][market]
    assets, caps = entry["assets"], config["class_caps"]
    if not assets or any(a["class"] not in caps for a in assets.values()):
        raise ValueError("所有资产必须有对应类别上限")
    if any(not 0 <= x <= 1 for x in caps.values()):
        raise ValueError("类别上限必须在[0,1]")
    return assets, caps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--cash-yields", type=Path, help="CSV: date,annual_yield; decimals, plus .meta.json")
    parser.add_argument("--market", choices=["CN", "US"], required=True)
    parser.add_argument("--max-drawdown", type=float, required=True)
    parser.add_argument("--frequency", choices=["daily", "weekly"], default="weekly")
    parser.add_argument("--method", choices=["cdar", "hrp", "inverse_vol"], default="cdar")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", action="store_true", help="比较3种算法x2种频率，末252日仅用于留出评估")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assets, caps = validate_config(config, args.market)
    kwargs = {k: v for k, v in config.items() if k in Settings.__dataclass_fields__}
    kwargs.update(max_drawdown=args.max_drawdown, frequency=args.frequency, method=args.method)
    s = Settings(**kwargs)
    if config.get("require_correlation") and s.method == "inverse_vol":
        raise ValueError("当前配置要求相关性模型，禁止逆波动率独立配置")
    prices = read_prices(args.prices, assets)
    meta_path = args.prices.with_suffix(".meta.json")
    if not meta_path.exists():
        raise ValueError("缺少数据元信息 .meta.json；必须声明 market/currency/price_type/source")
    meta = json.loads(meta_path.read_text())
    if meta.get("market") != args.market or meta.get("currency") != config["markets"][args.market]["currency"]:
        raise ValueError("市场/币种不一致；不接受未经统一时区及FX处理的混合面板")
    if meta.get("price_type") != "adjusted_close" or not meta.get("source"):
        raise ValueError("需要已核对分红拆分口径的 adjusted_close 数据及来源")
    cash_returns = None
    cash_meta = None
    cash_audit = None
    required_cash = config.get("cash_yield_source", {}).get("currency") == config["markets"][args.market]["currency"]
    if required_cash and args.cash_yields is None:
        raise ValueError("该人民币配置要求AAA一年期同业存单历史利率，必须提供--cash-yields；不允许静默回退零收益")
    if args.cash_yields is not None:
        from cash_rates import cash_accrual
        yields = pd.read_csv(args.cash_yields, index_col=0, parse_dates=True)
        cash_meta = json.loads(args.cash_yields.with_suffix(".meta.json").read_text())
        if cash_meta.get("currency") != config["markets"][args.market]["currency"] or cash_meta.get("unit") != "decimal":
            raise ValueError("现金序列币种或单位不一致")
        if required_cash and cash_meta.get("series_name") != config["cash_yield_source"]["name"]:
            raise ValueError("现金序列名称与用户指定AAA一年期同业存单不一致")
        cash_returns, cash_audit = cash_accrual(prices.index, yields["annual_yield"])
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("输出目录非空，请使用新的目录以保留研究记录")
    args.out.mkdir(parents=True, exist_ok=True)
    variants = [(args.method, args.frequency)]
    if args.compare:
        if len(prices) < s.lookback + 504:
            raise ValueError("比较模式需要至少504个评价日，末252日保留为测试期")
        methods = config.get("methods", ["cdar", "hrp", "inverse_vol"])
        if config.get("require_correlation") and any(m not in {"cdar", "hrp"} for m in methods):
            raise ValueError("对照算法必须全部考虑相关性")
        variants = [(m, f) for m in methods for f in ("daily", "weekly")]
    summary = []
    plot_curves = {}
    for method, freq in variants:
        chosen = replace(s, method=method, frequency=freq)
        curve, holdings, signals = backtest(prices, assets, caps, chosen, cash_returns=cash_returns)
        name = method + "_" + freq
        folder = args.out / name
        folder.mkdir()
        curve.to_csv(folder / "curve.csv")
        holdings.to_csv(folder / "holdings.csv")
        signals.to_csv(folder / "signals.csv")
        plot_curves[name] = curve.nav
        summary.append({"strategy": name, "segment": "full", **metrics(curve, chosen)})
        if args.compare:
            summary.extend([{"strategy": name, "segment": "validation", **metrics(curve.iloc[:-252], chosen)},
                            {"strategy": name, "segment": "holdout", **metrics(curve.iloc[-252:], chosen)}])
        for year, sub in curve.groupby(curve.index.year):
            summary.append({"strategy": name, "segment": str(year), **metrics(sub, chosen)})
    result = pd.DataFrame(summary)
    result.to_csv(args.out / "metrics.csv", index=False)
    if cash_audit is not None:
        cash_audit.to_csv(args.out / "cash_accrual_audit.csv", index=False)
    recommendation = None
    if args.compare:
        validation = result[(result.segment == "validation") & ~result.target_breached & ~result.halted].copy()
        # Keep non-dominated strategies across annual return, Sharpe and Calmar.
        cols = ["annual_return", "sharpe_cash_benchmark" if cash_returns is not None else "sharpe", "calmar"]
        validation = validation.dropna(subset=cols)
        frontier = []
        for i, row in validation.iterrows():
            other = validation[cols]
            dominated = ((other >= row[cols].astype(float)).all(axis=1) &
                         (other > row[cols].astype(float)).any(axis=1)).any()
            if not dominated:
                frontier.append(row.strategy)
        recommendation = {"selection_uses": "validation_only", "selection_metrics": cols, "pareto_candidates": frontier,
                          "note": "无可行候选时不放松回撤目标；留出期不得用于反复调参。"}
        (args.out / "selection.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2))
    manifest = {"settings": s.__dict__, "market": args.market, "data": meta, "config": config,
                "cash_model": cash_meta, "cash_yields_sha256": hashlib.sha256(args.cash_yields.read_bytes()).hexdigest() if args.cash_yields else None,
                "prices_sha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
                "execution": "signal after t close; fill at t+1 close; earn new weights from t+2 return",
                "warning": "Research simulation; historical constraints do not guarantee future drawdown."}
    (args.out / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, nav in plot_curves.items():
        axes[0].plot(nav.index, nav, label=name)
        peak = nav.cummax().clip(lower=1)
        axes[1].plot(nav.index, nav / peak - 1, label=name)
    axes[0].set_title("SYNTHETIC ENGINE CHECK - NOT INVESTMENT PERFORMANCE" if meta.get("synthetic") else "ETF Rotation - Research Backtest")
    axes[0].set_ylabel("NAV"); axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown"); axes[1].axhline(-s.max_drawdown, color="red", linestyle="--")
    fig.tight_layout(); fig.savefig(args.out / "report.png", dpi=140); plt.close(fig)
    print(result[result.segment == "full"].to_string(index=False))
    print("Written:", args.out.resolve())

if __name__ == "__main__":
    main()
