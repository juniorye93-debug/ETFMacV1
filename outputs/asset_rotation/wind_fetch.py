"""Run inside the user's already authorized WindPy environment; never installs unofficial SDKs."""
import argparse
import datetime as dt
import json
from pathlib import Path


def fetch(w, codes, start, end, market):
    import pandas as pd
    calendar = "SSE" if market == "CN" else "NYSE"
    options = f"Period=D;Days=Trading;TradingCalendar={calendar};PriceAdj=F;Fill=Blank"
    columns = []
    # One code per call avoids ambiguity between multi-code and multi-field WSD layouts.
    for code in codes:
        print(f"Fetching {code} {start}..{end}", flush=True)
        result = w.wsd(code, "close", start, end, options)
        if result.ErrorCode != 0:
            raise RuntimeError(f"Wind WSD失败 {code}: ErrorCode={result.ErrorCode}")
        if len(result.Data) != 1 or len(result.Data[0]) != len(result.Times):
            raise RuntimeError(f"Wind返回格式异常: {code}")
        columns.append(pd.Series(result.Data[0], index=pd.to_datetime(result.Times), name=code))
    return pd.concat(columns, axis=1).sort_index(), options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    p.add_argument("--market", choices=["CN", "US"], required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if dt.date.fromisoformat(a.start) >= dt.date.fromisoformat(a.end):
        raise ValueError("start必须早于end")
    if a.out.exists() or a.out.with_suffix(".meta.json").exists():
        raise ValueError("输出已存在；请使用新文件名")
    config = json.loads(a.config.read_text(encoding="utf-8"))
    entry = config["markets"][a.market]
    try:
        from WindPy import w
    except ImportError as exc:
        raise SystemExit("当前Python未安装官方WindPy。请在已授权且配置官方SDK的Python运行此脚本。") from exc
    started = w.start(waitTime=20)
    if started.ErrorCode != 0 or not w.isconnected():
        raise RuntimeError(f"Wind未连接，ErrorCode={started.ErrorCode}")
    prices, options = fetch(w, list(entry["assets"]), a.start, a.end, a.market)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    prices.index.name = "date"
    prices.to_csv(a.out)
    meta = {"source": "Wind WSD", "market": a.market, "currency": entry["currency"],
            "price_type": "adjusted_close", "synthetic": False, "options": options,
            "requested_start": a.start, "requested_end": a.end,
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_required": "核对Wind代码、复权是否含现金分红、成交额、停牌、QDII溢价；接口已实机连接并取得行情；完整交易可行性仍需检查"}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已导出", a.out, "；运行回测前请核对字段含义和缺失值。")

if __name__ == "__main__":
    main()
