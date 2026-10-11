"""Transparent weekly rebalanced baselines, with identical execution delay and costs."""
from pathlib import Path
import json
import pandas as pd
from rotation import Settings, transact, metrics

ROOT = Path(__file__).resolve().parent


def run_baseline(prices, target, settings):
    returns = prices.pct_change(fill_method=None)
    weights = pd.Series(0., index=prices.columns)
    nav = peak = 1.
    pending = None
    records = []
    weeks = prices.index.to_period('W-FRI')
    for i in range(settings.lookback, len(prices)):
        old_nav = nav
        r = returns.iloc[i]
        if r.isna().any():
            raise ValueError('基准要求已完整上市的资产历史')
        values = weights * (1 + r)
        cash = (1 - weights.sum()) * (1 + settings.cash_rate) ** (1 / 252)
        growth = values.sum() + cash
        nav *= growth
        weights = values / growth
        turnover = fee = 0.
        if pending is not None:
            nav, turnover, fee = transact(nav, weights, pending, (settings.fee_bps + settings.slippage_bps) / 10000)
            weights = pending.copy()
            pending = None
        peak = max(peak, nav)
        if i == settings.lookback or weeks[i] != weeks[i - 1]:
            pending = target.copy()
        records.append({'date': prices.index[i], 'nav': nav, 'return': nav / old_nav - 1,
                        'drawdown': 1 - nav / peak, 'turnover': turnover, 'cost_fraction': fee,
                        'exposure': weights.sum(), 'halted': False})
    return pd.DataFrame(records).set_index('date')


def main():
    config = json.loads((ROOT / 'config_legacy.json').read_text())
    rows = []
    s = Settings(.08)
    for market in ['CN', 'US']:
        p = pd.read_csv(ROOT / 'data' / f'{market.lower()}_wind_20260924.csv', index_col=0, parse_dates=True)
        assets = config['markets'][market]['assets']
        classes = sorted({a['class'] for a in assets.values()})
        equal = pd.Series(0., index=p.columns)
        for cls in classes:
            names = [a for a in assets if assets[a]['class'] == cls]
            equal[names] = 1 / len(classes) / len(names)
        balanced = equal * 0
        equity, bond = ('510300.SH', '511010.SH') if market == 'CN' else ('SPY.P', 'IEF.O')
        balanced[equity], balanced[bond] = .6, .4
        folder = ROOT / f'real_{market.lower()}_benchmarks'
        folder.mkdir(exist_ok=True)
        for name, weights in [('equal_asset_class', equal), ('equity60_bond40', balanced)]:
            curve = run_baseline(p, weights, s)
            curve.to_csv(folder / f'{name}.csv')
            for segment, frame in [('full', curve), ('validation', curve.iloc[:-252]), ('holdout', curve.iloc[-252:])]:
                rows.append({'market': market, 'strategy': name, 'segment': segment, **metrics(frame, s)})
    pd.DataFrame(rows).to_csv(ROOT / 'benchmark_metrics.csv', index=False)
    print(pd.DataFrame(rows).query('segment == "full"').to_string(index=False))

if __name__ == '__main__':
    main()
