"""Causal factor registry and execution-aligned, explicitly matured labels."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import PipelineError

FACTORS = {
    'momentum_20': {'family': 'momentum', 'direction': 1},
    'momentum_63': {'family': 'momentum', 'direction': 1},
    'momentum_126_skip21': {'family': 'momentum', 'direction': 1},
    'reversal_5': {'family': 'reversal', 'direction': 1},
    'low_volatility_63': {'family': 'defensive', 'direction': 1},
    'trend_distance_63': {'family': 'trend', 'direction': 1},
    'path_efficiency_63': {'family': 'trend', 'direction': 1},
    'relative_momentum_63': {'family': 'relative', 'direction': 1},
}


@dataclass(frozen=True)
class ResearchSettings:
    horizon_weeks: int = 4
    evaluation_horizons_weeks: tuple = (1, 4, 13)
    train_weeks: int = 260
    min_train_weeks: int = 104
    refit_weeks: int = 4
    evidence_weeks: int = 104
    min_evidence_weeks: int = 52
    ridge_alpha: float = 100.
    min_assets: int = 5
    quantiles: int = 5
    covariance_sessions: int = 252
    tilt_strength: float = 1.
    volatility_target: float = .10
    risk_paths: int = 128
    risk_horizon_sessions: int = 63
    risk_confidence: float = .95
    exposure_grid: int = 21
    fee_bps: float = 3.
    slippage_bps: float = 5.
    bootstrap_draws: int = 512
    bootstrap_block_sessions: int = 20
    alpha: float = .05
    seed: int = 8108
    factors: tuple = tuple(FACTORS)
    strategy_components: tuple = ('equal_factor_rp', 'ic_factor_rp', 'ridge_factor_rp')
    primary_strategy: str = 'ridge_risk_forecast_rp'
    cost_multipliers: tuple = (1., 2.)

    def __post_init__(self):
        if min(self.horizon_weeks, self.min_train_weeks, self.train_weeks, self.refit_weeks,
               self.min_evidence_weeks, self.evidence_weeks, self.min_assets, self.quantiles,
               self.risk_paths, self.risk_horizon_sessions, self.bootstrap_draws,
               self.bootstrap_block_sessions) < 1:
            raise ValueError('Research windows/counts must be positive')
        if self.train_weeks < self.min_train_weeks or self.evidence_weeks < self.min_evidence_weeks:
            raise ValueError('Training/evidence windows cannot be smaller than minimums')
        if self.exposure_grid < 2 or self.covariance_sessions < 126:
            raise ValueError('Need exposure_grid>=2 and covariance_sessions>=126')
        if not 0 < self.alpha < 1 or not 0 < self.risk_confidence < 1:
            raise ValueError('Invalid confidence level')
        if self.ridge_alpha <= 0 or self.tilt_strength < 0 or not 0 < self.volatility_target < 1:
            raise ValueError('Invalid model/risk settings')
        if min(self.fee_bps, self.slippage_bps) < 0 or not self.factors or len(set(self.factors)) != len(self.factors):
            raise ValueError('Invalid costs or factor list')
        if not self.cost_multipliers or min(self.cost_multipliers) <= 0 or 1. not in self.cost_multipliers:
            raise ValueError('Cost multipliers must be positive and include 1')
        if min(self.evaluation_horizons_weeks) < 1:
            raise ValueError('Invalid evaluation horizons')


def weekly_positions(index):
    week = index.to_period('W-FRI')
    return np.flatnonzero(np.r_[True, week[1:] != week[:-1]])


def build_factors(prices, cash):
    r = prices.pct_change(fill_method=None)
    sigma = r.rolling(63, min_periods=63).std().clip(lower=1e-6)
    cash_nav = (1+cash).cumprod()
    relative = prices.div(cash_nav, axis=0)
    factors = {
        'momentum_20': relative/relative.shift(20)-1,
        'momentum_63': relative/relative.shift(63)-1,
        'momentum_126_skip21': relative.shift(21)/relative.shift(126)-1,
        'reversal_5': -(relative/relative.shift(5)-1),
        'low_volatility_63': -sigma*np.sqrt(252),
        'trend_distance_63': prices/prices.rolling(63).mean()-1,
        'path_efficiency_63': (prices/prices.shift(63)-1)/r.abs().rolling(63).sum().replace(0, np.nan),
    }
    # Equal-weight eligible-price market proxy; only same-day observed values.
    market = r.mean(axis=1).add(1).cumprod()
    factors['relative_momentum_63'] = (prices/prices.shift(63)-1).sub(market/market.shift(63)-1, axis=0)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in factors.items()}


def load_external_factors(path, prices, names):
    """Accept custom factors only with explicit point-in-time source receipts."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if meta.get('synthetic') is not False or not meta.get('source'):
        raise PipelineError('FACTOR_PROVENANCE_INVALID', 'Custom factors need real source provenance')
    definitions = meta.get('factors', {})
    for name in names:
        if definitions.get(name, {}).get('pit_verified') is not True:
            raise PipelineError('FACTOR_PIT_UNVERIFIED', f'{name}: point-in-time availability unverified')
    frame = pd.read_csv(path, parse_dates=['date', 'available_at'])
    if frame[['date', 'available_at', 'code']].isna().any().any() or frame.duplicated(['date', 'code']).any() or (frame.available_at > frame.date).any():
        raise PipelineError('FACTOR_TIMING_INVALID', 'Duplicate rows or availability after signal date')
    if set(frame.code)-set(prices):
        raise PipelineError('FACTOR_UNIVERSE_MISMATCH', 'Custom factor assets do not match price universe')
    data = {name: frame.pivot(index='date', columns='code', values=name).reindex(index=prices.index, columns=prices.columns)
            for name in names}
    return data, definitions


def cross_sectional_z(frame):
    center = frame.median(axis=1)
    spread = frame.sub(center, axis=0).abs().median(axis=1)*1.4826
    spread = spread.where(spread > 1e-12, frame.std(axis=1)).replace(0, np.nan)
    clipped = frame.sub(center, axis=0).div(spread, axis=0).clip(-3, 3)
    std = clipped.std(axis=1).replace(0, np.nan)
    return clipped.sub(clipped.mean(axis=1), axis=0).div(std, axis=0).fillna(0.)


def build_panel(prices, cash, factors, settings, horizon=None):
    h = horizon or settings.horizon_weeks
    positions = weekly_positions(prices.index)
    r = prices.pct_change(fill_method=None)
    vol = r.rolling(63, min_periods=63).std().clip(lower=1e-6)
    short = r.rolling(20, min_periods=20).std().clip(lower=1e-6)
    standardized = {k: cross_sectional_z(v) for k, v in factors.items()}
    rows = []
    for j, pos in enumerate(positions):
        if pos < settings.covariance_sessions or pos+1 >= len(prices):
            continue
        fill = pos+1
        end = int(positions[j+h])+1 if j+h < len(positions) else None
        if end is not None and end >= len(prices):
            end = None
        valid = vol.iloc[pos].notna() & prices.iloc[pos].notna()
        for code in prices.columns[valid]:
            future_cash = np.prod(1+cash.iloc[fill+1:end+1]) if end is not None else np.nan
            target = prices.iloc[end][code]/prices.iloc[fill][code]/future_cash-1 if end is not None else np.nan
            future_vol = r.iloc[fill+1:end+1][code].std(ddof=1)*np.sqrt(252) if end is not None else np.nan
            # Forecast scale is known at signal time, including for immature labels.
            sessions = 5*h
            row = {'date': prices.index[pos], 'code': code, 'position': pos, 'signal_week': j,
                   'fill_date': prices.index[fill], 'label_end': prices.index[end] if end is not None else pd.NaT,
                   'target': target, 'future_vol': future_vol, 'horizon_weeks': h,
                   'scale': float(vol.iloc[pos][code])*np.sqrt(sessions),
                   'prior_vol': float(vol.iloc[pos][code])*np.sqrt(252),
                   'risk_vol_ratio': float(np.log(short.iloc[pos][code]/vol.iloc[pos][code]))}
            row.update({k: float(v.iloc[pos][code]) for k, v in factors.items()})
            row.update({'z_'+k: float(v.iloc[pos][code]) for k, v in standardized.items()})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['date', 'code']).reset_index(drop=True)
