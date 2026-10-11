"""Causal research signals: inputs for a forecast, never expected-return forecasts.

Independent implementation from published mathematical definitions. Source notes
and pinned reference code live in signal_research/. No allocation or trade gate here.
Trading-day 21/63/126/252 windows approximate calendar months; they are NOT a
drawdown estimation window. All windows and combinations remain research choices.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SignalSettings:
    month_sessions:int=21
    continuity_sessions:int=63
    valuation_sessions:int=850
    valuation_min_sessions:int=252
    valuation_lag_sessions:int=1

def relative_rank(frame):
    """Higher signal -> closer to one; ties use midranks; singleton is unknown."""
    count=frame.notna().sum(axis=1)
    return frame.rank(axis=1,method='average').sub(1).div((count-1).where(count>1),axis=0)

def trailing_return(prices,n):
    full=prices.rolling(n+1,min_periods=n+1).count().eq(n+1)
    return (prices/prices.shift(n)-1).where(full)

def price_signals(prices,cash,s=SignalSettings()):
    if not prices.index.equals(cash.index) or cash.isna().any() or (cash<=-1).any():
        raise ValueError('cash must contain complete aligned simple daily returns')
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('prices require sorted unique dates')
    if (prices<=0).any().any():raise ValueError('positive total-return-adjusted prices required')
    cashnav=(1+cash).cumprod();ret=prices.pct_change(fill_method=None)
    excess=ret.sub(cash,axis=0);out={}
    for m in (1,3,6,12):
        n=m*s.month_sessions
        out[f'mom_{m}m']=trailing_return(prices,n)
        cashgrowth=cashnav/cashnav.shift(n)
        out[f'cash_relative_{m}m']=(out[f'mom_{m}m']+1).div(cashgrowth,axis=0)-1
    for prefix in ('mom','cash_relative'):
        frames=[out[f'{prefix}_{m}m'] for m in (1,3,6,12)]
        out[prefix+'_13612U']=sum(frames)/4
        out[prefix+'_13612W']=sum(w*f for w,f in zip((12,4,2,1),frames))/19
    # Predeclared reversal direction, tested independently; never flip a factor's
    # sign after viewing future IC merely to manufacture a positive correlation.
    out['short_reversal_1m']=-out['mom_1m']
    # Skip the latest month: P(t-1m)/P(t-12m)-1, not 12m return minus 1m return.
    for m in (6,12):
        recent=s.month_sessions;old=m*s.month_sessions
        out[f'skip_{m}_1']=(prices.shift(recent)/prices.shift(old)-1).where(
            prices.shift(recent).rolling(old-recent+1,min_periods=old-recent+1).count().eq(old-recent+1)&prices.notna())
    n=s.continuity_sessions
    std=excess.rolling(n,min_periods=n).std().where(lambda x:x>1e-12)
    out['risk_adjusted_momentum']=excess.rolling(n,min_periods=n).mean()/std*np.sqrt(n)
    distance=ret.abs().rolling(n,min_periods=n).sum().where(lambda x:x>1e-12)
    out['path_efficiency']=trailing_return(prices,n)/distance
    out['max_daily_return']=ret.rolling(n,min_periods=n).max()
    out['multi_day_momentum_vol']=trailing_return(prices,s.month_sessions).rolling(n,min_periods=n).std()
    out['enhanced_sharpe_maxr']=relative_rank(out['risk_adjusted_momentum'])*relative_rank(-out['max_daily_return'])
    out['enhanced_path_mstd']=relative_rank(out['path_efficiency'])*relative_rank(-out['multi_day_momentum_vol'])
    out['rank_13612U']=relative_rank(out['mom_13612U'])
    out['rank_skip12_1']=relative_rank(out['skip_12_1'])
    # A slope or momentum score is neither an annual expected return nor a probability.
    return out

def valuation_signals(pe,pb,s=SignalSettings(),*,pit_verified=False,allow_unverified_diagnostic=False):
    if not pit_verified and not allow_unverified_diagnostic:
        raise ValueError('PE/PB vintages and historical index mappings are not PIT verified')
    out={}
    for key,frame in [('pe',pe),('pb',pb)]:
        valid=frame.where(frame>0).shift(s.valuation_lag_sessions)
        # Percentile within each index's own history; negative PE is not "cheap".
        rank=valid.rolling(s.valuation_sessions,min_periods=s.valuation_min_sessions).rank(pct=True)
        out[key+'_cheapness']=1-rank
    return out

def weekly_panel(prices,cash,s=SignalSettings()):
    features=price_signals(prices,cash,s)
    period=prices.index.to_period('W-FRI')
    pos=np.flatnonzero(np.r_[True,period[1:]!=period[:-1]])
    frames=[]
    for key,frame in features.items():
        frame=frame.iloc[pos].copy();frame.index.name='date';frame.columns.name='code'
        frames.append(frame.stack(future_stack=True).rename(key))
    return pd.concat(frames,axis=1).reset_index()

def manifest(s=SignalSettings()):
    return {'settings':asdict(s),'implementation':'paper-inspired independent signal library, not a full paper backtest reproduction','uses':'candidate predictors and diagnostics; not annualized forecasts or final weights','execution':'weekly first trading day W-FRI close; t+1 close assumed by diagnostics','adaptations':['21 trading sessions approximates one calendar month','same 63-session continuity window for initial ablation, not papers optimal fitted windows','850-session valuation window follows Huatai 2018 example; 252 minimum and one-session lag are local diagnostic assumptions','valuation unavailable to live/production models until vintage and index-history audit']}
