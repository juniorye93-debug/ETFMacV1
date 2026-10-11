"""Lagged cash accrual from annual yield observations; all annual inputs are decimals."""
import numpy as np
import pandas as pd


def cash_accrual(prices_index, annual_yields, max_stale_days=15):
    dates=pd.DatetimeIndex(prices_index)
    y=annual_yields.astype(float).dropna().sort_index()
    if y.index.has_duplicates or y.empty or not np.isfinite(y).all() or (y <= -1).any() or (y > .5).any():
        raise ValueError('现金利率序列异常；年化利率必须以小数表示')
    calendar=pd.date_range(dates[0]-pd.Timedelta(days=1),dates[-1],freq='D')
    union=y.index.union(calendar)
    known=y.reindex(union).ffill().reindex(calendar)
    published=pd.Series(y.index,index=y.index).reindex(union).ffill().reindex(calendar)
    # Interest on calendar day d uses the most recent observation no later than d-1.
    lagged=known.shift(1)
    observation_date=published.shift(1)
    daily=lagged/365
    out=pd.Series(0.,index=dates,name='cash_return')
    audit=[]
    for i in range(1,len(dates)):
        interval=pd.date_range(dates[i-1]+pd.Timedelta(days=1),dates[i],freq='D')
        rates=daily.reindex(interval)
        stamp=observation_date.reindex(interval)
        if rates.isna().any():
            raise ValueError(f'利率历史不足，不能回填未来数据: {dates[i]}')
        ages=(pd.Series(interval,index=interval)-stamp).dt.days
        if ages.max()>max_stale_days:
            raise ValueError(f'现金利率超过{max_stale_days}天未更新: {dates[i]}')
        out.iloc[i]=np.prod(1+rates.to_numpy())-1
        audit.append({'date':dates[i], 'calendar_days':len(interval), 'cash_return':out.iloc[i],
                      'last_known_annual_yield':float(lagged.loc[interval[-1]]),
                      'last_observation_date':stamp.iloc[-1]})
    return out,pd.DataFrame(audit)
