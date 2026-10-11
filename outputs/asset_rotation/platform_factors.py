"""Independent causal implementations of researched factor families.

Scores describe past information; none is an annual return forecast or confidence.
Window choices are declared research parameters, never drawdown constraints.
"""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from equity_signals import price_signals,relative_rank,valuation_signals,SignalSettings

@dataclass(frozen=True)
class FactorSettings:
    trend_days:int=25
    bias_ma_days:int=90
    bias_slope_days:int=20
    rsrs_regression_days:int=18
    rsrs_standardization_days:int=600
    rsrs_vol_rank_days:int=252
    fft_days:int=250
    fft_cutoff:int=60
    kalman_days:int=20
    residual_days:int=252
    residual_formation_days:int=63

BASE_FACTORS=['mom_1m','mom_3m','mom_6m','mom_12m','mom_13612U','mom_13612W','skip_12_1','skip_6_1','risk_adjusted_momentum','path_efficiency','enhanced_sharpe_maxr','enhanced_path_mstd','short_reversal_1m']
NEW_PRICE_FACTORS=['weighted_trend25','weighted_trend63','bias_velocity','fft_trend','kalman_filter_trend','kalman_smooth_window_trend','sector_momentum','momentum_acceleration','relative_strength','market_residual_momentum','low_volatility','dip_repair','cst_price_proxy','volume_pressure_proxy','low_crowding_proxy']
VALUATION_FACTORS=['pe_cheapness','pb_cheapness','earnings_yield','book_yield','value_momentum','cst_value_lowvol_momentum']
RSRS_FACTORS=['rsrs','rsrs_dampened']
FUNDAMENTAL_FACTORS=['roe_improvement','profit_growth','consensus_growth_revision','eps_revision','quality_value_momentum','fair_pb_proxy']
SHARE_FACTORS=['share_flow5','share_flow21','flow_momentum']

def regression_score(y,weights=None):
    """Slope x R² in daily units. WLS uses consistent squared polyfit weights."""
    y=np.asarray(y,float);n=y.shape[-1];x=np.arange(n,dtype=float)
    w=np.ones(n) if weights is None else np.asarray(weights,float)**2
    w=w/w.sum();xm=np.sum(w*x);ym=np.sum(y*w,axis=-1,keepdims=True)
    beta=np.sum((y-ym)*(x-xm)*w,axis=-1)/np.sum(w*(x-xm)**2)
    fitted=ym+beta[...,None]*(x-xm)
    sse=np.sum(w*(y-fitted)**2,axis=-1);sst=np.sum(w*(y-ym)**2,axis=-1)
    r2=np.where(sst>1e-18,np.clip(1-sse/np.maximum(sst,1e-18),0,1),0)
    valid=np.isfinite(y).all(axis=-1)
    return np.where(valid,beta*r2,np.nan),np.where(valid,beta,np.nan),np.where(valid,r2,np.nan)

def rolling_trend(frame,n,weighted=False):
    out=pd.DataFrame(np.nan,index=frame.index,columns=frame.columns)
    if len(frame)<n:return out
    for code in frame:
        windows=sliding_window_view(np.log(frame[code].to_numpy()),n)
        score,_,_=regression_score(windows,np.linspace(1,2,n) if weighted else None)
        out.iloc[n-1:,out.columns.get_loc(code)]=score
    return out

def filtered_trends(prices,s):
    fft=pd.DataFrame(np.nan,index=prices.index,columns=prices.columns)
    kf=fft.copy();ks=fft.copy()
    for code in prices:
        x=np.log(prices[code].to_numpy())
        if len(x)>=s.fft_days:
            windows=sliding_window_view(x,s.fft_days);good=np.isfinite(windows).all(axis=1)
            z=np.zeros_like(windows);z[good]=windows[good]
            freq=np.fft.fft(z,axis=1);freq[:,s.fft_cutoff:s.fft_days-s.fft_cutoff]=0
            smooth=np.fft.ifft(freq,axis=1).real
            score,_,_=regression_score(smooth[:,-20:]);score[~good]=np.nan
            fft.iloc[s.fft_days-1:,fft.columns.get_loc(code)]=score
        if len(x)<s.kalman_days:continue
        windows=sliding_window_view(x,s.kalman_days);good=np.isfinite(windows).all(axis=1)
        y=np.where(good[:,None],windows,0.);n=len(y)
        noise=np.maximum(np.var(np.diff(y,axis=1),axis=1),1e-10);q=.1*noise;r=noise
        state=np.zeros_like(y);variance=np.zeros_like(y);state[:,0]=y[:,0];variance[:,0]=r
        for j in range(1,s.kalman_days):
            pred=variance[:,j-1]+q;gain=pred/(pred+r)
            state[:,j]=state[:,j-1]+gain*(y[:,j]-state[:,j-1]);variance[:,j]=(1-gain)*pred
        smooth=state.copy()
        for j in range(s.kalman_days-2,-1,-1):
            gain=variance[:,j]/(variance[:,j]+q)
            smooth[:,j]=state[:,j]+gain*(smooth[:,j+1]-state[:,j])
        for target,values in [(kf,state),(ks,smooth)]:
            score,_,_=regression_score(values);score[~good]=np.nan
            target.iloc[s.kalman_days-1:,target.columns.get_loc(code)]=score
    return fft,kf,ks

def rsrs_signals(high,low,close,s):
    n=s.rsrs_regression_days
    vl=low.rolling(n,min_periods=n).var();vh=high.rolling(n,min_periods=n).var()
    cov=pd.DataFrame({c:high[c].rolling(n,min_periods=n).cov(low[c]) for c in close},index=close.index)
    beta=cov/vl.where(vl>1e-16);r2=(cov**2/(vl*vh).where(lambda x:x>1e-20)).clip(0,1)
    avg=beta.rolling(s.rsrs_standardization_days,min_periods=s.rsrs_standardization_days).mean()
    sd=beta.rolling(s.rsrs_standardization_days,min_periods=s.rsrs_standardization_days).std()
    z=(beta-avg)/sd.where(sd>1e-12)
    vol=close.pct_change(fill_method=None).rolling(20,min_periods=20).std()
    percentile=vol.rolling(s.rsrs_vol_rank_days,min_periods=s.rsrs_vol_rank_days).rank(pct=True)
    return {'rsrs_beta':beta,'rsrs_r2':r2,'rsrs':z*r2,'rsrs_dampened':z*r2.pow(2*percentile)}

def build_factors(prices,cash,benchmark,ohlcv=None,valuation=None,s=FactorSettings(),financial=None,shares=None):
    out=price_signals(prices,cash);ret=prices.pct_change(fill_method=None)
    out['weighted_trend25']=rolling_trend(prices,25,True)
    out['weighted_trend63']=rolling_trend(prices,63,True)
    bias=prices/prices.rolling(s.bias_ma_days,min_periods=s.bias_ma_days).mean()
    out['bias_velocity']=rolling_trend(bias,s.bias_slope_days)
    out['fft_trend'],out['kalman_filter_trend'],out['kalman_smooth_window_trend']=filtered_trends(prices,s)
    out['sector_momentum']=(12*out['mom_1m']+4*out['mom_3m']+2*out['mom_6m'])/18
    out['momentum_acceleration']=out['mom_1m']-out['mom_1m'].shift(21)
    ratio=prices.div(prices[benchmark],axis=0)
    out['relative_strength']=ratio/ratio.rolling(63,min_periods=63).mean()-1
    mr=ret[benchmark];mvar=mr.rolling(s.residual_days,min_periods=s.residual_days).var()
    beta=pd.DataFrame({c:ret[c].rolling(s.residual_days,min_periods=s.residual_days).cov(mr)/mvar for c in prices},index=prices.index)
    # Parameters available before the residual-return day; single-market proxy, not FF3.
    intercept=ret.rolling(s.residual_days,min_periods=s.residual_days).mean()-beta.mul(mr.rolling(s.residual_days,min_periods=s.residual_days).mean(),axis=0)
    residual=ret-intercept.shift(1)-beta.shift(1).mul(mr,axis=0)
    vol=residual.rolling(s.residual_formation_days,min_periods=s.residual_formation_days).std()
    out['market_residual_momentum']=residual.rolling(s.residual_formation_days,min_periods=s.residual_formation_days).mean()/vol.where(vol>1e-12)
    out['low_volatility']=-ret.rolling(63,min_periods=63).std()
    dd=prices/prices.rolling(252,min_periods=126).max()-1
    out['dip_repair']=relative_rank(-dd)*relative_rank(out['mom_1m']).where(out['mom_1m']>0)
    out['cst_price_proxy']=(relative_rank(-dd)+relative_rank(out['low_volatility'])+relative_rank(prices.shift(21)/prices.shift(61)-1))/3
    if ohlcv is not None:
        def panel(field):return ohlcv.pivot(index='date',columns='code',values=field).reindex(index=prices.index,columns=prices.columns)
        high,low,oclose=panel('high'),panel('low'),panel('close')
        # Only ratios need agree: forward-adjusted levels may change with retrieval date.
        discrepancy=(oclose.pct_change(fill_method=None)-ret).abs()
        good=(discrepancy.fillna(0)<.001)&oclose.notna()
        high=high.where(good);low=low.where(good)
        out.update(rsrs_signals(high,low,prices,s))
        volume=panel('volume').where(lambda x:x>=0)
        signed=(np.sign(ret)*volume).rolling(21,min_periods=21).sum()/volume.rolling(21,min_periods=21).sum().where(lambda x:x>0)
        out['volume_pressure_proxy']=signed
        # Volume change is not fund subscriptions/redemptions or actual institutional flow.
        turnover=volume/volume.rolling(63,min_periods=63).mean().where(lambda x:x>0)
        out['low_crowding_proxy']=-turnover.rolling(21,min_periods=21).mean()
    if valuation is not None:
        pe=valuation.pivot(index='date',columns='etf',values='pe_ttm').reindex(index=prices.index,columns=prices.columns)
        pb=valuation.pivot(index='date',columns='etf',values='pb_lf').reindex(index=prices.index,columns=prices.columns)
        out.update(valuation_signals(pe,pb,allow_unverified_diagnostic=True))
        out['earnings_yield']=1/pe.where(pe>0).shift(1)
        out['book_yield']=1/pb.where(pb>0).shift(1)
        cheap=(relative_rank(out['pe_cheapness'])+relative_rank(out['pb_cheapness']))/2
        out['value_momentum']=(cheap+relative_rank(out['skip_12_1']))/2
        out['cst_value_lowvol_momentum']=(relative_rank(out['earnings_yield'])+relative_rank(out['low_volatility'])+relative_rank(prices.shift(21)/prices.shift(61)-1))/3
    if financial is not None:
        def field(k):return financial.pivot(index='date',columns='etf',values=k).reindex(index=prices.index,columns=prices.columns).shift(1)
        roe=field('roe_ttm')/100;profit=field('profit_ttm');growth=field('west_netprofit_yoy')/100;eps=field('west_eps_fy1')
        out['roe_improvement']=roe-roe.shift(63)
        out['profit_growth']=(profit/profit.shift(252)-1).where((profit>0)&(profit.shift(252)>0))
        out['consensus_growth_revision']=growth-growth.shift(63)
        out['eps_revision']=(eps/eps.shift(63)-1).where((eps>0)&(eps.shift(63)>0))
        # Calendar-year FY1 rollover is not a genuine earnings revision. Mask windows
        # spanning Jan1; unidentified other provider rollovers remain a limitation.
        sameyear=pd.Series(prices.index.year,index=prices.index)==pd.Series(prices.index.year,index=prices.index).shift(64)
        out['eps_revision']=out['eps_revision'].where(sameyear,axis=0)
        if 'pe_cheapness' in out:
            cheap=(relative_rank(out['pe_cheapness'])+relative_rank(out['pb_cheapness']))/2
            out['quality_value_momentum']=(relative_rank(out['roe_improvement'])+cheap+relative_rank(out['skip_12_1']))/3
            # Residual-income fair-PB PROXY, not Guolian's full analyst-ROE FVB.
            # 8% is this valuation discount assumption, unrelated to account D.
            book=pd.DataFrame(1.,index=prices.index,columns=prices.columns);fair=book.copy();ke=.08
            initial=roe.clip(-.2,.5)
            for year in range(1,11):
                r=initial if year<=3 else initial+(ke-initial)*(year-3)/7
                fair+=(r-ke)*book/(1+ke)**year;book*=1+.5*r
            pb=valuation.pivot(index='date',columns='etf',values='pb_lf').reindex(index=prices.index,columns=prices.columns).shift(1)
            out['fair_pb_proxy']=(fair/pb-1).where(pb>0)
    if shares is not None:
        shares=shares.copy();shares['date']=pd.to_datetime(shares['date'],format='mixed').dt.normalize()
        units=shares.pivot(index='date',columns='code',values='shares').reindex(index=prices.index,columns=prices.columns).where(lambda x:x>0)
        for n in [5,21]:out[f'share_flow{n}']=units.shift(1)/units.shift(n+1)-1
        out['flow_momentum']=(relative_rank(out['share_flow21'])+relative_rank(out['mom_13612W']))/2
    for k in out:out[k]=out[k].where(prices.notna())
    return out

def factor_manifest(s=FactorSettings()):
    return {'settings':asdict(s),'base_factors':BASE_FACTORS,'new_price_factors':NEW_PRICE_FACTORS,'valuation_factors':VALUATION_FACTORS,'rsrs_factors':RSRS_FACTORS,'fundamental_factors':FUNDAMENTAL_FACTORS,'share_factors':SHARE_FACTORS,'source_definition':'Independent mathematical implementations; platform configurations adapted to weekly next-close trading','unverified_valuation':'Valuation, quality, consensus and shares diagnostic-only; excluded from learned prediction and accepted trading models','market_residual':'Single-market rolling regression adaptation; not paper Fama-French three-factor residual momentum','volume_proxy':'Signed volume and volume ratios, not ETF share flows','share_flow':'Lagged unit_total percentage change; historical unit revisions/splits not audited, not currency net inflow','fair_pb_proxy':'Residual-income valuation: currentROE3 years then7-year mean reversion to8% cost of equity,50% retention; no analyst expectedROE input, not original Guolian FVB','kalman':'Windowed log-price filter/smoother, Q=.1x daily variance and R=daily variance; different noise choices from source example','weighted_trend':'Daily log-price slope x consistently weighted R²; no exponential annual extrapolation','RSRS':'18-day high/low beta; trailing600 beta z score; no past252 fixed-weight drawdown constraint'}
