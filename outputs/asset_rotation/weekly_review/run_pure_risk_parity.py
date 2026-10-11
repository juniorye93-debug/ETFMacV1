"""Diagnostic pure equal-risk-contribution benchmark on the same 3 assets as 30/50/20.
Long-only, fully invested, no leverage, no timing/forecast/account stops or asset caps.
"""
import sys,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.covariance import LedoitWolf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import rotation as old
import rotation_v2 as v

def erc_weights(cov):
 c=np.asarray(cov,dtype=float);c=c/max(np.mean(np.diag(c)),1e-12);b=np.ones(len(c))/len(c)
 x=np.sqrt(b/np.diag(c))
 # Exact coordinate minimizers of the strictly convex log-barrier objective.
 for iteration in range(10000):
  for i in range(len(x)):
   cross=float(c[i]@x-c[i,i]*x[i]);disc=float(np.sqrt(cross*cross+4*c[i,i]*b[i]))
   x[i]=2*b[i]/(disc+cross) if cross>=0 else (disc-cross)/(2*c[i,i])
  if np.max(np.abs(x*(c@x)-b))<1e-11:break
 else:raise ValueError('ERC coordinate solver did not converge')
 w=x/x.sum();rc=w*(c@w);rc/=rc.sum()
 if np.max(np.abs(rc-b))>1e-6:raise ValueError('ERC risk contribution tolerance failure')
 return w,rc

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--market',required=True,choices=['CN','US']);a=parser.parse_args()
 cfg=json.loads((ROOT/'config.json').read_text());assets=cfg['markets'][a.market]['assets']
 file=ROOT/'data'/('cn_expanded28_20260924.csv' if a.market=='CN' else 'us_expanded_20260924.csv');allprices=old.read_prices(file,assets);cash,_,_=v.lagged_cash(allprices,a.market);prices=allprices.iloc[252:]
 codes=['510300.SH','CBA00101.CS','518880.SH'] if a.market=='CN' else ['SPY.P','IEF.O','GLD.P']
 dest=ROOT/'weekly_review'/a.market/'pure_erc';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'curve.csv').exists():raise ValueError('Do not overwrite')
 diagnostics=[]
 def allocate(history,current,*args):
  r=history[codes].pct_change(fill_method=None).iloc[-252:]
  cov=LedoitWolf().fit(r.to_numpy()).covariance_+np.eye(3)*1e-12
  w,rc=erc_weights(cov);target=pd.Series(0.,index=history.columns);target.loc[codes]=w
  diagnostics.append({'date':history.index[-1],**{c:float(v) for c,v in zip(codes,rc)}})
  return target,{'status':'equal_risk_contribution'}
 original=old.allocate
 try:
  old.allocate=allocate;s=old.Settings(.08,method='hrp',frequency='weekly',overlay_mode='none')
  result=old.backtest(prices,assets,cfg['class_caps'],s,cash_returns=cash)
 finally:old.allocate=original
 for name,frame in zip(['curve','holdings','signals'],result):frame.to_csv(dest/(name+'.csv'))
 pd.DataFrame(diagnostics).to_csv(dest/'risk_contributions.csv',index=False)
 summary={'market':a.market,'strategy':'pure_erc','segment':'full',**v.evaluate(result[0],v.Model(.08))};pd.DataFrame([summary]).to_csv(dest/'metrics.csv',index=False)
 (dest/'method.json').write_text(json.dumps({'assets':codes,'risk_contribution_budget':[1/3]*3,'lookback':252,'covariance':'LedoitWolf','frequency':'weekly','constraints':'long only, fully invested, no leverage; same unconstrained capital caps as 30/50/20; not an 8%-compliant candidate','execution':'same engine/date/cash/cost as v2 comparison'},indent=2));print(summary)
if __name__=='__main__':main()
