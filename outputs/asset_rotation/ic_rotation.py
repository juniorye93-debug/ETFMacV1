"""Causal IC-based cross-sectional ETF signal experiments on frozen Wind data.

Ranking alpha is distinct from market timing and expected absolute returns.
All history has already been explored. No source-paper or production claim.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse,json,html,hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from factor_practice import ROOT,PracticeSettings,load_market,prepare_weeks,simulate,weighted_target
from forecast_adaptive import mean_se

OUT=ROOT/'ic_rotation'
FEATURES=['skip_12_1','skip_6_1','mom_13612U','risk_adjusted_momentum',
          'enhanced_path_mstd','market_residual_momentum','momentum_acceleration',
          'rsrs_dampened','low_volatility','short_reversal_1m']
SCORES=['raw_12m','skip_12_1','ic_shrink','ridge_rank','ic_gate','ridge_gate']
LABELS={'pool_erc':'同池无信号ERC','raw_12m':'原12月横向动量','skip_12_1':'12–1横向动量',
        'ic_shrink':'历史IC收缩组合','ridge_rank':'未来排名Ridge',
        'ic_gate':'IC组合＋历史IC门控','ridge_gate':'排名Ridge＋历史IC门控',
        'cross_dual':'多ETF横向IC＋独立现金门槛','skip_dual':'12–1横向动量＋独立现金门槛'}

@dataclass(frozen=True)
class ICSettings:
    min_train_weeks:int=104
    train_weeks:int=208
    refit_weeks:int=13
    evidence_weeks:int=208
    min_gate_weeks:int=52
    ridge_alpha:float=100.
    select_fraction:float=1/3
    gate_family_error:float=.05
    start:str='2019-01-01'

def rank_center(s):
    out=pd.Series(np.nan,index=s.index,dtype=float);valid=s.dropna()
    if len(valid)>1:out.loc[valid.index]=(valid.rank(method='average')-1)/(len(valid)-1)-.5
    return out

def rank_ic(score,target):
    valid=pd.DataFrame({'s':score,'y':target}).dropna()
    if len(valid)<5 or valid.s.nunique()<2 or valid.y.nunique()<2:return np.nan
    return float(valid.s.rank().corr(valid.y.rank()))

def build_panel(d,h,scope):
    p=d['prices'];positions=d['positions'];cashnav=(1+d['cash']).cumprod();rows=[]
    for j,pos in enumerate(positions):
        if pos not in d['packages'] or pos+1>=len(p):continue
        codes=d['packages'][pos]['representatives']
        if scope=='equity':codes=[c for c in codes if d['assets'][c]['class']=='equity']
        if len(codes)<5:continue
        fill=pos+1;end=int(positions[j+h])+1 if j+h<len(positions) else None
        if end is not None and end>=len(p):end=None
        y=(p.iloc[end][codes]/p.iloc[fill][codes]/(cashnav.iloc[end]/cashnav.iloc[fill])-1
           if end is not None else pd.Series(np.nan,index=codes))
        data={'raw_12m':d['factors']['mom_12m'].iloc[pos][codes]}
        for f in FEATURES:
            data[f]=(-d['factors']['mom_1m'].iloc[pos][codes] if f=='short_reversal_1m'
                     else d['factors'][f].iloc[pos][codes])
        ranked={f:rank_center(x) for f,x in data.items()}
        yrank=rank_center(y)
        for c in codes:
            rows.append({'date':p.index[pos],'code':c,'scope':scope,'horizon_weeks':h,
                         'fill_date':p.index[fill],'label_end':p.index[end] if end is not None else pd.NaT,
                         'target':y[c],'target_rank':yrank[c],**{f:x[c] for f,x in ranked.items()}})
    return pd.DataFrame(rows).sort_values(['date','code']).reset_index(drop=True)

def feature_observations(panel):
    rows=[]
    for date,g in panel.groupby('date',sort=True):
        for f in FEATURES+['raw_12m']:
            ic=rank_ic(g[f],g.target)
            rows.append({'date':date,'label_end':g.label_end.iloc[0],'factor':f,
                         'ic':ic,'assets':int(g[[f,'target']].dropna().shape[0])})
    return pd.DataFrame(rows)

def evidence(values,h):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    m,se=mean_se(x,lags=max(4,h));return len(x),m,se

def walk_forward(panel,h,s=ICSettings()):
    """Every training target and IC used for t's score has label_end <= t."""
    panel=panel.copy();obs=feature_observations(panel);dates=sorted(panel.date.unique())
    for name in ['ic_shrink','ridge_rank','ic_gate','ridge_gate']:panel[name]=np.nan
    panel['ic_gate_active']=False;panel['ridge_gate_active']=False
    audits=[];prediction_ic=[];model=None;lastfit=-999;train_end=pd.NaT
    for j,date in enumerate(dates):
        ix=panel.index[panel.date==date];g=panel.loc[ix]
        mature=panel.loc[panel.label_end.notna()&(panel.label_end<=date)&panel.target_rank.notna()]
        train_dates=sorted(mature.date.unique())[-s.train_weeks:]
        train=mature.loc[mature.date.isin(train_dates)]
        known=obs.loc[obs.label_end.notna()&(obs.label_end<=date)]
        known_dates=sorted(known.date.unique())[-s.evidence_weeks:];known=known.loc[known.date.isin(known_dates)]
        weights={};row={'date':date,'train_label_end':train.label_end.max(),
                        'feature_evidence_label_end':known.label_end.max(),'train_weeks':len(train_dates)}
        for f in FEATURES:
            n,m,se=evidence(known.loc[known.factor==f,'ic'],h)
            weights[f]=max(0.,m-se) if n>=s.min_train_weeks and np.isfinite(se) else 0.
            row.update({f+'_mean_ic':m,f+'_ic_se':se,f+'_weeks':n})
        total=sum(weights.values());weights={f:w/total if total>0 else 0. for f,w in weights.items()}
        raw=sum(g[f].fillna(0)*weights[f] for f in FEATURES)
        panel.loc[ix,'ic_shrink']=raw
        row.update({f+'_weight':w for f,w in weights.items()});row['ic_features_active']=sum(w>0 for w in weights.values())
        if len(train_dates)>=s.min_train_weeks and (model is None or j-lastfit>=s.refit_weeks):
            model=Ridge(alpha=s.ridge_alpha).fit(train[FEATURES].fillna(0).to_numpy(),train.target_rank.to_numpy())
            train_end=train.label_end.max();lastfit=j
        pred=model.predict(g[FEATURES].fillna(0).to_numpy()) if model is not None else np.zeros(len(g))
        panel.loc[ix,'ridge_rank']=pred;row['ridge_train_label_end']=train_end
        prior_ic=pd.DataFrame(prediction_ic)
        for source,name in [('ic_shrink','ic_gate'),('ridge_rank','ridge_gate')]:
            if prior_ic.empty:past=pd.DataFrame()
            else:
                past=prior_ic.loc[(prior_ic['source']==source)&prior_ic.label_end.notna()&(prior_ic.label_end<=date)]
                past=past.tail(s.evidence_weeks)
            n,m,se=evidence(past.ic if len(past) else [],h)
            # Two declared learned score families at each fixed horizon/scope.
            z=float(norm.isf(s.gate_family_error/2));lcb=m-z*se
            active=n>=s.min_gate_weeks and np.isfinite(lcb) and lcb>0
            panel.loc[ix,name]=panel.loc[ix,source] if active else 0.
            panel.loc[ix,name+'_active']=active
            row.update({name+'_n':n,name+'_mean_ic':m,name+'_lcb':lcb,name+'_active':active,
                        name+'_evidence_label_end':past.label_end.max() if len(past) else pd.NaT})
        scored=panel.loc[ix]
        for source in ['ic_shrink','ridge_rank']:
            prediction_ic.append({'date':date,'label_end':g.label_end.iloc[0],'source':source,
                                  'ic':rank_ic(scored[source],g.target)})
        audits.append(row)
        if j%200==0:print('IC learning',panel.scope.iloc[0],h,j,str(pd.Timestamp(date).date()),flush=True)
    return panel,obs,pd.DataFrame(audits)

def summarize(panel,obs,market,start='2019-01-01'):
    rows=[];h=int(panel.horizon_weeks.iloc[0]);scope=panel.scope.iloc[0]
    candidates=FEATURES+['raw_12m','ic_shrink','ridge_rank','ic_gate','ridge_gate']
    for segment,a in [('2019_2026_explored',panel.loc[panel.date>=pd.Timestamp(start)]),
                      ('2022_2026_explored',panel.loc[panel.date>=pd.Timestamp('2022-01-01')])]:
        for source in candidates:
            ics=[];spreads=[];all_weeks=0;inactive=0
            for date,g in a.groupby('date'):
                if g.target.notna().sum()<5:continue
                all_weeks+=1;ic=rank_ic(g[source],g.target)
                if not np.isfinite(ic):inactive+=1;continue
                ics.append(ic)
                valid=g.dropna(subset=[source,'target']).sort_values(source,ascending=False,kind='stable')
                k=max(1,int(np.ceil(len(valid)/3)))
                spreads.append(float(valid.target.iloc[:k].mean()-valid.target.iloc[-k:].mean()))
            n,m,se=evidence(ics,h)
            rows.append({'market':market,'scope':scope,'segment':segment,'horizon_weeks':h,'signal':source,
                         'weeks':all_weeks,'defined_ic_weeks':n,'inactive_or_missing_weeks':inactive,
                         'mean_rank_ic':m if n else np.nan,'hac95_low':m-1.96*se,'hac95_high':m+1.96*se,
                         'rank_ic_ir':m/np.std(ics,ddof=1) if n>1 else np.nan,
                         'active_fraction':n/all_weeks if all_weeks else np.nan,
                         'effective_ic_inactive_as_zero':sum(ics)/all_weeks if all_weeks else np.nan,
                         'mean_horizon_top_bottom_spread':np.mean(spreads) if spreads else np.nan,
                         'status':'Exploratory IC; not a future confidence probability or absolute return forecast'})
    return rows

def cross_rule(panel,signal,scope,s=ICSettings(),absolute=False):
    dated={date:g.set_index('code') for date,g in panel.groupby('date')}
    def rule(d,pos,current,state):
        codes=d['packages'][pos]['representatives']
        if scope=='equity':codes=[c for c in codes if d['assets'][c]['class']=='equity']
        date=d['prices'].index[pos]
        if signal=='pool_erc':return weighted_target(d,pos,codes)
        if date not in dated:return weighted_target(d,pos,codes)
        score=dated[date][signal].reindex(codes).dropna()
        if score.nunique()<2:
            # No alpha evidence: transparent same-pool ERC fallback, never a hidden
            # 100% stock/bond bet or a fabricated forecast probability.
            chosen=codes
        else:
            k=max(1,int(np.ceil(len(score)*s.select_fraction)))
            chosen=score.sort_values(ascending=False,kind='stable').index[:k].tolist()
        if absolute:
            chosen=[c for c in chosen if pd.notna(d['factors']['cash_relative_13612U'].iloc[pos][c])
                    and d['factors']['cash_relative_13612U'].iloc[pos][c]>0]
        return weighted_target(d,pos,chosen)
    return rule

def cross_sectional_momentum_rule(panel,scope='all',selection_fraction=1/3):
    """Preferred research signal: fixed12–1 ranking over the whole causal pool.

    Multiple ETFs are selected; no old absolute timing filter. These are raw ERC
    diagnostic targets, requiring a separate account risk layer before trading.
    """
    if not 0<selection_fraction<=1:raise ValueError('invalid selection fraction')
    return cross_rule(panel,'skip_12_1',scope,ICSettings(select_fraction=selection_fraction),absolute=False)

def run(markets,settings=ICSettings()):
    OUT.mkdir(exist_ok=True);stats=[];metrics=[];latest=[];coverage=[]
    for market in markets:
        d=load_market(market);dest=OUT/market;dest.mkdir(exist_ok=True);d['dest']=dest
        # Build causal covariance/availability and dedup before portfolio start;
        # earlier real history is necessary for prequential IC evidence.
        prepare_weeks(d,PracticeSettings(start='2010-01-01'))
        for scope in ['equity','all']:
            for h in [1,4,13]:
                panel,obs,audit=walk_forward(build_panel(d,h,scope),h,settings)
                prefix=f'{scope}_{h}w';panel.to_csv(dest/(prefix+'_predictions.csv'),index=False)
                obs.to_csv(dest/(prefix+'_factor_ic.csv'),index=False);audit.to_csv(dest/(prefix+'_audit.csv'),index=False)
                stats+=summarize(panel,obs,market,settings.start)
                for column in ['train_label_end','feature_evidence_label_end','ridge_train_label_end','ic_gate_evidence_label_end','ridge_gate_evidence_label_end']:
                    v=audit.dropna(subset=[column]);assert (v[column]<=v.date).all(),column
                last=panel.loc[panel.date==panel.date.max()].copy();last['market']=market
                latest+=last.to_dict('records')
                if h in [1,4]:
                    ps=PracticeSettings(start=settings.start)
                    cases=['ridge_rank','ic_shrink'] if h==1 else ['pool_erc','raw_12m','skip_12_1','ic_gate','ridge_gate','cross_dual','ridge_rank','skip_dual']
                    for signal in cases:
                        source={'cross_dual':'ic_gate','skip_dual':'skip_12_1'}.get(signal,signal)
                        name=f'{scope}_{signal}_{h}w'
                        result=simulate(d,name,cross_rule(panel,source,scope,settings,absolute=signal in ['cross_dual','skip_dual']),ps)
                        metrics+=result
                        holdings=pd.read_csv(dest/name/'holdings.csv',index_col=0,parse_dates=True)
                        counts=(holdings.drop(columns='cash')>1e-7).sum(axis=1)
                        coverage.append({'market':market,'scope':scope,'model':name,
                                         'mean_nonzero_positions':counts.mean(),'latest_nonzero_positions':int(counts.iloc[-1]),
                                         'multi_asset_day_fraction':float((counts>=2).mean()),'account_risk_budget_enforced':False})
                        print(market,name,'CAGR',round(100*result[0]['annual_return'],2),'DD',round(100*result[0]['max_drawdown'],2),flush=True)
    pd.DataFrame(stats).to_csv(OUT/'ic_summary.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'portfolio_metrics.csv',index=False)
    pd.DataFrame(latest).to_csv(OUT/'latest_scores.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'selection_coverage.csv',index=False)
    method={'date_local':'2026-10-08','data_asof':'2026-09-24','settings':asdict(settings),'features':FEATURES,
            'label':'Next execution close to h-week future execution close; actual compounded cash-relative return',
            'IC':'Spearman cross-sectional correlation; same date causal eligible/deduplicated asset set; no pooling asset classes as time observations',
            'IC_weight':'max(0, rolling matured meanIC minus HAC standard error), normalized; 104 minimum weeks; a shrinkage heuristic, not95% confidence',
            'rank_model':'Ridge predicts future within-date return ranks, not unscaled absolute returns; impute missing centered ranks as0; no parameter search',
            'score_gate':'52 matured prequential IC weeks minimum; one-sided HAC lower bound with0.05/2 declared learned score families at fixed market/scope/horizon; does not control all repeated looks or all historical experiments',
            'selection':'Top one-third fraction, configurable research choice; all ETF variables, no class caps, full covariance ERC; no IC alpha -> explicit same-pool ERC',
            'absolute_cash_gate':'Only cross_dual and skip_dual ablations retain trailing1/3/6/12-month cash-relative gate; unvalidated timing ablation, NOT forecast certainty',
            'risk_budget':'Signal-isolation cases deliberately unguarded; none called8%-budget strategy; IC is not proof of positive absolute returns',
            'historical_status':'All inspected history exploratory; causal walk-forward execution is not an untouched out-of-sample claim',
            'preferred_research_signal':'Fixed12–1 cross-sectional rank; full available deduplicated ETF pool; no old absolute cash-trend gate',
            'ranking_models_adopted':False,
            'reason':'Positive nominal historical IC alone did not produce superior after-cost portfolio results; mechanical IC gates also failed',
            'unverified_data':'Valuation/financial/consensus/share vintages excluded; frozen Wind adjusted-return basis still requires total-return audit'}
    (OUT/'method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2))
    validate(markets);build_report()

def add_portfolio_ablations(markets,settings=ICSettings()):
    """Run new signal/absolute-gate ablations using saved predictions, no refit."""
    metrics=pd.read_csv(OUT/'portfolio_metrics.csv').to_dict('records')
    coverage=pd.read_csv(OUT/'selection_coverage.csv').to_dict('records')
    for market in markets:
        d=load_market(market);d['dest']=OUT/market
        prepare_weeks(d,PracticeSettings(start='2010-01-01'))
        for scope in ['equity','all']:
            for h,cases in [(1,['ridge_rank','ic_shrink']),(4,['ridge_rank','skip_dual'])]:
                panel=pd.read_csv(d['dest']/f'{scope}_{h}w_predictions.csv',parse_dates=['date','fill_date','label_end'])
                for signal in cases:
                    name=f'{scope}_{signal}_{h}w'
                    if any(r['market']==market and r['model']==name for r in coverage):continue
                    source='skip_12_1' if signal=='skip_dual' else signal
                    result=simulate(d,name,cross_rule(panel,source,scope,settings,absolute=signal=='skip_dual'),PracticeSettings(start=settings.start))
                    metrics+=result
                    hold=pd.read_csv(d['dest']/name/'holdings.csv',index_col=0);counts=(hold.drop(columns='cash')>1e-7).sum(axis=1)
                    coverage.append({'market':market,'scope':scope,'model':name,'mean_nonzero_positions':counts.mean(),
                                     'latest_nonzero_positions':int(counts.iloc[-1]),'multi_asset_day_fraction':float((counts>=2).mean()),'account_risk_budget_enforced':False})
                    print(market,name,'CAGR',round(100*result[0]['annual_return'],2),'DD',round(100*result[0]['max_drawdown'],2),flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'portfolio_metrics.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'selection_coverage.csv',index=False)
    timing_diagnostics(markets);validate(markets);build_report()

def timing_diagnostics(markets):
    """Same-asset time-series IC for old absolute signals, distinct from rank IC."""
    from scipy.stats import spearmanr
    rows=[]
    for market in markets:
        d=load_market(market)
        for h in [1,4,13]:
            p=pd.read_csv(OUT/market/f'equity_{h}w_predictions.csv',parse_dates=['date','label_end'])
            for segment,start in [('2019_2026_explored','2019-01-01'),('2022_2026_explored','2022-01-01')]:
                for code,g in p.loc[p.date>=pd.Timestamp(start)].groupby('code'):
                    g=g.dropna(subset=['target'])
                    for f in ['mom_12m','cash_relative_13612U','skip_12_1']:
                        x=d['factors'][f].reindex(g.date)[code].to_numpy();y=g.target.to_numpy();valid=np.isfinite(x)&np.isfinite(y)
                        if valid.sum()<52:continue
                        xx=x[valid];yy=y[valid];on=xx>0
                        rows.append({'market':market,'segment':segment,'code':code,'horizon_weeks':h,'signal':f,'weeks':int(valid.sum()),
                                     'same_asset_time_series_rank_ic':float(spearmanr(xx,yy).statistic),
                                     'mean_future_cash_excess_when_signal_positive':float(yy[on].mean()) if on.any() else np.nan,
                                     'mean_future_cash_excess_when_signal_nonpositive':float(yy[~on].mean()) if (~on).any() else np.nan,
                                     'interpretation':'Observed time-series relationship, not cross-sectional IC; serial dependence and many comparisons prevent confidence claim'})
    pd.DataFrame(rows).to_csv(OUT/'timing_ic.csv',index=False)

def validate(markets):
    curves=0;signals=0;audits_checked=0
    for market in markets:
        for file in (OUT/market).glob('*_audit.csv'):
            a=pd.read_csv(file,parse_dates=['date'])
            for key in ['train_label_end','feature_evidence_label_end','ridge_train_label_end','ic_gate_evidence_label_end','ridge_gate_evidence_label_end']:
                use=a.dropna(subset=[key]);assert (pd.to_datetime(use[key])<=use.date).all()
            audits_checked+=1
        for file in (OUT/market).glob('*/curve.csv'):
            c=pd.read_csv(file,parse_dates=['date','executed_signal_date']);hold=pd.read_csv(file.parent/'holdings.csv',index_col=0)
            np.testing.assert_allclose(c.nav,np.cumprod(1+c['return']),atol=1e-10)
            np.testing.assert_allclose(hold.sum(axis=1),1,atol=1e-8);assert hold.min().min()>-1e-8
            used=c.dropna(subset=['executed_signal_date']);assert (used.executed_signal_date<used.date).all()
            sig=pd.read_csv(file.parent/'signals.csv');curves+=1;signals+=len(sig)
    (OUT/'validation.json').write_text(json.dumps({'curves_validated':curves,'weekly_signals_checked':signals,
        'mature_label_audits_passed':True,'prediction_panels_with_label_audits':audits_checked,
        'same_day_execution_return_excluded':True,'all_history_exploratory':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

def build_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Arial Unicode MS';plt.rcParams['axes.unicode_minus']=False
    summary=pd.read_csv(OUT/'ic_summary.csv');m=pd.read_csv(OUT/'portfolio_metrics.csv');coverage=pd.read_csv(OUT/'selection_coverage.csv')
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col')
    for col,market in enumerate(['CN','US']):
        for row,scope in enumerate(['equity','all']):
            for signal,h in [('pool_erc',4),('raw_12m',4),('skip_12_1',4),('ridge_rank',4),('ic_shrink',1),('skip_dual',4)]:
                path=OUT/market/f'{scope}_{signal}_{h}w'/'curve.csv'
                if not path.exists():continue
                c=pd.read_csv(path,index_col=0,parse_dates=True);label=LABELS[signal]+(f'（{h}周预测）' if signal in ['ridge_rank','ic_shrink'] else '')
                axes[row,col].plot(c.index,c.nav,lw=1.4,label=label)
            axes[row,col].set_title(market+' · '+('股票ETF横向排序' if scope=='equity' else '每只ETF/中债财富指数为变量'))
            axes[row,col].grid(alpha=.16);axes[row,col].legend(fontsize=8);axes[row,col].set_ylabel('净值')
    fig.suptitle('先检验排序信号：多ETF横向轮动，同池、同协方差和成交口径',fontsize=15)
    fig.text(.5,.012,'2019—2026-09-24 · 风险预算尚未接入这些信号隔离案例 · 历史IC门控仅用已成熟标签 · 全部历史已观察',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(OUT/'nav.png',dpi=150);plt.close(fig)
    def e(x):return html.escape(str(x))
    def p(x):return '—' if pd.isna(x) else f'{x*100:.2f}%'
    def n(x):return '—' if pd.isna(x) else f'{x:.3f}'
    def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
    selected=summary.loc[(summary.segment=='2022_2026_explored')&summary.signal.isin(['raw_12m','skip_12_1','ic_shrink','ridge_rank','ic_gate','ridge_gate'])]
    icrows=[[e(r.market),e(r.scope),str(r.horizon_weeks),e(LABELS.get(r.signal,r.signal)),n(r.mean_rank_ic),n(r.hac95_low),n(r.hac95_high),p(r.active_fraction),n(r.effective_ic_inactive_as_zero)] for r in selected.itertuples()]
    full=m.loc[m.segment=='all_explored'];mr=[]
    for r in full.itertuples():
        z=coverage.loc[(coverage.market==r.market)&(coverage.model==r.model)].iloc[0]
        mr.append([e(r.market),e(r.model),p(r.annual_return),p(r.annual_cash_excess),p(r.annual_volatility),p(r.max_drawdown),f'{z.mean_nonzero_positions:.1f}',str(z.latest_nonzero_positions)])
    latestrows=[]
    for market in ['CN','US']:
        auditfile=OUT/market/'equity_4w_audit.csv'
        if not auditfile.exists():continue
        a=pd.read_csv(auditfile).iloc[-1]
        for f in FEATURES:
            if a[f+'_weight']>0:latestrows.append([market,e(f),n(a[f+'_mean_ic']),n(a[f+'_ic_se']),p(a[f+'_weight']),e(a.feature_evidence_label_end)])
    latest_positions=[]
    cfg=json.loads((ROOT/'config.json').read_text())
    for market in ['CN','US']:
        for scope in ['equity','all']:
            model=f'{scope}_skip_12_1_4w';file=OUT/market/model/'signals.csv'
            if not file.exists():continue
            signal=pd.read_csv(file,index_col=0).iloc[-1];date=signal.name
            for code,asset in cfg['markets'][market]['assets'].items():
                weight=float(signal.get(code,0))
                if weight>1e-7:latest_positions.append({'market':market,'scope':scope,'model':model,'signal_date':date,
                                                         'code':code,'name':asset['name'],'target_weight':weight})
    pd.DataFrame(latest_positions).to_csv(OUT/'latest_positions.csv',index=False)
    position_rows=[[r['market'],r['scope'],r['signal_date'],e(r['name']),e(r['code']),p(r['target_weight'])] for r in latest_positions]
    conclusions='''<h2>为什么原择时没有收益：实测诊断</h2><p>2022—2026沪深30012月动量的下一周时序IC为−0.043，标普为−0.065。它们与ETF横截面强弱IC不是同一件事，不能以“动量有效”直接推定市场择时有效。A股股票ETF12–1信号的下一周排序IC为0.083（HAC95%区间0.035～0.131），4周为0.111（0.031～0.192）；原多期限动量的4周IC仅0.019（−0.061～0.098）。</p><p>同股票池、同协方差、同成本下，12–1横向选券2019年以来CAGR13.49%，旧现金门槛额外加上后变成6.26%，最大回撤只从33.53%降到32.72%。股票择时开关削弱了选券收益，因此优先研究信号使用固定12–1横截面排名，停用未经证实的旧绝对门槛；只在明确标记的消融案例中保留。</p><p>预测未来排名的Ridge在A股4周平均IC为0.118，却仍没有胜过12–1的净收益，且年换手约28倍而12–1约15倍。模型选择看IC、尾部分组收益、扣费收益与分段稳定性；单一正IC不足以采用。IC机械启停也失败，A股4周组合门控只启用约7%的有效周，其活跃期IC为负，不能宣称已验证。美国4周模型IC区间跨零，保留拒绝采用状态。</p><p>以上区间是名义历史统计区间，未覆盖全部因子选择、多期限、两种池子与重复观察；统计正值不代表未来有95%成功概率。当前股票池具有幸存者和事后选池风险。</p>'''
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IC信号与多ETF横向轮动</title><style>body{margin:0;background:#f4f6f8;color:#23364b;font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1200px;margin:auto;padding:30px}.box{background:#fff;border:1px solid #d8e1e8;border-radius:12px;padding:22px}.note{background:#fff2dc;border-left:4px solid #b89035;padding:18px;margin:20px 0}h1{font-size:30px}h2{font-size:23px;margin-top:30px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}td,th{padding:10px;border-bottom:1px solid #dce1e7;white-space:nowrap;text-align:right}td:nth-child(4){text-align:left}img{width:100%}a{color:#126b91}code{background:#e8eef3;padding:3px}</style><main><div class="box"><h1>先改进信号，再讨论风险收益</h1><p>原双资产规则写死一只股票与一只债券，因此没有ETF横截面选择，也没有测出信号的横截面IC。新实现对全部当时可用且去重的ETF计算排名：仅股票池用于隔离股票选择能力；全池同时纳入债券财富指数、黄金、原油及股票，不设置股票总仓位或行业上限。候选集不是Top1，默认选择前1/3，完整协方差ERC配置多只资产。</p><p>新增10个预先列明的价格因子、IC收缩组合、直接预测未来排名的Ridge。1/4/13周预测期限各自检验，但所有调仓都保持周频。只使用已经结束的下一收盘成交→未来成交收盘收益标签，不能用全区间IC先挑赢家再从2017回填。</p></div><div class="note">正IC证明相对强弱，不证明市场整体会上涨。当前全部历史已观察，因子库也经历多次研究选择；本次因果逐步预测仍属于探索，不叫全新未使用样本外。门控只近似控制固定设置下两个模型的比较，不覆盖所有反复观察。IC不能直接年化成预期收益。</div><h2>2022—2026排序IC：正值、区间与覆盖一起看</h2>'''+table(['市场','范围','预测周数','信号','平均Rank IC','HAC95%下限','上限','IC有效周占比','停用周记0的有效IC'],icrows)+'''<p>IC是每周同池横截面Spearman相关，HAC处理至少4周及预测期重叠。恒定分数与缺失不强行计算相关；有效周占比显示停用程度，另列停用周记0的有效IC，避免只展示有利活跃期。绝不把平均IC或ICIR当作正确概率。</p><h2>最新股票信号构成：只是历史证据，不是年化收益预测</h2>'''+table(['市场','因子','成熟标签平均IC','HAC标准误','组合权重','证据截止'],latestrows)+'''<p>权重∝max(0，历史平均IC−1×HAC标准误)，至少104个成熟周。这是收缩规则，不是95%置信证明。随后组合和排名Ridge各自再检查至少52周已成熟的逐步预测IC；下限未转正时不启用增量，仓位退回同池无信号ERC。最新行情截至9月24日，末条周信号9月21日，不是10月当前建议。</p><h2>多ETF轮动实践</h2><img src="nav.png" alt="多ETF横向排序组合净值">'''+table(['市场','模型','CAGR','年化现金超额','年化波动','最大回撤','平均标的数','末日标的数'],mr)+'''<p>以上均从2019年开始、下一交易日收盘成交、真实持仓漂移、单边8bps。同池无信号ERC用于识别选券增量；原12月横向动量也使用相同新池，避免只把扩池效果当成新信号收益。全池ERC可能偏债，这是无信号基准，不是隐含股票上限。</p><div class="note">这些是信号隔离实验，未接入8%账户风险预算，不能称目标达标策略。cross_dual与skip_dual额外保留过去1/3/6/12月现金超额门槛，单独标记为尚未证明的择时消融。估值、财务、预期与份额历史时点未核验，未用于本轮训练；中债财富指数不可直接交易。</div><p><a href="ic_summary.csv">全部IC</a> · <a href="portfolio_metrics.csv">组合与分段指标</a> · <a href="latest_scores.csv">全池最新分数</a> · <a href="selection_coverage.csv">多标的覆盖</a> · <a href="method.json">流程与全部实验设定</a> · <a href="validation.json">时序与账本校验</a> · <a href="../ic_rotation.py">实现代码</a></p><p>研究动机：<a href="https://www.nber.org/papers/w7159">Jegadeesh与Titman动量持续性研究</a>；<a href="https://www.nber.org/papers/w2533">Lehmann短期反转研究</a>。股票论文结论不能直接当成ETF池的IC，以上均重新用本地Wind序列检验。</p></main></html>'''
    content=content.replace('<h2>2022—2026排序IC：正值、区间与覆盖一起看</h2>',conclusions+'<h2>2022—2026排序IC：正值、区间与覆盖一起看</h2>')
    content=content.replace('<h2>多ETF轮动实践</h2>', '<h2>9月21日实验目标仓位：已横向选择多只ETF</h2>'+table(['市场','池子','信号日期','名称','代码','目标占比'],position_rows)+'<p>仅为固定Wind快照的研究输出，未按8%预算调仓，不是10月当前仓位。前1/3是可调整的算法实验参数，不是用户给定上限。全池与股票池两种结果用于隔离风险溢价和选券能力，不表示规定股票总仓位。</p><h2>多ETF轮动实践</h2>')
    content=content.replace('<a href="selection_coverage.csv">多标的覆盖</a>', '<a href="selection_coverage.csv">多标的覆盖</a> · <a href="timing_ic.csv">原择时时序IC</a> · <a href="latest_positions.csv">多ETF实验目标仓位</a>')
    (OUT/'report.html').write_text(content)
    print(selected.loc[(selected.scope=='equity')&(selected.horizon_weeks==4),['market','signal','mean_rank_ic','hac95_low','hac95_high','active_fraction']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=['CN','US']);ap.add_argument('--select-fraction',type=float,default=1/3);ap.add_argument('--report-only',action='store_true');ap.add_argument('--portfolios-only',action='store_true');a=ap.parse_args()
    if a.report_only:build_report()
    else:
        if not 0<a.select_fraction<=1:raise ValueError('invalid selection fraction')
        if a.portfolios_only:add_portfolio_ablations(a.markets,ICSettings(select_fraction=a.select_fraction))
        else:run(a.markets,ICSettings(select_fraction=a.select_fraction));timing_diagnostics(a.markets)
