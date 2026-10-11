# Wind 独立因子库

只取数据、计算因子和导出；没有策略、交易信号或回测。默认示例池为贵州茅台、美的集团、招商银行，基准为沪深300；不是推荐组合。日频默认最近183个自然日，5分钟数据默认最近10个自然日。股票池可自行更换，当前备用接口限境内股票。

## 运行

在已配置官方 WindPy 的研究环境中，从项目根目录运行：

```bash
python -m riskparity.factor_library --codes 600519.SH,000333.SZ,600036.SH --start 2026-04-07 --end 2026-10-06 --minute-days 10 --out outputs/factor_library/my_run
```

输出目录必须是新目录。`--minute-days 0` 关闭分钟取数。结束日可以是非交易日，实际数据终点见 manifest。

- `factors.csv`：日期×标的的原始因子值。
- `latest.csv`：每个标的最后一个交易日的因子，不会用旧非空值冒充当天值。
- `cross_section_ranks.csv`：当日在指定股票池中的百分位排名，至少3个非空标的；没有统一多空方向。
- `coverage.csv`：每个标的、因子的有效样本数。
- `raw/`：原始取数文件。
- `manifest.json`：来源、请求范围、实际覆盖、取数时间、选项、失败原因、文件哈希和使用限制。

## 因子字典

r=日对数收益；C/O/H/L=一致前复权价格；A=成交金额；M=基准价格；所有窗口为交易日，波动率年化用252。

| 类别 | 列名 | 计算 |
|---|---|---|
| 波动率 | vol_5 / vol_20 / vol_60 | 日收益样本标准差×√252 |
| 波动率 | downside_vol_20 | √(252×20日平均 min(r,0)²) |
| 波动率 | parkinson_vol_20 | √(252×mean(log(H/L)²)/(4log2)) |
| 波动率 | vol_ratio_20_60 | vol_20 / vol_60 |
| 波动率 | vol_of_vol_20 | vol_20 的20日样本标准差 |
| 波动率 | atr_pct_20 | max(H-L, abs(H-C前日), abs(L-C前日)) 的20日均值/C |
| 流动性 | amount_20 / log_amount_20 | 20日平均成交额及其自然对数 |
| 流动性 | amihud_20 | 20日平均 abs(r)/A；越大表示越不流动 |
| 流动性 | turnover_20 | Wind turn百分数除100后，20日均值 |
| 流动性 | amount_ratio_20 | 当日A / 此前20日平均A，不包含当天 |
| 流动性 | zero_volume_fraction_20 | 20日零成交量天数占比 |
| 收益 | momentum_5 / momentum_20 / momentum_60 | C/C滞后N日 - 1 |
| 收益 | reversal_5 | -momentum_5 |
| 收益 | momentum_60_skip_5 | C滞后5日/C滞后60日 - 1 |
| 收益 | overnight_return / intraday_return | O/C前日-1；C/O-1 |
| 联动 | beta_60 / correlation_60 | 60日与基准对数收益的beta、相关系数 |
| 联动 | idiosyncratic_vol_60 | 60日带截距市场回归的残差标准差×√252，样本分母59 |
| 联动 | relative_momentum_20 | 标的20日动量 - 基准20日动量 |
| 基本面 | earnings_yield / book_to_price | 1/pe_ttm，1/pb_lf；负值保留，零值缺失 |
| 基本面 | roe / profit_growth / revenue_growth | Wind ROE、利润同比、营收同比，百分数除100 |
| 基本面 | gross_margin / debt_to_assets | 毛利率、资产负债率，百分数除100 |
| 高频低频化 | realized_variance / realized_vol | 每个5分钟bar的 log(close/open)² 求和及开根号 |
| 高频低频化 | downside_realized_variance | 只计负bar收益的平方和 |
| 高频低频化 | tail_amount_fraction | 14:30以后最后6个bar成交额/全天成交额 |
| 高频低频化 | amount_concentration | sum((bar成交额/日成交额)²) |

5分钟日内指标不含隔夜与午休跳变；采用每个bar开盘到收盘收益，而非跨bar的收盘收益。A股必须完整48个bar，否则该日高频因子置空。`bar_count` 是覆盖诊断，不是投资因子。

另外提供 `quote_factors()`，接受等间隔 Wind 盘口导出，列为 bid、ask、bid_depth、ask_depth（深度以人民币计），生成相对价差、总深度、买卖深度不平衡的日均值。当前CLI不自动拉取历史盘口；5分钟OHLC不能还原价差、盘口深度或主动买卖分类。

## 数据与时点

优先官方 Wind。境内股票日频或5分钟数据失败时，自动尝试 AKShare；成功时整段切换 AKShare 前复权OHLC和未复权成交量/额，不拼接不同来源。不可取得的财务及盘口数据保留缺失和原因；不从当前财报快照倒填历史。Wind成交量保留原生单位，不据此推导流通股数；Amihud用成交额，换手率直接用Wind turn。

财务历史字段没有独立核验公告时点和修订版本，标注 `pit_verified=false`，仅用于描述研究，不能直接宣称可交易历史因子。有的字段只在报告日期返回，代码不自动前向填充。银行毛利率可能不适用。

所有因子日期是观测日，日频完整数据在当天收盘后才知道；回测须匹配下一可交易时点，不能同日收盘成交。缺失窗口不补零。短区间未做252日动量，使用60日版本。

五项测试覆盖公式、市场回归、历史值不受未来数据影响、停牌处理、分钟完整性、盘口计算和异常OHLC校验。测试数据为合成数据，实际导出为Wind数据，两者独立。

## 本次实测结果

结果目录：`outputs/factor_library/wind_20261007/`。请求2026-04-07至2026-10-06，实际交易日期2026-04-07至2026-09-30；三只股票各122个交易日，共366行、37个因子，另有bar_count诊断列。本次成功查询来源全部为Wind。

分钟数据覆盖2026-09-28至2026-09-30的3个完整交易日。招商银行毛利率无有效数据，留空；利润同比、营收同比、负债率在区间内各只有1个报告日值，未倒填或向后填充。短窗口因子在预热完成前缺失。5项单元测试通过，覆盖数据样本仅用于软件测试。

## 扩大股票池实测（2026-10-07）

以2026-09-30为筛选日，Wind全A股5572家公司中970家总市值>200亿元，314家同时满足2026-08-19～2026-09-30的30个交易日日均成交额>10亿元。日频分析区间为2026-04-07～2026-09-30；预热从2026-01-01开始；分钟取2026-08-03～2026-09-30。

最终数据：`outputs/factor_library/ashare_calendar_20261007/factors.csv`，314家公司、37924行、37因子。254家日频来自Wind，60家因Wind最低价批次失败等原因整段使用AKShare/新浪；筛选市值与成交额、历史市值、行业和全部分钟数据来自Wind。Wind基准历史复用既有导出，无法补齐预热区间，60日联动因子有效检验区间较短。按公共SSE日历计算窗口与未来收益，接口缺失日期不跳过、不补零；较短IPO历史保留，长窗口因子留空。

最终报告：`outputs/factor_library/ashare_final_analysis_20261007/report.html`。37个因子中30个做收益检验；7个基本面缺完整历史可用时点或受Wind额度影响，暂不做有效性排名。收益标签从下一交易日开盘到第5/20交易日收盘；五组收益差为重叠持有期描述统计，不是策略累计绩效。已核验全部314家未来收益标签与原始数据的交易日对齐及精确排名恒等关系；9项单元测试通过。

新增代码：`src/riskparity/factor_universe_run.py` 批量取数；`src/riskparity/factor_universe_diagnostics.py` 大池检验；`work/screen_factor_universe.py` 筛选；`work/repair_factor_daily.py` / `work/repair_factor_minutes.py` 补取；`work/assemble_factor_universe.py` 合并与日历对齐；`work/build_universe_factor_report.py` 结果报告。这一轮脚本日期和目录为固定快照，重新运行应先指定或修改新输出目录，已有原始数据及报告不覆盖。

期末筛选池含选择和幸存者偏差，历史当天条件检查也只在该固定池内完成；控制项使用期末申万2021行业快照，未经历史行业变动核验，仅为敏感性诊断。HAC/FDR是探索性统计，不替代独立区间验证。
