# 不接入 AI，在 VS Code 使用

本程序不调用任何大模型、AI 插件或 AI API。不需要 Codex、Copilot 或聊天窗口。机器学习预测使用本地 scikit-learn，按固定规则训练；它不需要外部 AI 服务。

## 本机运行

在 VS Code 选择「文件 → 打开文件夹」，打开本项目根目录。选择 `.venv/bin/python` 为 Python 解释器。打开「终端 → 新建终端」：

```bash
source .venv/bin/activate
python -m riskparity doctor
python -m riskparity run --config configs/cn_erc.json
```

每次运行会自动新建 `runs/时间戳-随机ID/`，终端打印报告位置。用浏览器打开其中的 `report.html`；VS Code 内可直接打开 CSV 和 JSON。重复运行不会覆盖旧结果。

也可以通过「终端 → 运行任务」选择「运行中国风险平价」「比较中国五种组合算法」「运行单资产ETF模型」「运行全部测试」。F5 的「调试组合程序」可以设置断点，包括子进程。

## 新机器首次安装

Python 需要 3.12 或以上，推荐 3.12；Wind SDK 单独配置。macOS / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e '.[data,dev]'
python -m riskparity doctor
python project.py test
```

`requirements-lock.txt` 是既有研究环境的固定版本基线；如其他平台无法安装其中版本，可直接安装 `.[data,dev]` 解析依赖后运行全部测试，保留新环境锁文件再正式复现。当前不宣称所有平台已验证。

Windows：用 `py -3.12 -m venv .venv` 创建环境，再用 `.venv\Scripts\python.exe -m pip install -e ".[data,dev]"`。后面的 `python` 可以替换为 `.venv\Scripts\python.exe`；VS Code 的任务已配置 Windows 路径，解释器需选择 `.venv\Scripts\python.exe`。

## 配置与算法

| 配置 | 功能 |
| --- | --- |
| `configs/cn_erc.json` / `us_erc.json` | 三资产纯风险平价基准 |
| `configs/cn_flat.json` / `us_flat.json` | 全池平层优化 |
| `configs/cn_nested_target.json` / `us_nested_target.json` | 分层目标配置 |
| `configs/cn_protective_nested.json` / `us_protective_nested.json` | 分层防御配置 |
| `configs/cn_adaptive.json` / `us_adaptive.json` | 自动逐期预测、初始校准、自适应配置 |
| `configs/cn_single_etf.json` / `cn_single_stock.json` | 单资产趋势与概率模型 |
| `configs/cn_compare.json` / `us_compare.json` | 同一市场五种组合算法批量运行与对比 |

```bash
python -m riskparity validate --config configs/cn_flat.json
python -m riskparity run --config configs/cn_flat.json
python -m riskparity run --config configs/cn_single_etf.json
python -m riskparity batch --config configs/cn_compare.json
```

批量结果有 `index.html`、`comparison.csv`、`selection.json`。某个算法失败时仍会运行其余算法，并将该批次标为 `PARTIAL_FAILURE`。自适应模型需要逐期训练及优化，运行时间较长，进度在该运行目录的 `run.log`。

改参数只编辑 JSON。路径相对配置文件的位置解释，不依赖旧工作目录。组合回撤目标为 `max_drawdown`，例如 `0.08`；`excess_return_multiple: 2` 表示年化收益减同币种现金年化收益至少 16 个百分点。每种算法的 `parameters` 对应其既有设置字段；不支持的字段会报错。ERC 仅支持交易费用和滑点参数；其回撤/收益目标是事后验收，不是保证。

`start` 和 `end` 可以限制输入区间，`start` 含预热段；组合统一需要至少 507 行。ERC、平层、分层及自适应组合正式评价从第 505 行开始。预热数据不能删掉后期待在同一天开始评价。

默认配置使用已有 Wind 导出，组合截至 2026-09-24，单资产截至 2026-09-30。这些是历史研究快照。设置 `max_price_age_days` 和 `as_of` 可要求数据新鲜度；`as_of` 不填时使用上海当前日期。现金有单独的历史过期检查。

## 更新行情：Wind 优先，失败后备用源

先登录官方 Wind API 应用。首次连接 SDK：

```bash
python outputs/asset_rotation/setup_wind.py
python -m riskparity fetch --config configs/cn_erc.json --start 2015-01-01 --end 2026-10-07 --out data/cn_20261007
python -m riskparity run --config data/cn_20261007/job.json
```

以后修改 `--end` 为所需的已完成交易日，每次使用新输出目录。指定日期不等于接口必然有当日收盘数据。日志、每个接口的成功/失败原因、实际区间、时间、复权和成交量口径都在快照目录。整张组合价格表采用一个来源；失败的部分数据会丢弃，禁止无记录混拼。

境内股票/ETF 尝试 AKShare，再尝试 Yahoo；美国 ETF 尝试 Yahoo。中债财富指数没有验证过的备用映射时会明确停止，不用债券 ETF 偷换指数。人民币 AAA 一年期同业存单现金序列同样不以其他利率替代。美元现金先尝试 Wind，需在配置的 `wind_cash_code` 填写核对过的 EDB 代码；未配置或 Wind 失败会记录原因并尝试官方 FRED DGS3MO。此处默认未编造 Wind 代码。

也可以把 `start`、`end` 写进配置后，一条命令完成新快照与回测：

```bash
python -m riskparity run --config configs/my_cn_run.json --refresh
```

先复制一个现有配置为 `my_cn_run.json` 并填入日期。所有来源失败时返回错误，新快照为 `FAILED`；已有离线数据不会被覆盖，也不会被当作本次新行情继续运行。

## 输出和错误怎么读

| 文件 | 内容 |
| --- | --- |
| `run.json` | 运行状态、冻结配置、数据及元信息哈希、完整运行代码哈希、Python 与依赖版本 |
| `run.log` | 模型进度、警告和执行日志 |
| `metrics.json` | 该次回测的指标 |
| `assessment.json` | 目标验收与拒绝原因，及研究/部署状态 |
| `curve.csv` / `equity.csv` | 每日净值，单资产用后者 |
| `holdings.csv`、`signals.csv` | 组合实际漂移仓位与收盘目标；单资产用 decisions/trades |
| `latest_signal.json` | 最新研究信号及有效状态；不等于券商订单 |
| `report.html` | 仅从该运行目录的记录自动生成的报告 |
| `error.json` | 失败的固定错误码与原因 |

`COMPLETED` 表示软件流程完成；`GOALS_FAILED` 表示组合未达标；`HISTORICAL_GOALS_PASSED` 仅表示该历史样本达到目标，仍须纸面交易验证。单资产沿用 `NOT_VALIDATED` / `REQUIRES_PAPER_TRADING`。程序不会自动选择一个不达标版本或放松约束。

命令退出码：0 为软件流程完成，包括已明确输出的策略目标拒绝；2 为单次运行/配置/数据错误；批量返回 1 表示至少一个任务运行失败；130 为用户中断。自动任务可以据此判断是否发生程序故障，再读取 assessment 判断策略是否合格。

## 可重复执行的研究账本

```bash
python -m riskparity replay --run runs/某次完成的运行 --account cn_erc_v1
```

默认保存到 `state/research.sqlite`。相同运行再执行一次，新增记录为零；不同策略、代码或参数须用新的 account 名称；历史记录发生修订则拒绝追加并回滚。这是历史回放账本，不是假装已经实现了券商订单系统。

## 自动运行，不用 AI

下面的命令会自动使用上海当天日期、以上一个自然日为收盘数据截止，取数、校验、回测和生成报告，不用每天手改日期：

```bash
python -m riskparity daily --config configs/cn_erc.json
```

结果在 `runs/daily/日期-配置名/`；同一天同配置完成后重复调用会返回 `ALREADY_COMPLETED`，不重复运行。配置改变或上一尝试未完成时会拒绝重复写入。默认允许最近价格距上海当天不超过15个自然日，兼顾长假；可用 `--max-age-days 3` 收紧，但长假也会触发拒绝。这是自然日新鲜度检查，不冒充完整市场开闭市或夜盘日历。

此命令可直接交给 Windows 任务计划程序、macOS launchd 或你的现有调度系统，使用解释器、配置和 `--out-root` 的绝对路径。程序无需聊天会话。未在本轮擅自设置持续取数任务。

现有程序已覆盖离线研究和主备取数。实际券商下单、成交回报、涨跌停队列、期货换月与保证金追加仍需对应账户和市场接口，当前入口不宣称提供这些能力。
