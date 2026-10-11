# 独立量化研究程序

现在可以在 VS Code 或普通终端运行，不依赖 Codex、聊天或外部 AI API。保留既有研究算法，新增可安装 Python 包、统一配置、Wind 优先取数、五种组合算法及单资产流水线、目标验收、自动报告、日常刷新和幂等研究账本。

完整操作见 [VS Code 使用指南](docs/VS_CODE使用指南.md)。旧盘点保存在 [工程盘点](docs/工程盘点_20261007.md)。

## 新机器安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data]'
python project.py test
python -m riskparity doctor
```

GitHub 只保存核心代码、配置、测试和文档。行情数据、历史报告、参考仓库、运行账本和本机环境保留在本地，不随仓库发布。配置中的旧数据路径用于本机离线研究；新机器先执行取数：

```bash
python -m riskparity fetch --config configs/cn_erc.json --start 2015-01-01 --end 2026-10-09 --out data/cn_erc
python -m riskparity run --config data/cn_erc/job.json
```

日期按研究区间修改。先配置并登录官方 Wind API；Wind 失败时按项目约定记录原因并尝试备用接口。无核对过的同口径备用源时取数会失败。详见 [仓库管理](docs/仓库管理.md)。

## 本机开始

本机已建立根目录 `.venv`、安装程序和备用数据依赖，并配置官方 Wind SDK。打开本文件夹，打开终端：

```bash
source .venv/bin/activate
python -m riskparity doctor
python -m riskparity run --config configs/cn_erc.json
```

命令打印 `runs/` 内报告位置；用浏览器打开 `report.html`。也可以在 VS Code 的「终端 → 运行任务」选择对应任务。

```bash
# 五种中国组合算法：ERC、平层、两种分层、自适应
python -m riskparity batch --config configs/cn_compare.json
# 单资产ETF：训练、校准、回测、报告
python -m riskparity run --config configs/cn_single_etf.json
# 每日自动填日期，Wind主备取数后运行；需登录Wind
python -m riskparity daily --config configs/cn_erc.json
# 软件测试；不取行情、不下单
python project.py test
```

## 程序结构

- `src/riskparity/`：CLI、配置/来源校验、数据主备、算法适配、验收、报告、研究账本。
- `configs/`：独立可编辑的中美资产池、12份运行配置、2份批量配置。
- `outputs/asset_rotation/` 与 `outputs/single_asset/`：既有算法内核与历史研究记录。通过明确模块清单打包兼容内核，避免迁移破坏旧研究；新入口不读取固定实验日期或报告目录。
- `tests/`：流水线、主备顺序、拒绝逻辑、快照、重复运行及历史修订测试；原算法测试保留。
- `runs/`：每次独立运行，包含冻结数据/元信息、代码快照、哈希、环境、配置、日志、逐日记录、验收与报告；不覆盖旧目录。
- `data/`：新数据快照及主备尝试记录。
- `state/`：SQLite研究回放账本；不发送券商订单。
- `.vscode/`：解释器、运行任务、调试入口；`.github/workflows/`：自动测试配置。

## 安装与验证

Python 3.12。本机已安装；换机器的命令见使用指南。`requirements-lock.txt` 保存本机实际环境；`pyproject.toml` 提供标准安装与命令 `riskparity`。`dist/` 的 wheel 已在独立安装目录、从仓库外的工作目录跑通 ERC，证明安装包不依赖原脚本路径。

2026-10-07验收：27项工程测试、54项组合测试、14项单资产测试，共95项通过。组合短历史测试仍产生 sklearn 缺观测特征警告；未静默压制。中国五算法完整批量运行、美国ERC及单资产ETF运行完成；ERC、平层和单资产ETF的关键指标与旧结果一致。官方Wind的新组合行情、人民币现金及ETF日线快照已取得，并从新组合快照重新跑通。真实研究账本首次新增2348行，重复运行新增0行。详见 [工程验收](docs/工程验收.md)。

## 运行边界

`COMPLETED` 是程序成功；策略是否达标读取 `assessment.json`。本轮中国五算法仍没有达标组合，`selection.json` 的历史达标候选为空。软件工程化不会自动证明投资模型有效。

已有离线快照保留来源及截止日期。新取数先Wind，失败后自动尝试适合标的的备用源；整张组合价格表使用同一来源。中债财富指数和人民币指定现金序列没有核对过的同口径备用源时会明确失败，不偷偷替换产品或利率。美国现金Wind代码尚需用户核对并填入 `wind_cash_code`，失败后使用FRED DGS3MO并记录原因。

日报命令使用上海日期、以上一自然日为收盘截止，提供自然日新鲜度检查；不是完整交易所开闭市引擎。账本是历史研究回放。券商订单、真实成交对账、期货换月与保证金追加尚未接入，当前程序不自动下单。
