# KPL 官方赛事智能分析平台

> **项目定位**：面向 KPL 官方的实时赛事数据智能系统  
> **核心能力**：赛程自动监控、实时胜率预测、关键事件识别、战队/赛局复盘、数据健康中控  
> **技术栈**：Python + requests + pandas + scikit-learn + Plotly + Streamlit  
> **数据来源**：腾讯 KPL 官方比赛接口（赛程、比赛、对局、局内快照）

---

## 一、项目简介

本项目不是单一模型 Demo，而是一个端到端的 KPL 赛事智能分析平台：

```text
官方赛程 API
  -> 自动识别 match_id / battle_id
  -> 实时采集局内快照
  -> 统一 FeatureBuilder 生成线上特征
  -> V9 官方胜率模型预测
  -> Dashboard 实时中控 / 历史复盘 / 战队分析 / 数据健康
```

系统面向四类官方使用场景：

- **解说辅助**：实时回答“这波之后哪队赢面更大，为什么？”
- **导播支持**：在胜率大幅波动或关键资源刷新后切出数据面板。
- **赛后复盘**：定位每局关键转折点、资源控制、经济节奏和翻盘窗口。
- **数据运营**：生成战队对比、赛局战报、选手/位置表现分析素材。

---

## 二、最终版模块

| 模块 | 文件 | 说明 |
|---|---|---|
| 官方核心层 | `scripts/kpl_official_core.py` | 赛程中心、快照解析、统一特征、预测、复盘、健康报告 |
| 赛程归档 | `scripts/build_schedule_archive.py` | 跨赛季抓取/缓存赛程表，按日期定位 match_id/battle_id |
| 自动守护 | `scripts/auto_schedule_daemon.py` | 长期挂载：刷新赛程、等待开赛、检测对局、实时采集 |
| 历史回补 | `scripts/backfill_history.py` | 从赛程索引批量回补过去 battle 数据 |
| V11 模型训练 | `scripts/train_realtime_model_v9.py` | clean 真实优先 + 因果 FE + Early RF / Midlate Voting |
| V11 诚实回测 | `scripts/backtest_realtime_v9.py` | clean holdout 分阶段 Brier/ECE + gates（面试主指标） |
| 坏局诊断 | `scripts/diagnose_battle_v11.py` | 单局分钟时间线：金差 / 护栏 / 错分点 |
| 特征消融 | `scripts/ablate_v11_features.py` | 新 FE 组 zero-out + early RF 重训对比 |
| 自动监控采集 | `scripts/official_match_monitor.py` | 自动识别赛程/对局，保存 raw 快照、预测 CSV；日终异步重训 |
| 官方 Dashboard | `scripts/dashboard.py` | 赛事中心、实时中控、历史复盘、战队分析、数据面板 |
| 实时原始数据 | `data/realtime/raw_snapshots/` | `battle_id/timestamp.json` 粒度快照 |
| 实时预测输出 | `data/realtime/predictions/` | 每局预测时间线 CSV |
| 监控状态 | `data/realtime/monitor_status.json` | Dashboard 读取自动采集运行状态 |

---

## 三、核心特征体系

V11 在表层经济/击杀之上，使用**因果可解释**特征（训练与线上同一 `FeatureBuilder`）：

| 特征模块 | 代表特征 | 业务含义 |
|---|---|---|
| 基础优势 | `gold_ratio`、`kill_rate`、`tower_diff` | 当前局面谁领先 |
| 节奏动量 | `gold_diff_velocity`、`tempo_swing_score`、`gold_diff_accel` | 优势是否正在扩大 |
| 滚动窗口 | `gold_diff_roll4*`（消融证实 early 关键） | 近 4 分钟经济节奏 |
| 地图资源 | `objective_value_score`（暴君/先知/暗影/风暴/塔；弃用旧 lord） | 资源战略价值 |
| 位置对位 | `gold_diff_*`、`kill_diff_jungle/adc` | 核心位发育与击杀 |
| 核心发育 | `carry_dominance`、`hurt_conc_diff` | 双 C 与输出集中度 |
| 战队上下文 | `team_winrate_diff` | 开赛前因果胜率先验 |

这些特征既进入模型，也用于 Dashboard 的“为什么领先”解释面板。

## 四、运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练 V11（clean 真实 holdout 为主指标）
python scripts/train_realtime_model_v9.py

# 2b. 诚实回测 + 验收门禁
python scripts/backtest_realtime_v9.py --gates

# 2c. 坏局诊断 / 特征消融（可选）
python scripts/diagnose_battle_v11.py --battle-id 501236240_10_1783254441
python scripts/ablate_v11_features.py

# 3. 构建赛程索引
python scripts/build_schedule_archive.py --league-ids 20260002 --date 2026-05-23

# 从已有 processed/schedule.csv 快速构建
python scripts/build_schedule_archive.py --from-local

# 跨年份尝试抓取候选 league_id
python scripts/build_schedule_archive.py --start-year 2018 --end-year 2026

# 4. 启动自动赛程监控/采集
python scripts/official_match_monitor.py --interval 15

# 单次测试采集
python scripts/official_match_monitor.py --once
python scripts/official_match_monitor.py --date 2026-05-23 --once

# 指定某场比赛/某局
python scripts/official_match_monitor.py --match-id 2026052301 --once
python scripts/official_match_monitor.py --battle-id 1373651472_27_1779541129 --once

# 5. 长期挂载：自动刷新赛程并等待比赛开始
python scripts/auto_schedule_daemon.py --league-ids 20260002

# 只跑一轮调度逻辑，用于测试
python scripts/auto_schedule_daemon.py --league-ids 20260002 --once

# 6. 历史数据回补
python scripts/backfill_history.py --dry-run
python scripts/backfill_history.py --date 2026-05-23
python scripts/backfill_history.py --team AG --limit 20

# 7. 启动官方 Dashboard
streamlit run scripts/dashboard.py
```

Dashboard 默认地址：`http://localhost:8501`

## 五、Dashboard 页面

| 页面 | 面向用户 | 能力 |
|---|---|---|
| 赛事中心 | 官方运营/数据团队 | 查看官方赛程、比赛 ID、本地已采集 BO 赛局 |
| 实时中控 | 解说/导播 | 自动识别 battle_id，展示实时胜率、经济/击杀/推塔、领先原因 |
| 历史复盘 | 教练/分析师 | 按赛段、战队、赛局下钻，查看胜率曲线与关键事件 |
| 战队分析 | 数据运营/内容团队 | 战队胜率、经济差、推塔差、雷达图、逐局趋势 |
| 数据面板 | 数据工程/维护人员 | 模型版本、快照数、监控状态、缺失时间线、数据目录 |

## 六、数据与模型验证

当前评估口径（面试请用这套，不要只报混合 AUC）：

- **主指标**：clean 真实局 holdout 的 early(≤8) / mid(9–15) / late(≥16) **Brier / ECE / 方向正确率**
- **次要参考**：混合测试集 AUC（含降权后的终局模拟样本）
- **数据分层**：`raw_snapshots` 入库池 → `datasets/clean|quarantine`；训练/回测默认 clean
- **线上兜底**：金差一致性护栏（5–7′ 轻度 / ≥8′ 加强），防止「经济明显落后却报高胜率」
- 回测：`python scripts/backtest_realtime_v9.py --gates` → `data/realtime/backtest_v9_report.json`

**当前 holdout 数字（V11 + soft early gold guard，2026-07-20）**：

| 阶段 | Brier | Acc | 备注 |
|------|-------|-----|------|
| early | **0.057** | 0.94 | 护栏修复后 early 明显下降 |
| mid | **0.036** | 0.96 | |
| all | **0.049** | 0.94 | gates PASSED |
| 特例坏局 | Acc≥0.86 | | `970998288_6` / 碾压局可诊断 |

模型输出：

- Artifact：`output/models/v9_official_platform.joblib`（`version=V11`）
- Dashboard / 监控统一：`load_model()` / `predict_probability()`
- 面试讲解：见 [`docs/面试Demo脚本.md`](docs/面试Demo脚本.md)

## 七、简历表达

> 构建 KPL 官方赛事智能分析平台，逆向接入腾讯赛事接口，实现从赛程自动识别、实时局内快照采集、统一特征工程、胜率预测建模到可视化复盘的端到端闭环。系统支持实时中控、历史复盘、战队分析与数据健康监控；以 clean holdout 分阶段 Brier 为主指标，并用金差一致性护栏修复「和经济拧巴」的坏局模式。

## 八、面试讲法

完整口播见 [`docs/面试Demo脚本.md`](docs/面试Demo脚本.md)。摘要：

**业务价值**：不是只预测胜负，而是帮助解说/导播理解走势与转折。  
**技术难点**：API 副本不一致、早期噪声、小样本、模拟样本虚高 AUC。  
**解决方案**：统一 FeatureBuilder、clean 分层、分阶段模型+校准、金差护栏、坏局可诊断。  
**项目亮点**：实时链路 + 诚实评估 + 可讲清的坏局修复故事。

## 九、历史目录结构

```
02-KPL实时胜率预测系统/
├── README.md                          # 项目说明（本文件）
├── requirements.txt                   # 依赖清单
├── .gitignore                         # 排除大数据/venv
│
├── docs/                              # 项目文档
│   ├── 项目设计.md                    # 整体架构与思路
│   ├── 接口逆向笔记.md                # 腾讯接口规律存档
│   ├── 数据字典.md                    # 字段说明
│   ├── 思路任务书.md                  ⭐ 重点：每个步骤做什么、为什么
│   └── 技术学习路线.md                ⭐ 开工前看：要学什么、学到什么程度、推荐资源
│
├── data/
│   ├── raw/                           # 原始 JSON（不上 GitHub）
│   ├── processed/                     # 清洗后的 CSV
│   ├── realtime/                      # 直播采集的赛中快照
│   └── README.md
│
├── src/                               # 工程化代码
│   ├── config.py                      # URL/Header/路径常量
│   ├── crawler/                       # 爬虫模块
│   │   ├── api_client.py              # 接口调用基础类
│   │   └── battle_crawler.py          # 比赛数据爬虫
│   ├── parser/                        # 解析模块
│   │   └── battle_parser.py           # JSON → DataFrame
│   ├── features/                      # 特征工程模块
│   ├── models/                        # 训练与评估模块
│   ├── realtime/                      # 实时采集与快照解析
│   └── utils/
│       └── logger.py                  # 日志
│
├── app/
│   └── streamlit_app.py               # Dashboard 演示入口
│
├── notebooks/                         # 探索 + 分析
│   ├── 00_接口探索.ipynb              # 摸清 JSON 结构
│   ├── 01_单场爬虫.ipynb              # 单场接口请求
│   ├── 02_批量爬取.ipynb              # 赛程与批量 battle 爬取
│   ├── 03_数据清洗与建表.ipynb        # JSON → CSV 宽表
│   ├── 04_探索分析EDA.ipynb           # 业务 EDA
│   ├── 05_赛后建模V1.ipynb            # 赛前基线模型
│   ├── 06_赛前建模V2_BP特征增强.ipynb # BP 后预测
│   ├── 07_赛中实时采集V3.ipynb        # 实时快照采集
│   ├── 08_实时胜率建模V4.ipynb        # 时间切片实时模型
│   └── 09_Dashboard演示V5.ipynb       # Streamlit 演示设计
│
├── scripts/                           # 一次性脚本
│   └── 01_test_api.py                 # 接口连通测试（已验证 ✅）
│
└── reports/
    └── figures/                       # 输出图表
```

---

## 十、技术栈

| 模块 | 工具 |
|---|---|
| 爬虫 | `requests` |
| 数据处理 | `pandas`、`numpy` |
| 可视化 | `matplotlib`、`seaborn`、`plotly` |
| 建模 | `scikit-learn`、`xgboost` |
| Dashboard（V5） | `streamlit` |
| 日志 | `logging`（标准库） |

---

## 十一、与老项目（D:\\pythonProject5）的对比

| 维度 | 老项目 | 本项目（V1-V5） |
|---|---|---|
| 数据时效 | 2024 决赛（陈旧） | 2026 春季赛（最新） |
| 数据规模 | ~10 场决赛 | 春季赛全季 100+ 场 |
| 工程化 | 7 个散乱脚本 | 模块化 src/ 结构 |
| 特征工程 | 全部字段（含数据泄漏） | 严格区分赛前可知特征 |
| 预测维度 | 胜负二分类 | 赛前胜负 + BP 后胜率 + 实时胜率曲线 |
| 模型评估 | 单次切分 | 时间切分 + baseline 对比 + 概率评估 |
| 可演示 | 命令行打印 | Streamlit Dashboard |

---

## 十二、最终状态

- [x] 赛程中心：自动获取比赛时间、状态、`match_id`
- [x] 对局中心：自动识别当前 `battle_id`
- [x] 实时采集：保存 raw snapshots
- [x] 统一特征：训练、实时、Dashboard 共用 FeatureBuilder
- [x] V9 官方模型：解释型复合特征 + 真实赛中快照
- [x] 自动监控：输出预测 CSV 和 `monitor_status.json`
- [x] Dashboard：五大官方模块
- [x] 数据健康：快照质量、缺失局次、监控状态

## 十三、致谢

- 数据来源：腾讯 KPL 官方接口（`prod.comp.smoba.qq.com`）
- 项目灵感：作者 2024 年 KPL 决赛预测项目（D:\\pythonProject5）的全面升级版
