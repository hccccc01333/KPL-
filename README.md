# KPL 赛事数据智能平台

> **定位**：面向 KPL 赛事的实时数据分析系统（可演进为「KPL 数据中心」）  
> **能力**：赛程监控 · 实时胜率预测 · 关键事件识别 · 战队/赛局复盘 · 数据健康中控  
> **技术**：Python · pandas · scikit-learn · Plotly · Streamlit  
> **数据**：腾讯 KPL 公开赛事接口（赛程 / 对局 / 局内快照）

---

## 项目简介

端到端的赛事智能分析链路：

```text
官方赛程 API
  → 自动识别 match_id / battle_id
  → 实时采集局内快照
  → 统一 FeatureBuilder 生成特征
  → V11 胜率模型预测
  → Dashboard：实时中控 / 历史复盘 / 战队分析 / 数据健康
```

适用场景：

- **解说辅助**：当前哪队赢面更大，主要依据是什么  
- **导播支持**：胜率大幅波动或关键资源节点时的数据面板  
- **赛后复盘**：转折点、资源控制、经济节奏、翻盘窗口  
- **数据运营**：战队对比、赛局战报、位置表现素材  

---

## 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 官方核心层 | `scripts/kpl_official_core.py` | 赛程、快照解析、特征、预测、复盘、健康报告 |
| 赛程归档 | `scripts/build_schedule_archive.py` | 跨赛季赛程索引 |
| 自动守护 | `scripts/auto_schedule_daemon.py` | 刷新赛程、等待开赛、实时采集 |
| 历史回补 | `scripts/backfill_history.py` | 按赛程批量回补对局 |
| 模型训练 | `scripts/train_realtime_model_v9.py` | clean 真实优先 + 因果特征 + 分阶段模型 |
| 诚实回测 | `scripts/backtest_realtime_v9.py` | holdout 分阶段 Brier / ECE + 验收门禁 |
| 自动监控 | `scripts/official_match_monitor.py` | 识别对局、落盘快照与预测 |
| Dashboard | `scripts/dashboard.py` | 赛事中心 / 实时中控 / 复盘 / 战队 / 数据面板 |

---

## 特征体系

训练与线上共用同一套 `FeatureBuilder`（因果、可解释）：

| 模块 | 代表特征 | 含义 |
|------|----------|------|
| 基础优势 | `gold_ratio`、`kill_rate`、`tower_diff` | 当前谁领先 |
| 节奏动量 | `tempo_swing_score`、`gold_diff_accel` | 优势是否扩大 |
| 滚动窗口 | `gold_diff_roll4*` | 近窗经济节奏 |
| 地图资源 | `objective_value_score` | 暴君 / 先知 / 暗影 / 风暴 / 塔 |
| 位置对位 | `gold_diff_*`、`kill_diff_jungle/adc` | 核心位表现 |
| 核心发育 | `carry_dominance`、`hurt_conc_diff` | 双 C 与输出结构 |
| 战队先验 | `team_winrate_diff` | 开赛前历史胜率（无泄漏） |

---

## 快速运行

```bash
pip install -r requirements.txt

# 训练 / 回测
python scripts/train_realtime_model_v9.py
python scripts/backtest_realtime_v9.py --gates

# 监控采集
python scripts/official_match_monitor.py --interval 15

# Dashboard
streamlit run scripts/dashboard.py
```

Dashboard 默认：`http://localhost:8501`

---

## Dashboard

| 页面 | 能力 |
|------|------|
| 赛事中心 | 赛程、比赛 ID、本地已采集 BO |
| 实时中控 | 胜率曲线、经济/击杀/推塔、领先原因 |
| 历史复盘 | 按赛段/战队/赛局下钻 |
| 战队分析 | 胜率、经济差、推塔差、雷达与趋势 |
| 数据面板 | 模型版本、快照质量、监控状态 |

---

## 模型与评估

- **主指标**：clean 真实局 holdout 的 early / mid / late **Brier、ECE、方向正确率**（不以混合 AUC 为主）
- **数据分层**：`raw_snapshots` 入库 → `datasets/clean|quarantine`；训练与回测默认 clean
- **线上兜底**：金差一致性护栏，避免「经济明显落后却报高胜率」

**Holdout 结果（V11，2026-07-20）**

| 阶段 | Brier | Acc |
|------|-------|-----|
| early (≤8′) | 0.057 | 0.94 |
| mid (9–15′) | 0.036 | 0.96 |
| all | 0.049 | 0.94 |

模型产物：`output/models/v9_official_platform.joblib`（本地训练生成，仓库不收录大文件）

---

## 技术栈

| 模块 | 工具 |
|------|------|
| 接口与采集 | `requests` |
| 数据处理 | `pandas`、`numpy` |
| 建模 | `scikit-learn` |
| 可视化 | `plotly`、`matplotlib` |
| 产品界面 | `streamlit` |

---

## 仓库说明

- 大体积快照、模型权重已由 `.gitignore` 排除，仓库以**代码 + 文档 + 指标样例**为主  
- 本地采集数据后即可完整复现训练、回测与 Dashboard  

---

## 版本记录

当前线上模型：**V11**（artifact：`v9_official_platform.joblib`）

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| **V11** | 2026-07-20 | clean 数据集分层；因果特征（roll/动量/分路等）；Early RF + Midlate Voting；特征消融与模型选型；5–7′ 轻度金差护栏；holdout early Brier ≈0.057 / all ≈0.049，gates 通过 |
| **V10** | 2026-07 | 坏局诊断与金差一致性护栏（≥8′）；拐点/战报；midlate 校准择优；特殊局验收门禁 |
| **V9** | 2026-07 | 官方平台统一 FeatureBuilder；真实优先训练 + 因果战队先验；分阶段 early/midlate 混合预测 |
| **V8** | 2026-06 | 增强型实时特征与 stacking/集成探索 |
| **V7** | 2026-06 | 实时 stacking 与校准迭代 |
| **V6** | 2026-06 | 实时校准版投票/融合基线 |
| **V5** | 2026-05 | Streamlit Dashboard 演示闭环 |
| **V4** | 2026-05 | 赛中时间切片实时胜率建模 |
| **V3** | 2026-05 | 赛中快照采集链路 |
| **V2** | 2026-05 | BP 后赛前特征增强 |
| **V1** | 2026-05 | 赛后/赛前基线建模与接口探索 |

## 致谢

数据来源：腾讯 KPL 公开赛事接口（`prod.comp.smoba.qq.com`）
