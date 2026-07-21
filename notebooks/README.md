# Notebooks 学习路线

> 严格按顺序做，每完成一个 notebook 末尾的「✅ 完成自检」全勾上，再开下一个。

| 顺序 | 文件 | 主题 | 产出 | 对应 src 模块 |
|------|------|------|------|--------------|
| 0 | `00_接口探索.ipynb` | 摸清 JSON 结构 + battle_id 编码 | `data/raw/sample.json`, `docs/数据字典.md` | （无，纯探索） |
| 1 | `01_单场爬虫.ipynb` | 写带重试/缓存的 fetch_battle | 几十个原始 JSON | `src/crawler/api_client.py` |
| 2 | `02_批量爬取.ipynb` | 写赛程爬虫 + 跑全季 | `data/processed/schedule.csv`, 100+ JSON | `src/crawler/schedule_crawler.py` |
| 3 | `03_数据清洗与建表.ipynb` | JSON → 3 张宽表 | `battles.csv` / `players.csv` / `bp.csv` | `src/parser/*.py` |
| 4 | `04_探索分析EDA.ipynb` | 8+ 张图 + 业务洞察 | `reports/figures/*.png` | （无） |
| 5 | `05_赛后建模V1.ipynb` | 无泄漏特征工程 + 4 模型 | `output/models/v1_baseline.joblib` | （后续抽到 src/model/） |
| 6 | `06_赛前建模V2_BP特征增强.ipynb` | BP 后可知特征 + V1/V2 对比 | `output/models/v2_bp_features.joblib` | （后续抽到 src/model/） |
| 7 | `07_赛中实时采集V3.ipynb` | 直播接口轮询 + 快照表设计 | `data/realtime/realtime_snapshots.csv` | `src/realtime/` |
| 8 | `08_实时胜率建模V4.ipynb` | 时间切片实时胜率模型 | `output/models/v4_realtime_model.joblib` | `src/features/`, `src/models/` |
| 9 | `09_Dashboard演示V5.ipynb` | Streamlit 演示页面设计 | `app/streamlit_app.py` | `app/` |

## 工作模式（与项目 01 保持一致）

```
你做（每个 notebook）：
  1. 按 TODO 顺序逐个 cell 写代码
  2. 卡住先在「业务问题」框里思考，写下你的猜测
  3. 完成所有 TODO 后跑通 → 把 notebook 路径甩给我
我做：
  1. 检查代码（bug / 写法 / 性能）
  2. 检查思路（业务理解 / 简历包装）
  3. 给反馈，你改完再来
```

## 三个铁律

1. **不要跳着做**——00 的字段字典是 03 的输入；02 的 schedule.csv 是 03 的输入。
2. **每个 notebook 结尾必须填总结报告**——这些就是简历项目的「成果素材」。
3. **遇到不懂的不要硬猜**——直接问我，比闷头瞎写效率高 10 倍。
