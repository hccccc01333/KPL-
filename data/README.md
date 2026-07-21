# Data 目录说明

本目录存放项目所有数据文件，**不上传 GitHub**（见 `.gitignore`）。

## 子目录

```
data/
├── raw/         # 原始爬虫 JSON（每场比赛一个文件）
├── processed/   # 清洗后的 CSV（battles / players / features）
└── README.md    # 本文件
```

## 命名规范

### raw/
- `{battle_id}.json` —— 单场比赛原始数据
- 例：`1038107152_18_1742644777.json`

### processed/
- `battles.csv` —— 每场比赛一行
- `players.csv` —— 每场 10 行（每个选手一行）
- `features.csv` —— 模型用的特征矩阵
- `crawl_summary.csv` —— 爬虫记录（哪些 battle_id 爬过、成功 / 失败）

## 数据获取方式

```bash
# 1. 验证接口可用
python scripts/01api.py

# 2. 批量爬取（待开发）
python scripts/02_crawl_2026_spring.py

# 3. 解析为 CSV（待开发）
python scripts/03_parse_raw_to_csv.py
```

## 数据来源声明

数据来自腾讯王者荣耀职业联赛官方公开接口，**仅用于个人学习项目**，不做商业用途、不传播。
