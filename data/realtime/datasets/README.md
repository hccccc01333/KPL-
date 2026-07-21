# 干净 / 隔离数据集说明

## 角色

| 路径 | 作用 |
|------|------|
| `data/realtime/raw_snapshots/` | **入库池**（监控/回填写入）。不删、不搬。 |
| `data/realtime/datasets/clean/` | **干净集**：默认可训练 / 诚实回测。 |
| `data/realtime/datasets/quarantine/` | **不干净存档**：保留原因，不进默认训练。 |
| `catalog.json` / `*_manifest.csv` | 全量分类清单与统计。 |

## 干净判定（须全部满足）

- 解析后快照数 `>= 5`
- `win_camp ∈ {1, 2}`（优先有终局 `status==2`）
- 唯一分钟数 `>= 8`
- 非仅浅层 backfill：至少 1 个非 `backfill_*.json` 的 live 文件
- `time_sec` 基本单调（无严重时钟回跳）

任一不满足 → `quarantine`，写入 `reasons[]`，**不删除**。

## 命令

```bash
# 只出报告
python scripts/classify_realtime_battles.py --dry-run

# 分类并复制到 clean/ 与 quarantine/
python scripts/classify_realtime_battles.py --materialize
```

训练 / 回测在 `clean/` 非空时**默认只读 clean**；需要全量入库池时加 `--from-raw`。

## 注意

- 新局先进 `raw_snapshots`，再跑分类「毕业」进 clean。
- 放宽规则后可再次 `--materialize`；会重建 clean/quarantine 子目录内容。
