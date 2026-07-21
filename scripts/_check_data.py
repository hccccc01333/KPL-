import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from pathlib import Path

rt = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime')
print('=== 实时数据文件 ===')

csv = rt / 'live_snapshots.csv'
if csv.exists():
    df = pd.read_csv(csv)
    print(f'live_snapshots.csv: {len(df)} 行')
    bids = df["battle_id"].unique().tolist()
    print(f'  battle_ids: {bids}')
    print(f'  列数: {len(df.columns)}')
else:
    print('live_snapshots.csv 不存在')

pred_dir = rt / 'predictions'
if pred_dir.exists():
    for f in sorted(pred_dir.glob('*.csv')):
        df2 = pd.read_csv(f)
        print(f'\npredictions/{f.name}: {len(df2)} 行')
        if 'prob_camp1' in df2.columns:
            print(f'  prob_camp1 范围: [{df2["prob_camp1"].min():.3f}, {df2["prob_camp1"].max():.3f}]')
        if 'win_camp' in df2.columns:
            print(f'  最终获胜方: camp{int(df2["win_camp"].iloc[-1])}')

raw_dir = rt / 'raw_snapshots'
if raw_dir.exists():
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and d.name != '.gitkeep':
            jsons = list(d.glob('*.json'))
            print(f'\nraw_snapshots/{d.name}/: {len(jsons)} 个 JSON')
