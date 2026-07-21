"""量化采集延迟：墙钟时间 vs 游戏内时间"""
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from pathlib import Path
from datetime import datetime

pred_dir = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\predictions')
files = sorted(pred_dir.glob('*.csv'))
print(f"找到 {len(files)} 个预测 CSV")

for f in files[-3:]:  # 最近 3 局
    df = pd.read_csv(f, on_bad_lines='skip', engine='python')
    if len(df) < 5:
        continue
    print(f"\n=== {f.name} ===")
    print(f"  共 {len(df)} 条")

    df['t_wall'] = pd.to_datetime(df['collected_at'])

    time_col = 'snapshot_time_sec' if 'snapshot_time_sec' in df.columns else 'minute'
    if time_col == 'snapshot_time_sec':
        df['_game_sec'] = pd.to_numeric(df['snapshot_time_sec'], errors='coerce')
    else:
        df['_game_sec'] = pd.to_numeric(df['minute'], errors='coerce') * 60

    start_rows = df[df['_game_sec'] > 0]
    if len(start_rows) == 0:
        continue
    first = start_rows.iloc[0]
    t0_wall = first['t_wall']
    g0 = first['_game_sec']
    print(f"  锚点: 墙钟={t0_wall.strftime('%H:%M:%S')}, game={g0}s")

    df_clean = df[df['_game_sec'] > 0].copy()
    df_clean['wall_elapsed_s'] = (df_clean['t_wall'] - t0_wall).dt.total_seconds()
    df_clean['game_elapsed_s'] = df_clean['_game_sec'] - g0
    df_clean['lag_s'] = df_clean['wall_elapsed_s'] - df_clean['game_elapsed_s']

    if 'lag_sec' in df_clean.columns:
        reported = pd.to_numeric(df_clean['lag_sec'], errors='coerce').dropna()
        if len(reported) > 0:
            print(f"  记录 lag_sec 中位: {reported.median():+.1f}s")

    df_clean = df_clean[df_clean['_game_sec'].diff().fillna(1) >= 0]

    # 兼容不同列名
    gold_col = "gold_diff" if "gold_diff" in df_clean.columns else None

    # 取每分钟一个采样
    print(f"\n  墙钟经过 | 游戏经过 | 延迟(s) | 经济差")
    print(f"  ─────────────────────────────────────")
    sample_idx = list(range(0, len(df_clean), max(1, len(df_clean)//12)))
    for i in sample_idx:
        if i >= len(df_clean):
            break
        r = df_clean.iloc[i]
        gd = r[gold_col] if gold_col else 0
        print(f"  {r['wall_elapsed_s']:>8.1f}  | {r['game_elapsed_s']:>8.0f}  | "
              f"{r['lag_s']:+7.1f} | {gd:+5.0f}")

    if len(df_clean) > 5:
        print(f"\n  延迟统计 (排除前2条/后1条):")
        lag_stable = df_clean.iloc[2:-1]['lag_s']
        print(f"    平均: {lag_stable.mean():+.1f}s")
        print(f"    中位: {lag_stable.median():+.1f}s")
        print(f"    标准差: {lag_stable.std():.1f}s")
        print(f"    最小/最大: {lag_stable.min():+.1f} / {lag_stable.max():+.1f}")
