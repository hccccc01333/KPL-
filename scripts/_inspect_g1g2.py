import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

RAW_DIR = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\raw_snapshots')

# G1
bd = RAW_DIR / '1373651472_20_1779524218'
jsons = sorted(bd.glob('*.json'))
print(f"=== G1: {bd.name} ({len(jsons)} files) ===")
for jf in jsons:
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    data = d.get('data', {})
    c1 = data.get('camp1', {})
    c2 = data.get('camp2', {})
    dur = data.get('game_duration', 0)
    status = data.get('status', 0)
    wc = data.get('win_camp', 0)
    t1 = c1.get('team_name', '?')
    t2 = c2.get('team_name', '?')
    g1 = c1.get('gold', 0)
    g2 = c2.get('gold', 0)
    k1 = c1.get('kill_num', 0)
    k2 = c2.get('kill_num', 0)
    print(f"  {jf.name}: status={status} dur={dur}ms | {t1}(g={g1},k={k1}) vs {t2}(g={g2},k={k2}) | win_camp={wc}")

print()

# G2 - first and last snapshots
bd2 = RAW_DIR / '1373651472_21_1779526666'
jsons2 = sorted(bd2.glob('*.json'))
print(f"=== G2: {bd2.name} ({len(jsons2)} files) ===")
print("First 5:")
for jf in jsons2[:5]:
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    data = d.get('data', {})
    c1 = data.get('camp1', {})
    c2 = data.get('camp2', {})
    dur = data.get('game_duration', 0)
    status = data.get('status', 0)
    t1 = c1.get('team_name', '?')
    t2 = c2.get('team_name', '?')
    g1 = c1.get('gold', 0)
    g2 = c2.get('gold', 0)
    print(f"  {jf.name}: status={status} dur={dur}ms | {t1}(g={g1}) vs {t2}(g={g2})")

print("Last 3:")
for jf in jsons2[-3:]:
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    data = d.get('data', {})
    c1 = data.get('camp1', {})
    c2 = data.get('camp2', {})
    dur = data.get('game_duration', 0)
    status = data.get('status', 0)
    wc = data.get('win_camp', 0)
    t1 = c1.get('team_name', '?')
    t2 = c2.get('team_name', '?')
    g1 = c1.get('gold', 0)
    g2 = c2.get('gold', 0)
    k1 = c1.get('kill_num', 0)
    k2 = c2.get('kill_num', 0)
    print(f"  {jf.name}: status={status} dur={dur}ms | {t1}(g={g1},k={k1}) vs {t2}(g={g2},k={k2}) | win_camp={wc}")
