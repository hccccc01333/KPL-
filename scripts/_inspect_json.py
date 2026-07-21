import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

raw_dir = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\raw_snapshots')
bd = raw_dir / '1373651472_25_1779536753'
jsons = sorted(bd.glob('*.json'))
mid_idx = len(jsons) // 2
with open(jsons[mid_idx], 'r', encoding='utf-8') as f:
    d = json.load(f)
data = d.get('data', {})
print("game_duration:", data.get("game_duration"))
print("status:", data.get("status"))
print("win_camp:", data.get("win_camp"))
c1 = data.get('camp1', {})
c2 = data.get('camp2', {})
print(f"camp1: {c1.get('team_name')} gold={c1.get('gold')} kill={c1.get('kill_num')} assist={c1.get('assist_num')} death={c1.get('death_num')} tower={c1.get('push_tower_num')}")
print(f"camp2: {c2.get('team_name')} gold={c2.get('gold')} kill={c2.get('kill_num')} assist={c2.get('assist_num')} death={c2.get('death_num')} tower={c2.get('push_tower_num')}")
print(f"camp1 obj: tyrant={c1.get('kill_tyrant_num')} dark_tyrant={c1.get('kill_dark_tyrant_num')} lord={c1.get('kill_big_dragon_num')}")
print(f"camp2 obj: tyrant={c2.get('kill_tyrant_num')} dark_tyrant={c2.get('kill_dark_tyrant_num')} lord={c2.get('kill_big_dragon_num')}")
players = data.get('battle_player_list', [])
print(f"\nPlayers: {len(players)}")
if players:
    p = players[0]
    print("Player keys:", list(p.keys())[:20])
    print(f"Sample: camp={p.get('camp')} pos={p.get('position')} gold={p.get('gold')} hurt={p.get('hurt_to_hero_total')}")
