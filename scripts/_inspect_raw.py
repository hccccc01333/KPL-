import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
from pathlib import Path

raw_dir = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\raw_snapshots')
for d in sorted(raw_dir.iterdir()):
    if d.is_dir() and d.name != '.gitkeep':
        jsons = sorted(d.glob('*.json'))
        if len(jsons) < 5:
            continue
        # Pick a mid-game snapshot
        mid = jsons[len(jsons)//2]
        with open(mid, encoding='utf-8') as f:
            data = json.load(f)
        bd = data.get('data', {})
        camp1 = bd.get('camp1', {})
        camp2 = bd.get('camp2', {})
        print(f'=== {d.name} | file: {mid.name} ===')
        print(f'\n--- camp1 (team-level) ---')
        for k, v in camp1.items():
            if 'icon' not in k:
                print(f'  {k} = {v}')
        print(f'\n--- camp2 (team-level) ---')
        for k, v in camp2.items():
            if 'icon' not in k:
                print(f'  {k} = {v}')

        players = bd.get('battle_player_list', [])
        print(f'\n--- battle_player_list: {len(players)} players ---')
        if players:
            p0 = players[0]
            print(f'  player[0] keys: {list(p0.keys())}')
            for k, v in p0.items():
                if 'icon' not in k and 'list' not in k and 'url' not in k:
                    print(f'    {k} = {v}')
            # Also show equip_list structure
            if 'equip_list' in p0:
                eq = p0['equip_list']
                print(f'    equip_list: {len(eq)} items, first={eq[0] if eq else None}')
        break
