import sys, json, requests
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

RAW_DIR = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\raw_snapshots')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)',
    'Accept': 'application/json',
    'Referer': 'https://pvp.qq.com/',
    'Origin': 'https://pvp.qq.com'
}

print("=== 从 JSON 确定每局获胜方 ===\n")
for bd in sorted(RAW_DIR.iterdir()):
    jsons = sorted(bd.glob('*.json'))
    if len(jsons) < 5:
        continue
    battle_id = bd.name
    win_camp = 0
    c1_name = "?"
    c2_name = "?"

    # 先从JSON找
    for jf in reversed(jsons):
        with open(jf, 'r', encoding='utf-8') as f:
            try:
                d = json.load(f)
            except:
                continue
        data = d.get('data', {})
        c1_name = data.get('camp1', {}).get('team_name', '?')
        c2_name = data.get('camp2', {}).get('team_name', '?')
        if data.get('status') == 2:
            win_camp = data.get('win_camp', 0)
            break

    # 如果 JSON 里没有终局状态，从 API 查
    if win_camp == 0:
        r = requests.get(
            'https://prod.comp.smoba.qq.com/leaguesite/battle/open',
            params={'battle_id': battle_id}, headers=HEADERS, timeout=15
        )
        bd_data = r.json().get('data', {})
        win_camp = bd_data.get('win_camp', 0)
        c1_name = bd_data.get('camp1', {}).get('team_name', c1_name)
        c2_name = bd_data.get('camp2', {}).get('team_name', c2_name)

    winner = c1_name if win_camp == 1 else c2_name if win_camp == 2 else "UNKNOWN"
    src = "JSON" if win_camp != 0 else "API"
    print(f"  {battle_id}: camp1={c1_name} | camp2={c2_name} | win_camp={win_camp} -> Winner: {winner} [{src}]")

# 也从 match API 查官方逐局结果
print("\n=== 从 Match API 查官方逐局结果 ===\n")
r = requests.get(
    'https://prod.comp.smoba.qq.com/leaguesite/match/battles/open',
    params={'match_id': '2026052301'}, headers=HEADERS, timeout=15
)
battles_api = r.json().get('results', [])
for b in battles_api:
    bid = b.get('battle_id')
    seq = b.get('battle_seq')
    r2 = requests.get(
        'https://prod.comp.smoba.qq.com/leaguesite/battle/open',
        params={'battle_id': bid}, headers=HEADERS, timeout=15
    )
    bd_data = r2.json().get('data', {})
    wc = bd_data.get('win_camp', 0)
    c1 = bd_data.get('camp1', {}).get('team_name', '?')
    c2 = bd_data.get('camp2', {}).get('team_name', '?')
    winner = c1 if wc == 1 else c2 if wc == 2 else '?'
    print(f"  G{seq}: {c1} vs {c2} | win_camp={wc} -> {winner}")
