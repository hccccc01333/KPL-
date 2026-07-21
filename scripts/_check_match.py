"""快速检查当前比赛状态"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)',
    'Accept': 'application/json',
    'Referer': 'https://pvp.qq.com/',
    'Origin': 'https://pvp.qq.com'
}

# Check match battles
r = requests.get(
    'https://prod.comp.smoba.qq.com/leaguesite/match/battles/open',
    params={'match_id': '2026052301'}, headers=HEADERS, timeout=15
)
data = r.json()
battles = data.get('results', [])
print(f"Match 2026052301 battles: {len(battles)}")
for b in battles:
    bid = b.get('battle_id', '?')
    seq = b.get('battle_seq', '?')
    print(f"  seq={seq} battle_id={bid}")

# Check live match score
r2 = requests.get(
    'https://prod.comp.smoba.qq.com/leaguesite/matches/open',
    params={'league_id': 20260002}, headers=HEADERS, timeout=15
)
matches = r2.json().get('results', [])
for m in matches:
    if str(m.get('match_id')) == '2026052301':
        camp1 = m.get('camp1', {})
        camp2 = m.get('camp2', {})
        if isinstance(camp1, dict) and isinstance(camp2, dict):
            t1 = camp1.get('team_name', '?')
            t2 = camp2.get('team_name', '?')
            s1 = camp1.get('score', 0)
            s2 = camp2.get('score', 0)
            print(f"\nMatch status: {m.get('status')}")
            print(f"Score: {t1} {s1} : {s2} {t2}")
            print(f"Start: {m.get('start_time')}")
        break

# Check the single battle's live status
if battles:
    bid = battles[0].get('battle_id')
    r3 = requests.get(
        'https://prod.comp.smoba.qq.com/leaguesite/battle/open',
        params={'battle_id': bid}, headers=HEADERS, timeout=15
    )
    bd = r3.json().get('data', {})
    print(f"\nBattle {bid}:")
    print(f"  status={bd.get('status')}, win_camp={bd.get('win_camp')}")
    print(f"  game_duration={bd.get('game_duration')}")
    c1 = bd.get('camp1', {})
    c2 = bd.get('camp2', {})
    print(f"  {c1.get('team_name')} gold={c1.get('gold')} kill={c1.get('kill_num')} tower={c1.get('push_tower_num')}")
    print(f"  {c2.get('team_name')} gold={c2.get('gold')} kill={c2.get('kill_num')} tower={c2.get('push_tower_num')}")
