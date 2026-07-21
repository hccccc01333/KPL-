"""
Real-time data collection and snapshot simulation.

Two modes:
1. simulate_snapshots(): Generate time-series from post-game data (offline)
2. collect_live_snapshot(): Poll the API during a live match (online)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .snapshot_schema import MINUTE_BINS, SCALING_ALPHA, SNAPSHOT_COLUMNS

API_BASE = "https://prod.comp.smoba.qq.com/leaguesite/battle/open"


def simulate_snapshots(battles: pd.DataFrame) -> pd.DataFrame:
    """Generate simulated time-series snapshots from post-game battle data.

    For each battle, creates rows at minute 3/5/8/10 by proportional scaling.
    Only generates snapshots for time points BEFORE game_duration.

    Args:
        battles: DataFrame with columns including battle_id, game_duration,
                 camp1_gold, camp2_gold, camp1_kill_num, camp2_kill_num,
                 camp1_push_tower_num, camp2_push_tower_num,
                 camp1_kill_tyrant_num, camp2_kill_tyrant_num, win_camp

    Returns:
        DataFrame with SNAPSHOT_COLUMNS schema
    """
    rows = []
    for _, battle in battles.iterrows():
        duration = battle['game_duration']
        if pd.isna(duration) or duration <= 0:
            continue

        for minute in MINUTE_BINS:
            t_sec = minute * 60
            if t_sec >= duration:
                continue

            ratio_base = t_sec / duration

            def scale(final_val, stat_type):
                alpha = SCALING_ALPHA.get(stat_type, 1.0)
                return max(0, round(final_val * (ratio_base ** alpha)))

            c1_gold = scale(battle['camp1_gold'], 'gold')
            c2_gold = scale(battle['camp2_gold'], 'gold')
            c1_kill = scale(battle['camp1_kill_num'], 'kill')
            c2_kill = scale(battle['camp2_kill_num'], 'kill')
            c1_tower = scale(battle['camp1_push_tower_num'], 'tower')
            c2_tower = scale(battle['camp2_push_tower_num'], 'tower')
            c1_tyrant = scale(battle.get('camp1_kill_tyrant_num', 0), 'tyrant')
            c2_tyrant = scale(battle.get('camp2_kill_tyrant_num', 0), 'tyrant')

            rows.append({
                'battle_id': battle['battle_id'],
                'snapshot_time_sec': t_sec,
                'minute_bin': minute,
                'camp1_gold': c1_gold,
                'camp2_gold': c2_gold,
                'gold_diff': c1_gold - c2_gold,
                'camp1_kill_num': c1_kill,
                'camp2_kill_num': c2_kill,
                'kill_diff': c1_kill - c2_kill,
                'camp1_push_tower_num': c1_tower,
                'camp2_push_tower_num': c2_tower,
                'tower_diff': c1_tower - c2_tower,
                'camp1_tyrant': c1_tyrant,
                'camp2_tyrant': c2_tyrant,
                'tyrant_diff': c1_tyrant - c2_tyrant,
                'win_camp': battle['win_camp'],
                'is_simulated': True,
            })

    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def collect_live_snapshot(battle_id: str) -> dict | None:
    """Poll the live API once for a given battle_id.

    Returns parsed snapshot dict or None on failure.
    """
    try:
        resp = requests.get(API_BASE, params={"battle_id": battle_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            return None

        battle = data.get("data", {})
        return {
            "battle_id": battle_id,
            "snapshot_time_sec": battle.get("game_duration", 0) / 1000,
            "minute_bin": int(battle.get("game_duration", 0) / 1000 / 60),
            "camp1_gold": battle.get("camp_info", [{}])[0].get("gold", 0),
            "camp2_gold": battle.get("camp_info", [{}, {}])[1].get("gold", 0),
            "camp1_kill_num": battle.get("camp_info", [{}])[0].get("kill_num", 0),
            "camp2_kill_num": battle.get("camp_info", [{}, {}])[1].get("kill_num", 0),
            "camp1_push_tower_num": battle.get("camp_info", [{}])[0].get("push_tower_num", 0),
            "camp2_push_tower_num": battle.get("camp_info", [{}, {}])[1].get("push_tower_num", 0),
            "status": battle.get("status", 0),
            "win_camp": battle.get("win_camp", 0),
            "is_simulated": False,
        }
    except (requests.RequestException, KeyError, IndexError):
        return None


def collect_live_loop(
    battle_id: str,
    interval_sec: int = 30,
    max_duration_sec: int = 1800,
    output_dir: str | Path = "data/realtime/raw_snapshots",
) -> pd.DataFrame:
    """Continuously poll a live battle and save snapshots.

    Args:
        battle_id: The battle ID to poll
        interval_sec: Seconds between polls
        max_duration_sec: Safety timeout
        output_dir: Where to save the CSV

    Returns:
        DataFrame of all collected snapshots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = []
    start = time.time()

    while time.time() - start < max_duration_sec:
        snap = collect_live_snapshot(battle_id)
        if snap:
            snap['gold_diff'] = snap['camp1_gold'] - snap['camp2_gold']
            snap['kill_diff'] = snap['camp1_kill_num'] - snap['camp2_kill_num']
            snap['tower_diff'] = snap['camp1_push_tower_num'] - snap['camp2_push_tower_num']
            snapshots.append(snap)

            if snap.get('status') == 2:
                break

        time.sleep(interval_sec)

    df = pd.DataFrame(snapshots)
    if len(df) > 0:
        df.to_csv(output_dir / f"{battle_id}.csv", index=False)

    return df
