"""
KPL 历史数据回补爬虫

从 schedule_archive.csv 读取历史 match/battle 列表，批量拉取 battle/open 数据，
保存到 data/realtime/raw_snapshots/{battle_id}/backfill_*.json。

运行：
    python scripts/backfill_history.py --dry-run
    python scripts/backfill_history.py --date 2026-05-23
    python scripts/backfill_history.py --team AG --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from kpl_official_core import RAW_DIR, REALTIME_DIR, ScheduleCenter, fetch_freshest_battle


BACKFILL_STATUS_FILE = REALTIME_DIR / "backfill_status.json"


def write_status(payload: dict):
    BACKFILL_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    BACKFILL_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_battle(battle_id: str, payload: dict):
    out_dir = RAW_DIR / battle_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def collect_targets(center: ScheduleCenter, date: str | None, team: str | None, limit: int | None):
    archive = center.query_schedule_archive(date=date, team_keyword=team)
    targets = []
    for _, row in archive.iterrows():
        for battle_id in str(row.get("battle_ids", "")).split("|"):
            if battle_id:
                targets.append(
                    {
                        "league_id": row.get("league_id", ""),
                        "match_id": row.get("match_id", ""),
                        "battle_id": battle_id,
                        "date": row.get("date", ""),
                        "team1": row.get("team1", ""),
                        "team2": row.get("team2", ""),
                    }
                )
    dedup = []
    seen = set()
    for item in targets:
        if item["battle_id"] in seen:
            continue
        seen.add(item["battle_id"])
        dedup.append(item)
    return dedup[:limit] if limit else dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="按日期回补 YYYY-MM-DD")
    parser.add_argument("--team", default=None, help="按战队关键字回补")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.4)
    args = parser.parse_args()

    center = ScheduleCenter()
    targets = collect_targets(center, args.date, args.team, args.limit)
    write_status({"state": "planning", "target_count": len(targets), "dry_run": args.dry_run})
    print(f"targets: {len(targets)}")
    if args.dry_run:
        for item in targets[:30]:
            print(f"{item['date']} {item['match_id']} {item['battle_id']} {item['team1']} vs {item['team2']}")
        write_status({"state": "dry_run_done", "target_count": len(targets)})
        return

    ok = 0
    failed = 0
    for i, item in enumerate(targets, 1):
        battle_id = item["battle_id"]
        raw = fetch_freshest_battle(battle_id, n_probes=2, gap_sec=0.3)
        if raw:
            path = save_battle(battle_id, raw)
            ok += 1
            print(f"[{i}/{len(targets)}] OK {battle_id} -> {path.name}")
        else:
            failed += 1
            print(f"[{i}/{len(targets)}] FAIL {battle_id}")
        write_status(
            {
                "state": "running",
                "target_count": len(targets),
                "processed": i,
                "success": ok,
                "failed": failed,
                "current_battle_id": battle_id,
            }
        )
        time.sleep(args.sleep)

    write_status({"state": "done", "target_count": len(targets), "success": ok, "failed": failed})


if __name__ == "__main__":
    main()

