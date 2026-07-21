"""
KPL 自动赛程守护爬虫

长期挂着运行：
1. 定时刷新官方赛程表；
2. 从赛程表判断今天/未来是否有比赛；
3. 根据真实开始时间等待；
4. 比赛开始后不断检查 battle_id 是否生成；
5. battle_id 生成后实时采集、预测并写入状态文件。

运行：
    python scripts/auto_schedule_daemon.py --league-ids 20260003
    python scripts/auto_schedule_daemon.py --once --league-ids 20260003
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from kpl_official_core import LEAGUE_ID, REALTIME_DIR, ScheduleCenter, SCHEDULE_ARCHIVE_PATH, load_model
from official_match_monitor import run_once, write_status


DAEMON_STATUS_FILE = REALTIME_DIR / "daemon_status.json"


def write_daemon_status(payload: dict):
    DAEMON_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    DAEMON_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seconds_until(start_time: str) -> int:
    dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    return max(0, int((dt - datetime.now()).total_seconds()))


def pick_next_match(center: ScheduleCenter):
    archive = center.load_schedule_archive()
    if archive.empty:
        matches = center.fetch_matches()
        pending = [m for m in matches if m.status != 2]
        pending = sorted(pending, key=lambda m: m.start_time or datetime.max)
        if not pending:
            return None
        match = pending[0]
        return {
            "league_id": str(match.raw.get("league_id") or center.league_id),
            "match_id": match.match_id,
            "team1": match.team1,
            "team2": match.team2,
            "start_time": match.start_time.strftime("%Y-%m-%d %H:%M:%S") if match.start_time else "",
            "status": match.status,
            "battle_ids": "",
        }

    df = archive.copy()
    df["start_dt"] = df["start_time"].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S") if str(x) else None)
    now = datetime.now()
    window_start = now - timedelta(hours=8)
    candidates = df[
        df["start_dt"].apply(lambda x: x is not None and x >= window_start)
        & (df["status"].astype(str) != "2")
    ].sort_values("start_dt")
    if candidates.empty:
        return None
    return candidates.iloc[0].to_dict()


def refresh_archive(center: ScheduleCenter, args):
    league_ids = args.league_ids if args.league_ids else None
    archive = center.build_schedule_archive(
        league_ids=league_ids,
        start_year=args.start_year,
        end_year=args.end_year,
        event_codes=range(args.event_start, args.event_end + 1),
        sleep_sec=args.archive_sleep,
        save_path=SCHEDULE_ARCHIVE_PATH,
    )
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-ids", nargs="*", default=None, help="指定 league_id；为空则尝试候选年份赛事")
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--event-start", type=int, default=1)
    parser.add_argument("--event-end", type=int, default=12)
    parser.add_argument("--archive-refresh-min", type=int, default=30, help="赛程表刷新间隔")
    parser.add_argument("--poll-interval", type=int, default=15, help="比赛开始后的采集间隔")
    parser.add_argument("--prestart-poll", type=int, default=60, help="未开赛等待间隔")
    parser.add_argument("--archive-sleep", type=float, default=0.1)
    parser.add_argument("--once", action="store_true", help="只跑一轮调度逻辑")
    args = parser.parse_args()

    artifact = load_model()
    if artifact is None:
        raise RuntimeError("未找到模型，请先训练 V9 或保留 V8 模型")

    center = ScheduleCenter()
    history_by_battle: dict[str, list[dict]] = {}
    last_archive_refresh = 0.0

    write_daemon_status({"state": "starting", "model_name": artifact.get("model_name", "Unknown")})
    while True:
        now_ts = time.time()
        if now_ts - last_archive_refresh >= args.archive_refresh_min * 60 or not SCHEDULE_ARCHIVE_PATH.exists():
            archive = refresh_archive(center, args)
            last_archive_refresh = now_ts
            write_daemon_status({"state": "archive_refreshed", "archive_rows": len(archive), "archive_path": str(SCHEDULE_ARCHIVE_PATH)})

        match = pick_next_match(center)
        if not match:
            write_daemon_status({"state": "waiting_schedule", "message": "暂无可等待比赛"})
            if args.once:
                break
            time.sleep(args.prestart_poll)
            continue

        wait_sec = seconds_until(match["start_time"])
        write_daemon_status(
            {
                "state": "waiting_start" if wait_sec > 0 else "checking_battle",
                "match_id": match["match_id"],
                "match": f"{match['team1']} vs {match['team2']}",
                "start_time": match["start_time"],
                "seconds_until_start": wait_sec,
                "battle_ids_from_schedule": match.get("battle_ids", ""),
            }
        )

        if wait_sec > 0:
            if args.once:
                break
            time.sleep(min(args.prestart_poll, wait_sec))
            continue

        monitor_args = argparse.Namespace(
            league_id=int(str(match["league_id"]) if str(match["league_id"]).isdigit() else LEAGUE_ID),
            match_id=str(match["match_id"]),
            battle_id=None,
            date=None,
            probes=3,
            probe_gap=0.6,
        )
        outcome = run_once(
            monitor_args,
            artifact,
            history_by_battle,
            str(match["match_id"]),
        )
        write_daemon_status(
            {
                "state": "finished_match" if outcome == "match_finished" else "collecting",
                "match_id": match["match_id"],
                "match": f"{match['team1']} vs {match['team2']}",
                "start_time": match["start_time"],
                "last_outcome": outcome,
            }
        )

        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

