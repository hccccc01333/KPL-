"""Audit schedule archive vs local realtime collection."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from pathlib import Path

import pandas as pd

from kpl_official_core import LEAGUE_ID, RAW_DIR, PREDICTION_DIR, ScheduleCenter

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "realtime" / "schedule_archive.csv"


def main():
    archive = pd.read_csv(ARCHIVE)
    local = {p.name for p in RAW_DIR.iterdir() if p.is_dir()} if RAW_DIR.exists() else set()
    preds = {p.stem for p in PREDICTION_DIR.glob("*.csv")} if PREDICTION_DIR.exists() else set()

    rows = []
    for _, r in archive.iterrows():
        bids = [b for b in str(r.get("battle_ids", "")).split("|") if b]
        covered = [b for b in bids if b in local]
        rows.append({
            "match_id": str(r["match_id"]),
            "date": r["date"],
            "team1": r["team1"],
            "team2": r["team2"],
            "status": r["status_text"],
            "bo": int(r.get("bo_type", 5) or 5),
            "archive_n": len(bids),
            "local_n": len(covered),
            "bids": bids,
            "covered": covered,
        })
    df = pd.DataFrame(rows)

    print("=== ARCHIVE ===")
    print(f"matches: {len(df)}  |  dates: {df['date'].min()} ~ {df['date'].max()}")
    print("status:", df["status"].value_counts().to_dict())
    print(f"battle_ids in archive: {df['archive_n'].sum()}")
    print(f"matches with 0 battle_ids: {(df['archive_n']==0).sum()}")

    print("\n=== LOCAL ===")
    print(f"raw battle dirs: {len(local)}")
    print(f"prediction csv: {len(preds)}")

    ended = df[df["status"] == "已结束"]
    if len(ended):
        ended = ended.copy()
        ended["pct"] = ended.apply(lambda r: r["local_n"] / r["archive_n"] if r["archive_n"] else 0, axis=1)
        print("\n=== ENDED MATCHES (archive) ===")
        for _, r in ended.iterrows():
            print(f"  {r['date']} {r['team1']} vs {r['team2']}: {r['local_n']}/{r['archive_n']} ({r['pct']:.0%})")

    missing = df[(df["archive_n"] > 0) & (df["local_n"] == 0)]
    print(f"\n=== HAS BATTLE IDS BUT NO LOCAL DATA: {len(missing)} ===")
    for _, r in missing.head(15).iterrows():
        print(f"  {r['date']} [{r['status']}] {r['team1']} vs {r['team2']} ({r['archive_n']} games)")

    partial = df[(df["archive_n"] > 0) & (df["local_n"] > 0) & (df["local_n"] < df["archive_n"])]
    print(f"\n=== PARTIAL BO COVERAGE: {len(partial)} ===")
    for _, r in partial.iterrows():
        miss = [b for b in r["bids"] if b not in local]
        print(f"  {r['date']} {r['team1']} vs {r['team2']}: {r['local_n']}/{r['archive_n']} miss={miss}")

    # snapshot depth
    depths = []
    for bid in local:
        n = len(list((RAW_DIR / bid).glob("*.json")))
        depths.append((bid, n))
    depths.sort(key=lambda x: -x[1])
    single = sum(1 for _, n in depths if n <= 1)
    print(f"\n=== SNAPSHOT DEPTH ===")
    print(f"battles <=1 snapshot: {single}/{len(depths)}")
    if depths:
        print("top5:", depths[:5])
        print("bottom5:", depths[-5:])

    # live API compare
    print("\n=== LIVE API (summer 2026) ===")
    center = ScheduleCenter(LEAGUE_ID)
    live = center.fetch_matches()
    live_today = [m for m in live if m.start_time and m.start_time.strftime("%Y-%m-%d") >= "2026-06-17"]
    print(f"live matches since 6/17: {len(live_today)}")
    status_cnt = {}
    for m in live_today:
        status_cnt[m.status_text] = status_cnt.get(m.status_text, 0) + 1
    print("live status:", status_cnt)

    archive_ids = set(df["match_id"])
    live_ids = {m.match_id for m in live_today}
    print(f"in live not archive: {len(live_ids - archive_ids)}")
    print(f"in archive not live: {len(archive_ids - live_ids)}")

    stale = df[(df["date"] >= "2026-06-19") & (df["status"] == "未开始")]
    print(f"\narchive stale (>=6/19 still 未开始): {len(stale)} rows")

    print("\n=== LIVE API PER MATCH (Jun 2026+) ===")
    from collections import defaultdict

    by_date = defaultdict(list)
    for m in center.fetch_matches():
        if not m.start_time or m.start_time < pd.Timestamp("2026-06-17"):
            continue
        bids = center.fetch_battles(m.match_id)
        bid_list = [str(b.get("battle_id")) for b in bids if b.get("battle_id")]
        cov = sum(1 for b in bid_list if b in local)
        by_date[m.start_time.strftime("%Y-%m-%d")].append({
            "match": f"{m.team1} vs {m.team2}",
            "status": m.status_text,
            "api_n": len(bid_list),
            "local_n": cov,
            "missing": [b for b in bid_list if b not in local],
        })

    total_api = total_local = 0
    miss_matches = part_matches = ok_matches = 0
    for d in sorted(by_date):
        print(f"\n--- {d} ---")
        for x in by_date[d]:
            total_api += x["api_n"]
            total_local += x["local_n"]
            if x["api_n"] == 0:
                flag = "EMPTY"
            elif x["local_n"] == x["api_n"]:
                flag = "OK"
                ok_matches += 1
            elif x["local_n"] > 0:
                flag = "PART"
                part_matches += 1
            else:
                flag = "MISS"
                miss_matches += 1
            print(f"  [{flag}] {x['match']} {x['status']} local={x['local_n']}/{x['api_n']}")
            if x["missing"] and 0 < x["local_n"] < x["api_n"]:
                print(f"       missing: {x['missing']}")

    print(f"\n=== SUMMARY Jun17+ ===")
    print(f"API battles: {total_api}  local covered: {total_local}  rate={total_local/total_api:.0%}" if total_api else "no api battles")
    print(f"matches OK/PART/MISS: {ok_matches}/{part_matches}/{miss_matches}")


if __name__ == "__main__":
    main()
