"""
Diagnose one real battle with the current V11 artifact (minute timeline).

Usage:
    python scripts/diagnose_battle_v11.py
    python scripts/diagnose_battle_v11.py --battle-id 501236240_10_1783254441
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd

from backtest_realtime_v9 import list_labeled_real_battles, timeline_predictions
from kpl_official_core import REALTIME_DIR, load_model, phase_of_minute


def find_battle(battle_id: str, from_raw: bool = False) -> dict | None:
    for b in list_labeled_real_battles(from_raw=from_raw):
        if b["battle_id"] == battle_id or b["battle_id"].startswith(battle_id):
            return b
    return None


def summarize(tl: pd.DataFrame, battle: dict) -> dict:
    wrong = tl[tl["correct"] == 0].copy()
    by_phase = {}
    for phase in ("early", "mid", "late"):
        sub = tl[tl["minute_bin"].map(phase_of_minute) == phase]
        if sub.empty:
            by_phase[phase] = {"n": 0, "acc": None, "brier": None, "avg_prob": None}
            continue
        y = sub["label"].astype(int)
        p = sub["prob"].astype(float)
        by_phase[phase] = {
            "n": int(len(sub)),
            "acc": float((p >= 0.5).eq(y).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "avg_prob": float(p.mean()),
            "avg_gold_diff": float(sub["gold_diff"].mean()),
            "guard_hits": int(sub["gold_guard"].sum()),
        }
    return {
        "battle_id": battle["battle_id"],
        "win_camp": battle["win_camp"],
        "camp1": battle["snapshots"][0].get("camp1_team", ""),
        "camp2": battle["snapshots"][0].get("camp2_team", ""),
        "n_points": int(len(tl)),
        "acc": float(tl["correct"].mean()) if len(tl) else None,
        "brier": float(((tl["prob"] - tl["label"]) ** 2).mean()) if len(tl) else None,
        "gold_guard_hits": int(tl["gold_guard"].sum()),
        "wrong_minutes": wrong[["minute_bin", "prob", "prob_raw", "gold_diff", "gold_guard", "top_factor"]].to_dict(
            "records"
        )
        if len(wrong)
        else [],
        "by_phase": by_phase,
        "timeline": tl.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battle-id", default="501236240_10_1783254441")
    parser.add_argument("--from-raw", action="store_true")
    args = parser.parse_args()

    battle = find_battle(args.battle_id, from_raw=args.from_raw)
    if battle is None:
        raise SystemExit(f"battle not found: {args.battle_id}")

    artifact = load_model()
    tl = timeline_predictions(artifact, battle)
    report = summarize(tl, battle)

    out = REALTIME_DIR / f"diagnose_{battle['battle_id']}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== diagnose {battle['battle_id']} ===")
    print(f"  {report['camp1']} vs {report['camp2']} | win_camp={report['win_camp']}")
    print(f"  acc={report['acc']:.3f} brier={report['brier']:.4f} guard_hits={report['gold_guard_hits']}")
    for phase, m in report["by_phase"].items():
        if not m["n"]:
            continue
        print(
            f"  {phase:5s} n={m['n']} acc={m['acc']:.3f} brier={m['brier']:.4f} "
            f"avg_p={m['avg_prob']:.3f} gold={m['avg_gold_diff']:.0f} guard={m['guard_hits']}"
        )
    print("\nminute timeline:")
    print(
        tl[
            ["minute_bin", "prob", "prob_raw", "gold_diff", "gold_guard", "correct", "top_factor"]
        ].to_string(index=False)
    )
    if report["wrong_minutes"]:
        print(f"\nwrong minutes ({len(report['wrong_minutes'])}):")
        for row in report["wrong_minutes"]:
            print(
                f"  m={row['minute_bin']} p={row['prob']:.3f} raw={row['prob_raw']:.3f} "
                f"gold={row['gold_diff']:.0f} guard={row['gold_guard']} top={row['top_factor']}"
            )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
