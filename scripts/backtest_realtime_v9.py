"""
KPL V9/V11 诚实回测：真实局整局 holdout + early/mid/late 主指标。

只走 kpl_official_core.predict_probability，避免特征漂移。

运行：
    python scripts/backtest_realtime_v9.py
    python scripts/backtest_realtime_v9.py --holdout 8 --gates
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import joblib
import numpy as np
import pandas as pd

from kpl_official_core import (
    PREDICTION_DIR,
    RAW_DIR,
    REALTIME_DIR,
    classify_battle_swing,
    evaluate_phase_metrics,
    load_battle_jsons,
    load_model,
    predict_probability,
    resolve_snapshot_root,
)


REPORT_PATH = REALTIME_DIR / "backtest_v9_report.json"
BAD_BATTLE_PREFIX = "970998288_6"

# Baseline from V9 real-first release (2026-07-19)
BASELINE = {
    "early_brier": 0.1335,
    "mid_brier": 0.0590,
    "late_brier": 0.0594,
    "all_brier": 0.0888,
}
# V11 acceptance (plan): mid≤0.065, early≤0.110, all≤0.085; bad battle Acc≥0.60
GATES = {
    "early_brier_max": 0.110,
    "all_brier_max": 0.085,
    "mid_brier_max": 0.065,
    "late_brier_max": 0.100,
    "worst_acc_min": 0.35,
    "comeback_acc_min": 0.70,
    "bad_battle_acc_min": 0.60,
    "bad_battle_late_dir_min": 0.55,
}


def battle_timestamp(battle_id: str) -> int:
    try:
        return int(str(battle_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


def list_labeled_real_battles(min_snaps: int = 4, *, from_raw: bool = False) -> list[dict]:
    rows = []
    root = resolve_snapshot_root(from_raw=from_raw)
    if not root.exists():
        return rows
    for battle_dir in root.iterdir():
        if not battle_dir.is_dir():
            continue
        snaps = load_battle_jsons(battle_dir)
        if len(snaps) < min_snaps:
            continue
        win_camp = 0
        for snap in reversed(snaps):
            if int(snap.get("win_camp", 0) or 0) > 0:
                win_camp = int(snap["win_camp"])
                break
        if win_camp not in (1, 2):
            continue
        minutes = {max(int(s.get("minute", 0)), 1) for s in snaps}
        rows.append(
            {
                "battle_id": battle_dir.name,
                "win_camp": win_camp,
                "snapshots": snaps,
                "n_snaps": len(snaps),
                "n_minutes": len(minutes),
                "ts": battle_timestamp(battle_dir.name),
            }
        )
    rows.sort(key=lambda x: x["ts"])
    return rows


def select_holdout(battles: list[dict], n_holdout: int) -> tuple[list[dict], list[dict]]:
    if not battles:
        return [], []
    n = min(max(n_holdout, 1), max(1, len(battles) // 3), len(battles))
    holdout = battles[-n:]
    train_like = battles[:-n] if n < len(battles) else []
    return holdout, train_like


def is_comeback_battle(battle: dict) -> bool:
    info = classify_battle_swing(battle.get("snapshots") or [], int(battle.get("win_camp", 0) or 0))
    return bool(info.get("comeback"))


def find_battle_by_prefix(battles: list[dict], prefix: str) -> dict | None:
    for b in battles:
        if str(b["battle_id"]).startswith(prefix):
            return b
    return None


def timeline_predictions(artifact: dict, battle: dict) -> pd.DataFrame:
    history: deque = deque(maxlen=40)
    seen = set()
    rows = []
    label = int(battle["win_camp"] == 1)
    for snap in battle["snapshots"]:
        minute_bin = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
        history.append(snap)
        if minute_bin in seen:
            continue
        seen.add(minute_bin)
        prob, feats, explain = predict_probability(artifact, snap, list(history))
        gold_diff = float(snap.get("camp1_gold", 0) or 0) - float(snap.get("camp2_gold", 0) or 0)
        rows.append(
            {
                "battle_id": battle["battle_id"],
                "minute": float(snap.get("minute", minute_bin)),
                "minute_bin": minute_bin,
                "label": label,
                "prob": prob,
                "prob_raw": feats.get("_prob_raw", prob),
                "gold_diff": gold_diff,
                "gold_guard": bool(feats.get("_gold_guard_applied", 0)),
                "low_confidence": bool(feats.get("_low_confidence", 0)),
                "top_factor": explain[0]["factor"] if explain else "",
                "correct": int((prob >= 0.5) == bool(label)),
            }
        )
    return pd.DataFrame(rows)


def battle_late_direction_acc(tl: pd.DataFrame, minute_min: int = 8) -> float:
    """Direction accuracy vs label for mid/late minutes."""
    if tl.empty:
        return 0.0
    sub = tl[tl["minute_bin"] >= minute_min]
    if sub.empty:
        return float((tl["prob"] >= 0.5).eq(tl["label"]).mean())
    return float((sub["prob"] >= 0.5).eq(sub["label"]).mean())


def evaluate_csv_predictions(battle_ids: list[str] | None = None) -> dict:
    if not PREDICTION_DIR.exists():
        return {}
    rows = []
    for path in PREDICTION_DIR.glob("*.csv"):
        if battle_ids is not None and path.stem not in battle_ids:
            continue
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if df.empty or "camp1_win_prob" not in df.columns:
            continue
        win_camp = pd.to_numeric(df.get("win_camp"), errors="coerce").dropna()
        win_camp = int(win_camp.iloc[-1]) if len(win_camp) else 0
        if win_camp not in (1, 2):
            continue
        label = int(win_camp == 1)
        sub = df.dropna(subset=["camp1_win_prob"]).copy()
        if sub.empty:
            continue
        sub["minute"] = pd.to_numeric(sub.get("minute"), errors="coerce").fillna(1)
        sub["prob"] = pd.to_numeric(sub["camp1_win_prob"], errors="coerce")
        sub = sub.dropna(subset=["prob"])
        for _, r in sub.iterrows():
            rows.append({"minute": r["minute"], "label": label, "prob": float(r["prob"])})
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    return evaluate_phase_metrics(frame["minute"].values, frame["label"].values, frame["prob"].values)


def check_gates(
    holdout_metrics: dict,
    worst_acc: float,
    comeback_acc: float | None = None,
    *,
    bad_battle_acc: float | None = None,
    bad_battle_late_dir: float | None = None,
) -> dict:
    early_b = holdout_metrics.get("early", {}).get("brier", 1.0)
    mid_b = holdout_metrics.get("mid", {}).get("brier", 1.0)
    late_b = holdout_metrics.get("late", {}).get("brier", 1.0)
    all_b = holdout_metrics.get("all", {}).get("brier", 1.0)
    worst_or_comeback = (worst_acc >= GATES["worst_acc_min"]) or (
        comeback_acc is not None and comeback_acc >= GATES["comeback_acc_min"]
    )
    checks = {
        "early_brier_le_0.110": early_b <= GATES["early_brier_max"],
        "all_brier_le_0.085": all_b <= GATES["all_brier_max"],
        "mid_brier_le_0.065": mid_b <= GATES["mid_brier_max"],
        "late_brier_le_0.100": late_b <= GATES["late_brier_max"],
        "worst_acc_or_comeback_ok": worst_or_comeback,
    }
    if bad_battle_acc is not None:
        checks["bad_battle_acc_ge_0.60"] = bad_battle_acc >= GATES["bad_battle_acc_min"]
    if bad_battle_late_dir is not None:
        checks["bad_battle_late_dir_ge_0.55"] = bad_battle_late_dir >= GATES["bad_battle_late_dir_min"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": {
            "early_brier": early_b,
            "mid_brier": mid_b,
            "late_brier": late_b,
            "all_brier": all_b,
            "worst_acc": worst_acc,
            "comeback_acc": comeback_acc,
            "bad_battle_acc": bad_battle_acc,
            "bad_battle_late_dir": bad_battle_late_dir,
            "baseline": BASELINE,
            "gates": GATES,
        },
    }


def run_backtest(
    artifact: dict,
    holdout_n: int = 8,
    verbose: bool = True,
    check: bool = False,
    *,
    from_raw: bool = False,
) -> dict:
    battles = list_labeled_real_battles(from_raw=from_raw)
    holdout, _ = select_holdout(battles, holdout_n)
    if verbose:
        root = resolve_snapshot_root(from_raw=from_raw)
        print(f"snapshot root: {root} (from_raw={from_raw})")
        print(f"labeled real battles: {len(battles)}, holdout: {len(holdout)}")
        print("holdout ids:", [b["battle_id"] for b in holdout])
        print(f"model version={artifact.get('version')} name={artifact.get('model_name')}")

    frames = []
    per_battle = []
    comeback_frames = []
    for battle in holdout:
        tl = timeline_predictions(artifact, battle)
        if tl.empty:
            continue
        frames.append(tl)
        phase = evaluate_phase_metrics(tl["minute"].values, tl["label"].values, tl["prob"].values)
        comeback = is_comeback_battle(battle)
        if comeback:
            comeback_frames.append(tl)
        per_battle.append(
            {
                "battle_id": battle["battle_id"],
                "win_camp": battle["win_camp"],
                "n_points": int(len(tl)),
                "is_comeback": comeback,
                "acc": phase["all"]["accuracy"],
                "brier": phase["all"]["brier"],
                "metrics": phase,
            }
        )
        if verbose:
            m = phase["all"]
            tag = " [COMEBACK]" if comeback else ""
            print(
                f"  {battle['battle_id']}: n={m['n']} Brier={m['brier']:.4f} "
                f"ECE={m['ece']:.4f} Acc={m['accuracy']:.3f} Dir={m['direction_acc']:.3f}{tag}"
            )

    if not frames:
        report = {"state": "empty", "holdout_battles": [], "holdout_real": {}}
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    all_df = pd.concat(frames, ignore_index=True)
    holdout_metrics = evaluate_phase_metrics(all_df["minute"].values, all_df["label"].values, all_df["prob"].values)
    csv_metrics = evaluate_csv_predictions([b["battle_id"] for b in holdout])
    worst = sorted(per_battle, key=lambda x: x["acc"])[:5]
    worst_acc = float(worst[0]["acc"]) if worst else 0.0

    if comeback_frames:
        c_df = pd.concat(comeback_frames, ignore_index=True)
        comeback_metrics = evaluate_phase_metrics(c_df["minute"].values, c_df["label"].values, c_df["prob"].values)
    else:
        comeback_metrics = {}

    # Special-case bad battle (may be outside holdout set)
    bad_battle = find_battle_by_prefix(battles, BAD_BATTLE_PREFIX)
    bad_battle_report = None
    bad_battle_acc = None
    bad_battle_late_dir = None
    if bad_battle is not None:
        bad_tl = timeline_predictions(artifact, bad_battle)
        if not bad_tl.empty:
            bad_phase = evaluate_phase_metrics(bad_tl["minute"].values, bad_tl["label"].values, bad_tl["prob"].values)
            bad_battle_acc = float(bad_phase["all"]["accuracy"])
            bad_battle_late_dir = battle_late_direction_acc(bad_tl, 8)
            swing_info = classify_battle_swing(bad_battle["snapshots"], bad_battle["win_camp"])
            bad_battle_report = {
                "battle_id": bad_battle["battle_id"],
                "acc": bad_battle_acc,
                "late_dir_acc": bad_battle_late_dir,
                "brier": bad_phase["all"]["brier"],
                "metrics": bad_phase,
                "swing": swing_info,
                "gold_guard_hits": int(bad_tl["gold_guard"].sum()) if "gold_guard" in bad_tl.columns else 0,
            }
            if verbose:
                print(
                    f"\nBad battle {bad_battle['battle_id']}: Acc={bad_battle_acc:.3f} "
                    f"late_dir={bad_battle_late_dir:.3f} Brier={bad_phase['all']['brier']:.4f} "
                    f"guard_hits={bad_battle_report['gold_guard_hits']}"
                )

    gates = check_gates(
        holdout_metrics,
        worst_acc,
        comeback_metrics.get("all", {}).get("accuracy") if comeback_metrics else None,
        bad_battle_acc=bad_battle_acc,
        bad_battle_late_dir=bad_battle_late_dir,
    )

    report = {
        "state": "ok",
        "version": artifact.get("version"),
        "model_name": artifact.get("model_name"),
        "trained_at": artifact.get("trained_at"),
        "holdout_battles": [b["battle_id"] for b in holdout],
        "holdout_real": holdout_metrics,
        "comeback_battles": [p["battle_id"] for p in per_battle if p["is_comeback"]],
        "comeback_metrics": comeback_metrics,
        "bad_battle": bad_battle_report,
        "worst_battles": [{"battle_id": w["battle_id"], "acc": w["acc"], "brier": w["brier"], "is_comeback": w["is_comeback"]} for w in worst],
        "per_battle": per_battle,
        "prediction_csv_holdout": csv_metrics,
        "gates": gates,
        "headline": {
            "metric": "holdout_real_brier",
            "early_brier": holdout_metrics.get("early", {}).get("brier"),
            "mid_brier": holdout_metrics.get("mid", {}).get("brier"),
            "late_brier": holdout_metrics.get("late", {}).get("brier"),
            "all_brier": holdout_metrics.get("all", {}).get("brier"),
            "all_ece": holdout_metrics.get("all", {}).get("ece"),
            "worst_acc": worst_acc,
            "comeback_acc": comeback_metrics.get("all", {}).get("accuracy") if comeback_metrics else None,
            "bad_battle_acc": bad_battle_acc,
            "bad_battle_late_dir": bad_battle_late_dir,
            "note": "面试主数字请引用 holdout_real 分阶段 Brier/ECE，而非混合 AUC",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print("\n=== Holdout Real (headline) ===")
        for phase, m in holdout_metrics.items():
            print(
                f"  {phase:5s} n={m['n']:4d} Brier={m['brier']:.4f} ECE={m['ece']:.4f} "
                f"Acc={m['accuracy']:.3f} AUC={m['auc']:.3f} Dir={m['direction_acc']:.3f}"
            )
        if comeback_metrics:
            print("Comeback subset:", comeback_metrics.get("all"))
        print("Worst battles:", report["worst_battles"])
        print("Gates:", gates["checks"], "PASSED" if gates["passed"] else "FAILED")
        print(f"saved: {REPORT_PATH}")

    if check and not gates["passed"]:
        raise SystemExit("Acceptance gates failed — see gates in backtest_v9_report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--model", default=None, help="joblib path; default load_model()")
    parser.add_argument("--gates", action="store_true", help="exit non-zero if acceptance gates fail")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="use raw_snapshots instead of curated datasets/clean",
    )
    args = parser.parse_args()

    if args.model:
        artifact = joblib.load(args.model)
    else:
        artifact = load_model()
    if not artifact:
        raise FileNotFoundError("未找到可用模型")
    run_backtest(
        artifact,
        holdout_n=args.holdout,
        verbose=True,
        check=args.gates,
        from_raw=args.from_raw,
    )


if __name__ == "__main__":
    main()
