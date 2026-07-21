"""
V11 causal FE group ablation on clean holdout.

For each new FE group:
  1) inference zero-out with current artifact (does the model rely on it?)
  2) retrain early RF without the group (does keeping it help early Brier?)

Output: data/realtime/ablation_v11_report.json

Usage:
    python scripts/ablate_v11_features.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from backtest_realtime_v9 import list_labeled_real_battles, select_holdout
from kpl_official_core import FeatureBuilder, REALTIME_DIR, load_model, phase_of_minute, predict_probability
from train_realtime_model_v9 import (
    BLEND_END,
    BLEND_START,
    EARLY_FEATURE_COLUMNS,
    V9_FEATURE_COLUMNS,
    build_causal_team_winrate_maps,
    extract_real_snapshots,
    finalize_holdout_probs,
)


REPORT_PATH = REALTIME_DIR / "ablation_v11_report.json"

# New causal FE groups (relative to pre-V11 baseline)
FE_GROUPS = {
    "accel": ["gold_diff_accel"],
    "momentum": ["kill_momentum_diff"],
    "lane_crush": ["lane_crush_count"],
    "roll4": [
        "gold_diff_roll4",
        "gold_diff_roll4_per_min",
        "gold_diff_jungle_roll4",
        "gold_diff_adc_roll4",
    ],
    "roll10": ["gold_diff_roll10", "gold_diff_roll10_per_min"],
    "win35": ["win35_kill_diff", "win35_death_diff", "win35_hurt_diff"],
    "win911": ["win911_kill_diff", "win911_death_diff", "win911_hurt_diff"],
    "concentration": ["hurt_conc_diff", "behurt_conc_diff"],
    "obj_convert": ["obj_tower_convert"],
    "pos_kda": [
        "kill_diff_jungle",
        "kill_diff_adc",
        "death_diff_jungle",
        "death_diff_adc",
        "behurt_diff_top",
        "behurt_diff_support",
    ],
}


def _build_holdout_df(holdout: list[dict], fallback_wr: dict) -> pd.DataFrame:
    rows = []
    for battle in holdout:
        history = []
        seen = set()
        camp1 = battle["snapshots"][0].get("camp1_team", "")
        camp2 = battle["snapshots"][0].get("camp2_team", "")
        team_wr = {camp1: fallback_wr.get(camp1, 0.5), camp2: fallback_wr.get(camp2, 0.5)}
        label = int(battle["win_camp"] == 1)
        for snap in battle["snapshots"]:
            minute = max(int(snap.get("minute", 0)), 1)
            if minute in seen:
                history.append(snap)
                continue
            seen.add(minute)
            feats = FeatureBuilder(team_wr).build(snap, history)
            rows.append(
                {
                    "battle_id": battle["battle_id"],
                    "minute_bin": minute,
                    "label": label,
                    "gold_ratio": feats.get("gold_ratio", 0.5),
                    **feats,
                }
            )
            history.append(snap)
    return pd.DataFrame(rows).fillna(0)


def _phase_brier(y: np.ndarray, p: np.ndarray, minutes: np.ndarray, phase: str) -> float:
    mask = np.array([phase_of_minute(m) == phase for m in minutes])
    if not np.any(mask):
        return float("nan")
    return float(brier_score_loss(y[mask], p[mask]))


def _fit_early_rf(x: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def run_ablation(holdout_n: int = 8) -> dict:
    battles_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "battles.csv"
    battles = pd.read_csv(battles_path)

    per_battle_wr, fallback_wr = build_causal_team_winrate_maps(battles)
    labeled = list_labeled_real_battles(from_raw=False)
    holdout, _ = select_holdout(labeled, holdout_n)
    holdout_ids = {b["battle_id"] for b in holdout}
    hold_df = _build_holdout_df(holdout, fallback_wr)

    # Real-only for ablation (sim needs position aggregate from players.csv)
    real_df = extract_real_snapshots(per_battle_wr, fallback_wr, exclude_ids=holdout_ids, from_raw=False).fillna(0)
    train_df = real_df.assign(is_real=1)

    artifact = load_model()
    early_cols = [c for c in artifact.get("early_feature_columns", EARLY_FEATURE_COLUMNS) if c in V9_FEATURE_COLUMNS]
    mid_cols = [c for c in V9_FEATURE_COLUMNS if c in hold_df.columns]

    # --- 1) inference zero-out with current artifact ---
    print("=== inference zero-out (current artifact) ===")
    base_rows = []
    for battle in holdout:
        history = []
        seen = set()
        for snap in battle["snapshots"]:
            minute = max(int(snap.get("minute", 0)), 1)
            history.append(snap)
            if minute in seen:
                continue
            seen.add(minute)
            p, feats, _ = predict_probability(artifact, snap, list(history))
            base_rows.append(
                {
                    "battle_id": battle["battle_id"],
                    "minute_bin": minute,
                    "label": int(battle["win_camp"] == 1),
                    "prob": p,
                    "feats": feats,
                }
            )
    # For zero-out we re-score via FeatureBuilder + model internals is hard;
    # instead zero columns in hold_df and re-run finalize path using artifact models.
    model_early = artifact["model_early"]
    model_mid = artifact["model"]
    cal_early = artifact.get("calibrator_early")
    cal_mid = artifact.get("calibrator")
    early_clip = tuple(artifact.get("early_prob_clip", (0.02, 0.98)))

    def score_holdout(df: pd.DataFrame, drop_cols: list[str] | None = None) -> np.ndarray:
        x = df[mid_cols].copy()
        if drop_cols:
            for c in drop_cols:
                if c in x.columns:
                    x[c] = 0.0
        xe = x.copy()
        for c in early_cols:
            if c not in xe.columns:
                xe[c] = 0.0
        xe = xe[early_cols]
        p_e = model_early.predict_proba(xe)[:, 1]
        p_m = model_mid.predict_proba(x[mid_cols])[:, 1]
        minutes = df["minute_bin"].astype(float).values
        blend_w = np.clip((minutes - BLEND_START) / max(BLEND_END - BLEND_START, 1e-6), 0.0, 1.0)
        gold_diff = df["gold_diff"].values if "gold_diff" in df.columns else np.zeros(len(df))
        if "gold_diff" not in df.columns and "gold_diff_per_min" in df.columns:
            gold_diff = df["gold_diff_per_min"].values * np.maximum(minutes, 1.0)
        fake = pd.DataFrame(
            {
                "battle_id": df["battle_id"].values,
                "minute_bin": minutes,
                "label": df["label"].values,
                "gold_ratio": df["gold_ratio"].values,
                "gold_diff": gold_diff,
                "prob_early": p_e,
                "prob_midlate": p_m,
                "blend_w": blend_w,
                "tyrant_diff": df["tyrant_diff"].values if "tyrant_diff" in df.columns else 0.0,
                "dark_tyrant_diff": df["dark_tyrant_diff"].values if "dark_tyrant_diff" in df.columns else 0.0,
                "storm_diff": df["storm_diff"].values if "storm_diff" in df.columns else 0.0,
            }
        )
        return finalize_holdout_probs(fake, cal_early, cal_mid, early_clip, use_gold_guard=True)

    # Need hold_df to include gold_ratio and aligned mid cols
    for c in mid_cols:
        if c not in hold_df.columns:
            hold_df[c] = 0.0
    y = hold_df["label"].astype(int).values
    minutes = hold_df["minute_bin"].values
    p_base = score_holdout(hold_df)
    base_early = _phase_brier(y, p_base, minutes, "early")
    base_mid = _phase_brier(y, p_base, minutes, "mid")
    base_all = float(brier_score_loss(y, p_base))
    print(f"  baseline early={base_early:.4f} mid={base_mid:.4f} all={base_all:.4f}")

    zero_results = []
    for name, cols in FE_GROUPS.items():
        present = [c for c in cols if c in mid_cols]
        if not present:
            continue
        p = score_holdout(hold_df, drop_cols=present)
        early_b = _phase_brier(y, p, minutes, "early")
        mid_b = _phase_brier(y, p, minutes, "mid")
        all_b = float(brier_score_loss(y, p))
        row = {
            "group": name,
            "cols": present,
            "early_brier": early_b,
            "mid_brier": mid_b,
            "all_brier": all_b,
            "delta_early": early_b - base_early,
            "delta_mid": mid_b - base_mid,
            "delta_all": all_b - base_all,
            # zeroing improves (lower brier) => group hurts / noisy
            "hurts_if_zero_improves": early_b + 1e-9 < base_early,
        }
        zero_results.append(row)
        print(
            f"  {name:14s} early={early_b:.4f}(Δ{early_b-base_early:+.4f}) "
            f"mid={mid_b:.4f}(Δ{mid_b-base_mid:+.4f}) all={all_b:.4f}"
        )

    # --- 2) retrain early RF without each early-relevant group ---
    print("\n=== retrain early RF (drop group) ===")
    train_early = train_df[train_df["minute_bin"] <= 8].copy()
    for c in early_cols:
        if c not in train_early.columns:
            train_early[c] = 0.0
        if c not in hold_df.columns:
            hold_df[c] = 0.0

    base_model = _fit_early_rf(train_early[early_cols], train_early["label"])
    p_e_base = base_model.predict_proba(hold_df[early_cols])[:, 1]
    early_mask = np.array([phase_of_minute(m) == "early" for m in minutes])
    retrain_base_early = float(brier_score_loss(y[early_mask], p_e_base[early_mask])) if early_mask.any() else float("nan")
    print(f"  early-RF baseline early_brier={retrain_base_early:.4f}")

    retrain_results = []
    drop_candidates = []
    early_groups = {
        k: v
        for k, v in FE_GROUPS.items()
        if k in ("accel", "momentum", "lane_crush", "roll4", "win35", "concentration")
    }
    for name, cols in early_groups.items():
        present = [c for c in cols if c in early_cols]
        if not present:
            continue
        keep_cols = [c for c in early_cols if c not in present]
        model = _fit_early_rf(train_early[keep_cols], train_early["label"])
        p = model.predict_proba(hold_df[keep_cols])[:, 1]
        early_b = float(brier_score_loss(y[early_mask], p[early_mask])) if early_mask.any() else float("nan")
        delta = early_b - retrain_base_early
        # if dropping improves early Brier meaningfully, mark drop
        suggest_drop = early_b + 0.001 < retrain_base_early
        if suggest_drop:
            drop_candidates.append(name)
        retrain_results.append(
            {
                "group": name,
                "cols": present,
                "early_brier": early_b,
                "delta_early": delta,
                "suggest_drop": suggest_drop,
            }
        )
        print(f"  drop {name:14s} early={early_b:.4f} (Δ{delta:+.4f}) drop={suggest_drop}")

    # recommended early set
    drop_cols = []
    for name in drop_candidates:
        drop_cols.extend(FE_GROUPS[name])
    early_recommended = [c for c in early_cols if c not in drop_cols]

    report = {
        "holdout_battles": [b["battle_id"] for b in holdout],
        "inference_zero_out": {
            "baseline_early_brier": base_early,
            "baseline_mid_brier": base_mid,
            "baseline_all_brier": base_all,
            "groups": zero_results,
        },
        "retrain_early_rf": {
            "baseline_early_brier": retrain_base_early,
            "groups": retrain_results,
            "suggest_drop_groups": drop_candidates,
            "early_feature_columns_recommended": early_recommended,
        },
        "note": (
            "Zero-out Δ>0 means group helps current model. "
            "Retrain suggest_drop means removing group lowers early Brier by >0.001."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsuggest drop groups: {drop_candidates}")
    print(f"early recommended ({len(early_recommended)}): {early_recommended}")
    print(f"saved: {REPORT_PATH}")
    return report


def main() -> None:
    run_ablation(holdout_n=8)


if __name__ == "__main__":
    main()
