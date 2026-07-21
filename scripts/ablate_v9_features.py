"""
V9/V10 解释型特征组消融：固定真实 holdout，看去掉某组对 early Brier 的影响。

输出：
  data/realtime/ablation_v10_report.json
  推荐的 early 特征子集（伤 early 或无增益的组剔除）

运行：
    python scripts/ablate_v9_features.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtest_realtime_v9 import list_labeled_real_battles, select_holdout
from kpl_official_core import FeatureBuilder, REALTIME_DIR, evaluate_phase_metrics, phase_of_minute
from train_realtime_model_v9 import (
    V9_FEATURE_COLUMNS,
    battle_timestamp,
    build_causal_team_winrate_maps,
    extract_real_snapshots,
)


REPORT_PATH = REALTIME_DIR / "ablation_v10_report.json"

FEATURE_GROUPS = {
    "tempo": ["tempo_swing_score", "gold_diff_velocity", "gold_diff_delta"],
    "resource": [
        "objective_value_score",
        "resource_control_rate",
        "tyrant_diff",
        "dark_tyrant_diff",
        "lord_diff",
        "prophet_diff",
        "shadow_diff",
        "storm_diff",
    ],
    "carry_share": ["carry_gold_share_diff", "carry_dominance"],
    "damage_conversion": [
        "damage_conversion_diff",
        "hurt_diff_mid",
        "hurt_diff_support",
        "hurt_diff_jungle",
        "hurt_diff_top",
        "hurt_diff_adc",
    ],
    "map_pressure": ["map_pressure_index", "lane_dominance_max", "late_game_scaling_proxy"],
}

# Stable core always kept for early expert
EARLY_CORE = [
    "gold_diff_per_min",
    "gold_ratio",
    "kill_diff_per_min",
    "kill_rate",
    "assist_diff_per_min",
    "death_diff",
    "kda_diff",
    "tower_diff",
    "minute_bin",
    "team_winrate_diff",
    "gold_diff_mid",
    "gold_diff_support",
    "gold_diff_jungle",
    "gold_diff_top",
    "gold_diff_adc",
]


def _fit_early_lr(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=0.25, random_state=42)),
        ]
    )
    model.fit(x, y)
    return model


def _early_brier(model, x: pd.DataFrame, y: pd.Series, minutes: np.ndarray) -> float:
    if len(x) == 0:
        return 1.0
    p = model.predict_proba(x)[:, 1]
    mask = np.array([phase_of_minute(m) == "early" for m in minutes])
    if not np.any(mask):
        return float(brier_score_loss(y, p))
    return float(brier_score_loss(y[mask], p[mask]))


def run_ablation(holdout_n: int = 8) -> dict:
    battles_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "battles.csv"
    battles = pd.read_csv(battles_path)
    per_battle_wr, fallback_wr = build_causal_team_winrate_maps(battles)

    labeled = list_labeled_real_battles()
    holdout, _ = select_holdout(labeled, holdout_n)
    holdout_ids = {b["battle_id"] for b in holdout}

    real_df = extract_real_snapshots(per_battle_wr, fallback_wr, exclude_ids=holdout_ids).fillna(0)
    # Build holdout feature rows with FeatureBuilder for fairness
    hold_rows = []
    for battle in holdout:
        history = []
        seen = set()
        camp1 = battle["snapshots"][0].get("camp1_team", "")
        camp2 = battle["snapshots"][0].get("camp2_team", "")
        team_wr = {
            camp1: fallback_wr.get(camp1, 0.5),
            camp2: fallback_wr.get(camp2, 0.5),
        }
        label = int(battle["win_camp"] == 1)
        for snap in battle["snapshots"]:
            minute = max(int(snap.get("minute", 0)), 1)
            if minute in seen:
                history.append(snap)
                continue
            seen.add(minute)
            feats = FeatureBuilder(team_wr).build(snap, history)
            hold_rows.append({"minute_bin": minute, "label": label, **feats})
            history.append(snap)
    hold_df = pd.DataFrame(hold_rows).fillna(0)

    cols = [c for c in V9_FEATURE_COLUMNS if c in real_df.columns and c in hold_df.columns]
    train_early = real_df[real_df["minute_bin"] <= 8]
    x_tr, y_tr = train_early[cols], train_early["label"]
    x_ho, y_ho = hold_df[cols], hold_df["label"]
    minutes = hold_df["minute_bin"].values

    baseline = _fit_early_lr(x_tr, y_tr)
    base_early = _early_brier(baseline, x_ho, y_ho, minutes)
    base_all = float(brier_score_loss(y_ho, baseline.predict_proba(x_ho)[:, 1]))

    results = []
    drop_groups = []
    for name, group_cols in FEATURE_GROUPS.items():
        present = [c for c in group_cols if c in cols]
        if not present:
            continue
        # Zero-out group at inference (leave-one-group-out effect)
        x_abl = x_ho.copy()
        for c in present:
            x_abl[c] = 0.0
        early_b = _early_brier(baseline, x_abl, y_ho, minutes)
        all_b = float(brier_score_loss(y_ho, baseline.predict_proba(x_abl)[:, 1]))
        delta_early = early_b - base_early
        # Positive delta = removing group helps (baseline used those features badly)
        # For ablation of "keeping" group: if zeroing improves early (delta_early < 0), group hurts → drop
        keep = delta_early > -0.002  # keep if removing doesn't improve early by >0.002
        # Actually: if zeroing lowers Brier (early_b < base_early), features hurt → drop from early
        if early_b + 1e-9 < base_early:
            drop_groups.append(name)
            keep = False
        else:
            keep = True
        results.append(
            {
                "group": name,
                "cols": present,
                "early_brier_zeroed": early_b,
                "all_brier_zeroed": all_b,
                "delta_early": early_b - base_early,
                "keep_in_early": keep,
            }
        )
        print(
            f"  {name:18s} early={early_b:.4f} (Δ{early_b - base_early:+.4f}) "
            f"all={all_b:.4f} keep={keep}"
        )

    early_features = list(EARLY_CORE)
    for name, group_cols in FEATURE_GROUPS.items():
        if name in drop_groups:
            continue
        for c in group_cols:
            if c in cols and c not in early_features:
                # Only add non-noisy groups that ablation kept; still prefer stable
                if name in ("carry_share",) and c == "carry_dominance":
                    early_features.append(c)
                elif name not in drop_groups and c in EARLY_CORE:
                    pass
    # Plan default: exclude tempo/resource/map/damage from early unless ablation insists keep AND in core
    # Final early set = EARLY_CORE + carry_dominance if not dropped
    if "carry_share" not in drop_groups and "carry_dominance" not in early_features:
        early_features.append("carry_dominance")

    # Deduplicate preserve order
    seen = set()
    early_final = []
    for c in early_features:
        if c in cols and c not in seen:
            seen.add(c)
            early_final.append(c)

    report = {
        "baseline_early_brier": base_early,
        "baseline_all_brier": base_all,
        "holdout_battles": [b["battle_id"] for b in holdout],
        "groups": results,
        "drop_from_early": drop_groups,
        "early_feature_columns": early_final,
        "note": "Groups whose zero-out lowers early Brier are dropped from V10 early expert.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nearly features ({len(early_final)}): {early_final}")
    print(f"drop groups: {drop_groups}")
    print(f"saved: {REPORT_PATH}")
    return report


def main():
    print("=== V10 feature-group ablation (early Brier) ===")
    run_ablation(holdout_n=8)


if __name__ == "__main__":
    main()
