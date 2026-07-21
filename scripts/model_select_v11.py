"""
V11 模型选择 / 轻量调参（不引入新模型族）。

在 clean holdout 上比较 early / midlate 组合与少量超参，主指标：
  all Brier（优先）→ early Brier → mid Brier
约束：early<=0.110, mid<=0.065, all<=0.085（相对当前 FE 基线不显著变差）

运行：
    python scripts/model_select_v11.py
"""

from __future__ import annotations

import json
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kpl_official_core import REALTIME_DIR, evaluate_phase_metrics
from train_realtime_model_v9 import (
    EARLY_FEATURE_COLUMNS,
    HOLDOUT_REAL_BATTLES,
    PROCESSED_DIR,
    V9_FEATURE_COLUMNS,
    add_noise_augmentation,
    aggregate_position_data,
    build_causal_team_winrate_maps,
    choose_midlate_calibrator,
    collect_holdout_blend_probs,
    compute_sample_weights,
    detect_swing_sets,
    evaluate,
    extract_real_snapshots,
    finalize_holdout_probs,
    fit_calibrator,
    list_labeled_real_battles,
    load_early_feature_columns,
    simulate_from_final_battles,
)


REPORT_PATH = REALTIME_DIR / "model_select_v11_report.json"

# Current FE+clean baseline (backtest)
BASELINE = {
    "early_brier": 0.0771,
    "mid_brier": 0.0362,
    "late_brier": 0.0576,
    "all_brier": 0.0567,
}


def _lr(c: float) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, C=c, random_state=42))]
    )


def _rf(n: int, depth: int, leaf: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n, max_depth=depth, min_samples_leaf=leaf, random_state=42, n_jobs=-1
    )


def _et(n: int, depth: int, leaf: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n, max_depth=depth, min_samples_leaf=leaf, random_state=42, n_jobs=-1
    )


def _gbdt(n: int, depth: int, lr: float, leaf: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.82,
        min_samples_leaf=leaf,
        random_state=42,
    )


def build_early(name: str, x, y):
    configs = {
        "early_baseline": (
            [("LR", _lr(0.22)), ("RF", _rf(180, 6, 14))],
            [1.4, 1.0],
        ),
        "early_lr_soft": (
            [("LR", _lr(0.35)), ("RF", _rf(180, 6, 14))],
            [1.0, 1.0],
        ),
        "early_rf_focus": (
            [("LR", _lr(0.22)), ("RF", _rf(220, 8, 10))],
            [1.0, 1.4],
        ),
        "early_rf_only": ([("RF", _rf(220, 7, 12))], [1.0]),
        "early_lr_only": ([("LR", _lr(0.28))], [1.0]),
        "early_shallow_rf": (
            [("LR", _lr(0.22)), ("RF", _rf(160, 5, 18))],
            [1.5, 1.0],
        ),
    }
    estimators, weights = configs[name]
    fitted = []
    for est_name, model in estimators:
        m = deepcopy(model)
        m.fit(x, y)
        fitted.append((est_name, m))
    if len(fitted) == 1:
        return fitted[0][1]
    vote = VotingClassifier(estimators=fitted, voting="soft", weights=weights, n_jobs=-1)
    vote.fit(x, y)
    return vote


def build_midlate(name: str, x_train, y_train, x_test, y_test):
    specs = {
        "mid_baseline": {
            "LR": _lr(0.35),
            "RF": _rf(360, 10, 10),
            "ET": _et(480, 11, 8),
            "GBDT": _gbdt(260, 4, 0.045, 14),
        },
        "mid_drop_lr": {
            "RF": _rf(360, 10, 10),
            "ET": _et(480, 11, 8),
            "GBDT": _gbdt(260, 4, 0.045, 14),
        },
        "mid_drop_gbdt": {
            "LR": _lr(0.35),
            "RF": _rf(360, 10, 10),
            "ET": _et(480, 11, 8),
        },
        "mid_rf_et": {
            "RF": _rf(400, 10, 10),
            "ET": _et(520, 11, 8),
        },
        "mid_rf_strong": {
            "LR": _lr(0.35),
            "RF": _rf(500, 12, 8),
            "ET": _et(480, 11, 8),
            "GBDT": _gbdt(260, 4, 0.045, 14),
        },
        "mid_gbdt_soft": {
            "LR": _lr(0.35),
            "RF": _rf(360, 10, 10),
            "ET": _et(480, 11, 8),
            "GBDT": _gbdt(200, 3, 0.05, 16),
        },
        "mid_equal_vote": {
            "LR": _lr(0.35),
            "RF": _rf(360, 10, 10),
            "ET": _et(480, 11, 8),
            "GBDT": _gbdt(260, 4, 0.045, 14),
        },
    }
    models = specs[name]
    fitted = []
    scores = {}
    for n, model in models.items():
        m = deepcopy(model)
        m.fit(x_train, y_train)
        scores[n] = evaluate(y_test, m.predict_proba(x_test)[:, 1])
        fitted.append((n, m))
    if name == "mid_equal_vote":
        weights = [1.0] * len(fitted)
    else:
        weights = [
            max(0.2, scores[n]["auc"] * 2 + scores[n]["accuracy"] - scores[n]["brier"]) for n, _ in fitted
        ]
    if len(fitted) == 1:
        return fitted[0][1], scores
    vote = VotingClassifier(estimators=fitted, voting="soft", weights=weights, n_jobs=-1)
    vote.fit(x_train, y_train)
    scores["Voting"] = evaluate(y_test, vote.predict_proba(x_test)[:, 1])
    return vote, scores


def prepare_data():
    battles = pd.read_csv(PROCESSED_DIR / "battles.csv")
    players = pd.read_csv(PROCESSED_DIR / "players.csv")
    per_battle_wr, fallback_wr = build_causal_team_winrate_maps(battles)
    early_cols = load_early_feature_columns()
    labeled = list_labeled_real_battles(from_raw=False)
    n_holdout = min(HOLDOUT_REAL_BATTLES, max(1, len(labeled) // 4)) if labeled else 0
    holdout_battles = labeled[-n_holdout:] if n_holdout else []
    holdout_ids = {b["battle_id"] for b in holdout_battles}
    train_labeled = [b for b in labeled if b["battle_id"] not in holdout_ids]
    comeback_ids, swing_ids = detect_swing_sets(train_labeled)

    pos_df = aggregate_position_data(players)
    sim_df = simulate_from_final_battles(battles, pos_df, per_battle_wr, fallback_wr, exclude_ids=holdout_ids)
    real_df = extract_real_snapshots(per_battle_wr, fallback_wr, exclude_ids=holdout_ids, from_raw=False)
    train_base = pd.concat([sim_df, real_df], ignore_index=True).fillna(0)
    for c in V9_FEATURE_COLUMNS:
        if c not in train_base.columns:
            train_base[c] = 0.0
    train_base = train_base[["battle_id", "label", "is_real", *V9_FEATURE_COLUMNS]].copy()
    train_aug = add_noise_augmentation(train_base, repeats=2)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=42)
    train_idx, test_idx = next(splitter.split(train_aug, train_aug["label"], train_aug["battle_id"]))
    train = train_aug.iloc[train_idx].copy()
    test = train_aug.iloc[test_idx].copy()
    w_train = compute_sample_weights(train, comeback_ids, swing_ids)
    rng = np.random.default_rng(42)
    probs = w_train / max(w_train.sum(), 1e-9)
    resample_idx = rng.choice(len(train), size=len(train), replace=True, p=probs)
    train_fit = train.iloc[resample_idx].reset_index(drop=True)

    early_mask = train_fit["minute_bin"] <= 8
    train_early = train_fit[early_mask] if early_mask.sum() >= 50 else train_fit
    return {
        "early_cols": early_cols,
        "fallback_wr": fallback_wr,
        "holdout_battles": holdout_battles,
        "train_fit": train_fit,
        "train_early": train_early,
        "test": test,
    }


def score_holdout(model_early, early_cols, model_midlate, fallback_wr, holdout_battles) -> dict:
    holdout_raw = collect_holdout_blend_probs(
        model_early, early_cols, model_midlate, V9_FEATURE_COLUMNS, fallback_wr, holdout_battles
    )
    if holdout_raw.empty:
        return {"all": {"brier": 1.0}, "early": {"brier": 1.0}, "mid": {"brier": 1.0}, "late": {"brier": 1.0}}
    early_rows = holdout_raw[holdout_raw["minute_bin"] <= 8]
    cal_early = (
        fit_calibrator(early_rows["prob_early"].values, early_rows["label"].values, method="isotonic")
        if len(early_rows)
        else fit_calibrator(holdout_raw["prob_early"].values, holdout_raw["label"].values, method="platt")
    )
    early_clip = (0.02, 0.98)
    cal_midlate, mid_name, _ = choose_midlate_calibrator(holdout_raw, cal_early, early_clip)
    probs = finalize_holdout_probs(holdout_raw, cal_early, cal_midlate, early_clip, use_gold_guard=True)
    metrics = evaluate_phase_metrics(holdout_raw["minute"].values, holdout_raw["label"].values, probs)
    metrics["_midlate_cal"] = mid_name
    return metrics


def passes_gates(m: dict) -> bool:
    return (
        m.get("early", {}).get("brier", 1) <= 0.110
        and m.get("mid", {}).get("brier", 1) <= 0.065
        and m.get("all", {}).get("brier", 1) <= 0.085
    )


def rank_key(m: dict) -> tuple:
    return (
        m.get("all", {}).get("brier", 1.0),
        m.get("early", {}).get("brier", 1.0),
        m.get("mid", {}).get("brier", 1.0),
    )


def main():
    print("=== V11 model select (clean holdout) ===")
    data = prepare_data()
    early_cols = [c for c in data["early_cols"] if c in V9_FEATURE_COLUMNS] or list(EARLY_FEATURE_COLUMNS)
    train_fit = data["train_fit"]
    train_early = data["train_early"]
    test = data["test"]
    x_e, y_e = train_early[early_cols], train_early["label"]
    x_tr, y_tr = train_fit[V9_FEATURE_COLUMNS], train_fit["label"]
    x_te, y_te = test[V9_FEATURE_COLUMNS], test["label"]

    early_names = [
        "early_baseline",
        "early_lr_soft",
        "early_rf_focus",
        "early_rf_only",
        "early_lr_only",
        "early_shallow_rf",
    ]
    mid_names = [
        "mid_baseline",
        "mid_drop_lr",
        "mid_drop_gbdt",
        "mid_rf_et",
        "mid_rf_strong",
        "mid_gbdt_soft",
        "mid_equal_vote",
    ]

    # Stage 1: fix early=baseline, sweep midlate
    print("\n-- stage1: midlate sweep (early=baseline) --")
    early0 = build_early("early_baseline", x_e, y_e)
    mid_results = []
    for mid_name in mid_names:
        print(f"  {mid_name} ...", flush=True)
        mid_model, sec = build_midlate(mid_name, x_tr, y_tr, x_te, y_te)
        metrics = score_holdout(early0, early_cols, mid_model, data["fallback_wr"], data["holdout_battles"])
        row = {
            "early": "early_baseline",
            "midlate": mid_name,
            "holdout": {k: metrics[k] for k in ("all", "early", "mid", "late") if k in metrics},
            "midlate_cal": metrics.get("_midlate_cal"),
            "secondary_auc": {k: v.get("auc") for k, v in sec.items()},
            "gates_ok": passes_gates(metrics),
            "delta_all_vs_baseline": metrics.get("all", {}).get("brier", 1) - BASELINE["all_brier"],
        }
        mid_results.append(row)
        print(
            f"    all={row['holdout']['all']['brier']:.4f} early={row['holdout']['early']['brier']:.4f} "
            f"mid={row['holdout']['mid']['brier']:.4f} gates={row['gates_ok']}"
        )

    best_mid = min(mid_results, key=lambda r: rank_key(r["holdout"]))
    best_mid_name = best_mid["midlate"]
    print(f"best midlate so far: {best_mid_name}")

    # Stage 2: fix best midlate, sweep early
    print(f"\n-- stage2: early sweep (midlate={best_mid_name}) --")
    mid_best_model, _ = build_midlate(best_mid_name, x_tr, y_tr, x_te, y_te)
    early_results = []
    for early_name in early_names:
        print(f"  {early_name} ...", flush=True)
        early_model = build_early(early_name, x_e, y_e)
        metrics = score_holdout(
            early_model, early_cols, mid_best_model, data["fallback_wr"], data["holdout_battles"]
        )
        row = {
            "early": early_name,
            "midlate": best_mid_name,
            "holdout": {k: metrics[k] for k in ("all", "early", "mid", "late") if k in metrics},
            "midlate_cal": metrics.get("_midlate_cal"),
            "gates_ok": passes_gates(metrics),
            "delta_all_vs_baseline": metrics.get("all", {}).get("brier", 1) - BASELINE["all_brier"],
        }
        early_results.append(row)
        print(
            f"    all={row['holdout']['all']['brier']:.4f} early={row['holdout']['early']['brier']:.4f} "
            f"mid={row['holdout']['mid']['brier']:.4f} gates={row['gates_ok']}"
        )

    all_rows = mid_results + [r for r in early_results if not (r["early"] == "early_baseline" and r["midlate"] == best_mid_name)]
    # include the combo early_baseline+best_mid already in mid_results; add other early combos
    all_rows = mid_results + [r for r in early_results if r["early"] != "early_baseline"]

    gated = [r for r in all_rows if r["gates_ok"]]
    pool = gated or all_rows
    winner = min(pool, key=lambda r: rank_key(r["holdout"]))
    improved = winner["holdout"]["all"]["brier"] < BASELINE["all_brier"] - 0.001

    report = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": BASELINE,
        "holdout_battles": [b["battle_id"] for b in data["holdout_battles"]],
        "early_feature_n": len(early_cols),
        "midlate_feature_n": len(V9_FEATURE_COLUMNS),
        "stage1_midlate": mid_results,
        "stage2_early": early_results,
        "winner": winner,
        "improved_vs_baseline": improved,
        "recommend_apply": bool(improved and winner["gates_ok"]),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== winner ===")
    print(json.dumps(winner, ensure_ascii=False, indent=2))
    print(f"improved_vs_baseline={improved} recommend_apply={report['recommend_apply']}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
