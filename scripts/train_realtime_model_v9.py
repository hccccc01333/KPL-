"""
KPL 官方平台 V10 模型训练（Early 稳定 + 翻盘稳健）

- 因果战队先验；real/sim 权重；sim 后期降权
- early / midlate 双专家 + 分钟过渡混合
- 分阶段 isotonic 校准 + early 概率 clip
- 真实翻盘局加权；去掉 early label-flip
- 主指标：真实 holdout early/mid/late Brier

运行：
    python scripts/train_realtime_model_v9.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kpl_official_core import (
    FeatureBuilder,
    MODEL_DIR,
    POSITIONS,
    RAW_DIR,
    REALTIME_DIR,
    apply_gold_consistency_guard,
    classify_battle_swing,
    evaluate_phase_metrics,
    load_battle_jsons,
    phase_of_minute,
    resolve_snapshot_root,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_STATUS_FILE = REALTIME_DIR / "train_status.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

V9_FEATURE_COLUMNS = [
    "gold_diff_per_min",
    "gold_ratio",
    "kill_diff_per_min",
    "kill_rate",
    "assist_diff_per_min",
    "death_diff",
    "kda_diff",
    "tower_diff",
    "minute_bin",
    "gold_diff_delta",
    "gold_diff_velocity",
    "gold_diff_accel",
    "kill_momentum_diff",
    "lane_crush_count",
    "tyrant_diff",
    "dark_tyrant_diff",
    "prophet_diff",
    "shadow_diff",
    "storm_diff",
    "gold_diff_mid",
    "gold_diff_support",
    "gold_diff_jungle",
    "gold_diff_top",
    "gold_diff_adc",
    "hurt_diff_mid",
    "hurt_diff_support",
    "hurt_diff_jungle",
    "hurt_diff_top",
    "hurt_diff_adc",
    "kill_diff_jungle",
    "kill_diff_adc",
    "death_diff_jungle",
    "death_diff_adc",
    "behurt_diff_top",
    "behurt_diff_support",
    "hurt_conc_diff",
    "behurt_conc_diff",
    "gold_diff_roll4",
    "gold_diff_roll4_per_min",
    "gold_diff_roll10",
    "gold_diff_roll10_per_min",
    "gold_diff_jungle_roll4",
    "gold_diff_adc_roll4",
    "win35_kill_diff",
    "win35_death_diff",
    "win35_hurt_diff",
    "win911_kill_diff",
    "win911_death_diff",
    "win911_hurt_diff",
    "obj_tower_convert",
    "carry_dominance",
    "team_winrate_diff",
    "objective_value_score",
    "lane_dominance_max",
    "map_pressure_index",
    "resource_control_rate",
    "carry_gold_share_diff",
    "damage_conversion_diff",
    "tempo_swing_score",
    "late_game_scaling_proxy",
]

MINUTE_BINS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25]
SCALING_ALPHA = {
    "gold": 1.0,
    "kill": 1.35,
    "assist": 1.25,
    "death": 1.35,
    "tower": 1.55,
    "objective": 1.25,
    "hurt": 1.12,
}
HOLDOUT_REAL_BATTLES = 8
REAL_WEIGHT = 2.5
SIM_WEIGHT = 0.55
SIM_LATE_WEIGHT = 0.18  # minute_bin >= 10
COMEBACK_EARLY_MULT = 1.3
COMEBACK_MIDLATE_MULT = 2.0
SWING_MID_MULT = 1.8
MID_MINUTE_MULT = 1.35
BLEND_START = 6.0
BLEND_END = 9.0
EARLY_PROB_CLIP = (0.30, 0.70)

# Early expert: avoid late-only window features (win911) and storm-heavy proxies
EARLY_FEATURE_COLUMNS = [
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
    "carry_dominance",
    "gold_diff_accel",
    "kill_momentum_diff",
    "lane_crush_count",
    "gold_diff_roll4",
    "gold_diff_roll4_per_min",
    "gold_diff_jungle_roll4",
    "gold_diff_adc_roll4",
    "win35_kill_diff",
    "win35_death_diff",
    "hurt_conc_diff",
]


def load_early_feature_columns() -> list[str]:
    """Prefer curated EARLY_FEATURE_COLUMNS; ablation file only if still valid subset."""
    path = REALTIME_DIR / "ablation_v10_report.json"
    base = [c for c in EARLY_FEATURE_COLUMNS if c in V9_FEATURE_COLUMNS]
    if path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            cols = [c for c in report.get("early_feature_columns", []) if c in V9_FEATURE_COLUMNS]
            # Only reuse ablation if it covers most of the new early set; else use base
            if cols and len(cols) >= max(10, len(base) - 5) and set(base).issubset(set(cols) | set(base)):
                # merge: ablation stables + new early extras
                merged = list(dict.fromkeys(cols + base))
                return [c for c in merged if c in V9_FEATURE_COLUMNS]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return base


def midlate_blend_weight(minute: float) -> float:
    """0=all early expert, 1=all midlate expert; linear ramp BLEND_START→BLEND_END."""
    m = float(minute)
    if m <= BLEND_START:
        return 0.0
    if m >= BLEND_END:
        return 1.0
    return (m - BLEND_START) / (BLEND_END - BLEND_START)


def battle_timestamp(battle_id: str) -> int:
    try:
        return int(str(battle_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


def write_train_status(payload: dict) -> None:
    TRAIN_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    TRAIN_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_causal_team_winrate_maps(battles: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per-battle priors using only earlier matches; plus final prior for serving."""
    frame = battles.copy()
    frame["_ts"] = frame["battle_id"].map(battle_timestamp)
    frame = frame.sort_values("_ts").reset_index(drop=True)

    wins: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    per_battle: dict[str, dict[str, float]] = {}

    def smooth(team: str) -> float:
        c = counts[team]
        return (wins[team] + 0.5 * 6) / (c + 6) if team else 0.5

    for _, b in frame.iterrows():
        bid = str(b["battle_id"])
        t1 = str(b.get("camp1_team_name", "") or "")
        t2 = str(b.get("camp2_team_name", "") or "")
        per_battle[bid] = {t1: smooth(t1), t2: smooth(t2)}
        if int(b.get("win_camp", 0) or 0) == 1:
            wins[t1] += 1
            counts[t1] += 1
            counts[t2] += 1
        elif int(b.get("win_camp", 0) or 0) == 2:
            wins[t2] += 1
            counts[t2] += 1
            counts[t1] += 1

    final = {team: (wins[team] + 0.5 * 6) / (counts[team] + 6) for team in counts}
    return per_battle, final


def team_wr_for_battle(
    battle_id: str,
    camp1: str,
    camp2: str,
    per_battle: dict[str, dict[str, float]],
    fallback: dict[str, float],
) -> dict[str, float]:
    prior = per_battle.get(str(battle_id), {})
    return {
        camp1: prior.get(camp1, fallback.get(camp1, 0.5)),
        camp2: prior.get(camp2, fallback.get(camp2, 0.5)),
    }


def aggregate_position_data(players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for battle_id, grp in players.groupby("battle_id"):
        row = {"battle_id": battle_id}
        for pos_id, pos_name in POSITIONS.items():
            c1 = grp[(grp["camp"] == 1) & (grp["position"] == pos_id)]
            c2 = grp[(grp["camp"] == 2) & (grp["position"] == pos_id)]
            row[f"c1_gold_{pos_name}"] = c1["gold"].sum() if len(c1) else 0
            row[f"c2_gold_{pos_name}"] = c2["gold"].sum() if len(c2) else 0
            hurt_c = "hurt_to_hero_total" if "hurt_to_hero_total" in grp.columns else None
            behurt_c = "be_hurt_by_hero_total" if "be_hurt_by_hero_total" in grp.columns else None
            row[f"c1_hurt_{pos_name}"] = c1[hurt_c].sum() if hurt_c and len(c1) else 0
            row[f"c2_hurt_{pos_name}"] = c2[hurt_c].sum() if hurt_c and len(c2) else 0
            row[f"c1_behurt_{pos_name}"] = c1[behurt_c].sum() if behurt_c and len(c1) else 0
            row[f"c2_behurt_{pos_name}"] = c2[behurt_c].sum() if behurt_c and len(c2) else 0
            for stat, col in (("kill", "kill_num"), ("death", "death_num"), ("assist", "assist_num")):
                row[f"c1_{stat}_{pos_name}"] = c1[col].sum() if col in grp.columns and len(c1) else 0
                row[f"c2_{stat}_{pos_name}"] = c2[col].sum() if col in grp.columns and len(c2) else 0
            row[f"c1_level_{pos_name}"] = 0
            row[f"c2_level_{pos_name}"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def scaled(value: float, ratio: float, stat: str) -> float:
    if pd.isna(value):
        value = 0
    return float(value) * (ratio ** SCALING_ALPHA.get(stat, 1.0))


def simulate_from_final_battles(
    battles: pd.DataFrame,
    pos_df: pd.DataFrame,
    per_battle_wr: dict[str, dict[str, float]],
    fallback_wr: dict[str, float],
    exclude_ids: set[str] | None = None,
) -> pd.DataFrame:
    exclude_ids = exclude_ids or set()
    merged = battles.merge(pos_df, on="battle_id", how="left").fillna(0)
    rows = []
    for _, b in merged.iterrows():
        bid = str(b["battle_id"])
        if bid in exclude_ids:
            continue
        duration = float(b.get("game_duration", 0) or 0)
        if duration > 1000:
            duration /= 1000
        if duration <= 180:
            continue
        camp1 = str(b.get("camp1_team_name", "") or "")
        camp2 = str(b.get("camp2_team_name", "") or "")
        team_wr = team_wr_for_battle(bid, camp1, camp2, per_battle_wr, fallback_wr)
        history = []
        for minute in MINUTE_BINS:
            sec = minute * 60
            if sec >= duration:
                continue
            ratio = sec / duration
            snap = {
                "battle_id": bid,
                "time_sec": sec,
                "minute": minute,
                "minute_bin": minute,
                "camp1_team": camp1,
                "camp2_team": camp2,
                "camp1_gold": scaled(b.get("camp1_gold", 0), ratio, "gold"),
                "camp2_gold": scaled(b.get("camp2_gold", 0), ratio, "gold"),
                "camp1_kill": scaled(b.get("camp1_kill_num", 0), ratio, "kill"),
                "camp2_kill": scaled(b.get("camp2_kill_num", 0), ratio, "kill"),
                "camp1_assist": scaled(b.get("camp1_assist_num", 0), ratio, "assist"),
                "camp2_assist": scaled(b.get("camp2_assist_num", 0), ratio, "assist"),
                "camp1_death": scaled(b.get("camp1_death_num", 0), ratio, "death"),
                "camp2_death": scaled(b.get("camp2_death_num", 0), ratio, "death"),
                "camp1_tower": scaled(b.get("camp1_push_tower_num", 0), ratio, "tower"),
                "camp2_tower": scaled(b.get("camp2_push_tower_num", 0), ratio, "tower"),
                "camp1_tyrant": scaled(b.get("camp1_kill_tyrant_num", 0), ratio, "objective"),
                "camp2_tyrant": scaled(b.get("camp2_kill_tyrant_num", 0), ratio, "objective"),
                "camp1_dark_tyrant": scaled(b.get("camp1_kill_dark_tyrant_num", 0), ratio, "objective"),
                "camp2_dark_tyrant": scaled(b.get("camp2_kill_dark_tyrant_num", 0), ratio, "objective"),
                "camp1_lord": scaled(b.get("camp1_kill_big_dragon_num", 0), ratio, "objective"),
                "camp2_lord": scaled(b.get("camp2_kill_big_dragon_num", 0), ratio, "objective"),
                "camp1_prophet": scaled(b.get("camp1_kill_prophet_dragon_num", 0), ratio, "objective"),
                "camp2_prophet": scaled(b.get("camp2_kill_prophet_dragon_num", 0), ratio, "objective"),
                "camp1_shadow": scaled(b.get("camp1_kill_shadow_dragon_num", 0), ratio, "objective"),
                "camp2_shadow": scaled(b.get("camp2_kill_shadow_dragon_num", 0), ratio, "objective"),
                "camp1_storm": scaled(b.get("camp1_kill_storm_dragon_king_num", 0), ratio, "objective"),
                "camp2_storm": scaled(b.get("camp2_kill_storm_dragon_king_num", 0), ratio, "objective"),
            }
            for _, pos_name in POSITIONS.items():
                snap[f"c1_gold_{pos_name}"] = scaled(b.get(f"c1_gold_{pos_name}", 0), ratio, "gold")
                snap[f"c2_gold_{pos_name}"] = scaled(b.get(f"c2_gold_{pos_name}", 0), ratio, "gold")
                snap[f"c1_hurt_{pos_name}"] = scaled(b.get(f"c1_hurt_{pos_name}", 0), ratio, "hurt")
                snap[f"c2_hurt_{pos_name}"] = scaled(b.get(f"c2_hurt_{pos_name}", 0), ratio, "hurt")
                snap[f"c1_behurt_{pos_name}"] = scaled(b.get(f"c1_behurt_{pos_name}", 0), ratio, "hurt")
                snap[f"c2_behurt_{pos_name}"] = scaled(b.get(f"c2_behurt_{pos_name}", 0), ratio, "hurt")
                snap[f"c1_kill_{pos_name}"] = scaled(b.get(f"c1_kill_{pos_name}", 0), ratio, "kill")
                snap[f"c2_kill_{pos_name}"] = scaled(b.get(f"c2_kill_{pos_name}", 0), ratio, "kill")
                snap[f"c1_death_{pos_name}"] = scaled(b.get(f"c1_death_{pos_name}", 0), ratio, "death")
                snap[f"c2_death_{pos_name}"] = scaled(b.get(f"c2_death_{pos_name}", 0), ratio, "death")
                snap[f"c1_assist_{pos_name}"] = scaled(b.get(f"c1_assist_{pos_name}", 0), ratio, "assist")
                snap[f"c2_assist_{pos_name}"] = scaled(b.get(f"c2_assist_{pos_name}", 0), ratio, "assist")
                snap[f"c1_level_{pos_name}"] = 0
                snap[f"c2_level_{pos_name}"] = 0
            feats = FeatureBuilder(team_wr).build(snap, history)
            rows.append(
                {
                    "battle_id": bid,
                    "minute_bin": minute,
                    "label": int(b["win_camp"] == 1),
                    "is_real": False,
                    **feats,
                }
            )
            history.append(snap)
    return pd.DataFrame(rows)


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
        rows.append(
            {
                "battle_id": battle_dir.name,
                "win_camp": win_camp,
                "snapshots": snaps,
                "ts": battle_timestamp(battle_dir.name),
            }
        )
    rows.sort(key=lambda x: x["ts"])
    return rows


def extract_real_snapshots(
    per_battle_wr: dict[str, dict[str, float]],
    fallback_wr: dict[str, float],
    exclude_ids: set[str] | None = None,
    *,
    from_raw: bool = False,
) -> pd.DataFrame:
    exclude_ids = exclude_ids or set()
    rows = []
    for battle in list_labeled_real_battles(from_raw=from_raw):
        bid = battle["battle_id"]
        if bid in exclude_ids:
            continue
        snaps = battle["snapshots"]
        win_camp = battle["win_camp"]
        camp1 = snaps[0].get("camp1_team", "")
        camp2 = snaps[0].get("camp2_team", "")
        team_wr = team_wr_for_battle(bid, camp1, camp2, per_battle_wr, fallback_wr)
        history = []
        seen = set()
        for snap in snaps:
            minute = max(int(snap.get("minute", 0)), 1)
            if minute in seen:
                history.append(snap)
                continue
            seen.add(minute)
            feats = FeatureBuilder(team_wr).build(snap, history)
            rows.append(
                {
                    "battle_id": bid,
                    "minute_bin": minute,
                    "label": int(win_camp == 1),
                    "is_real": True,
                    **feats,
                }
            )
            history.append(snap)
    return pd.DataFrame(rows)


def detect_comeback_battle_ids(battles: list[dict]) -> set[str]:
    """Backward-compatible: only comeback flag."""
    return {b["battle_id"] for b in battles if classify_battle_swing(b.get("snapshots") or [], int(b.get("win_camp", 0) or 0))["comeback"]}


def detect_swing_sets(battles: list[dict]) -> tuple[set[str], set[str]]:
    comebacks, swings = set(), set()
    for battle in battles:
        info = classify_battle_swing(battle.get("snapshots") or [], int(battle.get("win_camp", 0) or 0))
        bid = battle["battle_id"]
        if info["comeback"]:
            comebacks.add(bid)
        if info["swing"]:
            swings.add(bid)
    return comebacks, swings


def compute_sample_weights(
    df: pd.DataFrame,
    comeback_ids: set[str] | None = None,
    swing_ids: set[str] | None = None,
) -> np.ndarray:
    comeback_ids = comeback_ids or set()
    swing_ids = swing_ids or set()
    weights = []
    for _, row in df.iterrows():
        if bool(row.get("is_real", False)):
            w = REAL_WEIGHT
        else:
            w = SIM_WEIGHT
            if int(row.get("minute_bin", 0)) >= 10:
                w *= SIM_LATE_WEIGHT / SIM_WEIGHT
        minute = int(row.get("minute_bin", 0))
        if 9 <= minute <= 15:
            w *= MID_MINUTE_MULT
        bid = str(row.get("battle_id", ""))
        if bid in comeback_ids:
            if minute <= 8:
                w *= COMEBACK_EARLY_MULT
            else:
                w *= COMEBACK_MIDLATE_MULT
        if bid in swing_ids and 9 <= minute <= 15:
            w *= SWING_MID_MULT
        weights.append(w)
    return np.asarray(weights, dtype=float)


def add_noise_augmentation(df: pd.DataFrame, repeats: int = 2, sigma: float = 0.045) -> pd.DataFrame:
    """Noise only — no early label flip (conflicts with calibration)."""
    parts = [df.copy()]
    feature_cols = [c for c in V9_FEATURE_COLUMNS if c in df.columns]
    rng = np.random.default_rng(42)
    for _ in range(repeats):
        noisy = df.copy()
        for col in feature_cols:
            std = float(noisy[col].std() or 0)
            if std > 0:
                noisy[col] = noisy[col] + rng.normal(0, std * sigma, len(noisy))
        parts.append(noisy)
    return pd.concat(parts, ignore_index=True)


def evaluate(y_true, prob) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else 0.5,
        "accuracy": float(accuracy_score(y_true, prob >= 0.5)),
        "logloss": float(log_loss(y_true, np.clip(prob, 0.01, 0.99))),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def fit_calibrator(probs: np.ndarray, labels: np.ndarray, min_n: int = 15, method: str = "isotonic"):
    if len(probs) < min_n or len(set(labels.tolist())) < 2:
        return None
    if method == "platt":
        x = np.asarray(probs, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        lr.fit(x, y)
        return lr
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, labels)
    return cal


def apply_calibrator(calibrator, probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(probs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if hasattr(calibrator, "predict_proba") and not isinstance(calibrator, IsotonicRegression):
        return np.asarray(calibrator.predict_proba(probs.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(calibrator.predict(probs), dtype=float)


def collect_holdout_blend_probs(
    model_early,
    early_cols: list[str],
    model_midlate,
    midlate_cols: list[str],
    team_wr: dict,
    holdout_battles: list[dict],
) -> pd.DataFrame:
    rows = []
    for battle in holdout_battles:
        history: deque = deque(maxlen=40)
        seen = set()
        label = int(battle["win_camp"] == 1)
        for snap in battle["snapshots"]:
            minute_bin = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
            history.append(snap)
            if minute_bin in seen:
                continue
            seen.add(minute_bin)
            feats = FeatureBuilder(team_wr).build(snap, list(history))
            x_e = pd.DataFrame([[feats.get(c, 0.0) for c in early_cols]], columns=early_cols)
            x_m = pd.DataFrame([[feats.get(c, 0.0) for c in midlate_cols]], columns=midlate_cols)
            p_e = float(model_early.predict_proba(x_e)[0, 1])
            p_m = float(model_midlate.predict_proba(x_m)[0, 1])
            w = midlate_blend_weight(minute_bin)
            rows.append(
                {
                    "battle_id": battle["battle_id"],
                    "minute": float(snap.get("minute", minute_bin)),
                    "minute_bin": minute_bin,
                    "label": label,
                    "prob_early": p_e,
                    "prob_midlate": p_m,
                    "blend_w": w,
                    "prob_raw": (1 - w) * p_e + w * p_m,
                    "gold_ratio": float(feats.get("gold_ratio", 0.0)),
                    "gold_diff": float(
                        feats.get("gold_diff")
                        if feats.get("gold_diff") is not None
                        else feats.get("gold_diff_per_min", 0.0) * minute_bin
                    ),
                    "tyrant_diff": float(feats.get("tyrant_diff", 0.0)),
                    "dark_tyrant_diff": float(feats.get("dark_tyrant_diff", 0.0)),
                    "lord_diff": float(feats.get("lord_diff", 0.0)),
                    "storm_diff": float(feats.get("storm_diff", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def finalize_holdout_probs(
    holdout_df: pd.DataFrame,
    cal_early,
    cal_midlate,
    early_clip: tuple[float, float],
    *,
    use_gold_guard: bool = True,
) -> np.ndarray:
    if holdout_df.empty:
        return np.array([])
    p_e = apply_calibrator(cal_early, holdout_df["prob_early"].values)
    p_m = apply_calibrator(cal_midlate, holdout_df["prob_midlate"].values)
    lo, hi = early_clip
    p_e = np.clip(p_e, lo, hi)
    w = holdout_df["blend_w"].values
    blended = np.clip((1 - w) * p_e + w * p_m, 0.02, 0.98)
    if not use_gold_guard or "gold_ratio" not in holdout_df.columns:
        return blended
    out = []
    for i, p in enumerate(blended):
        row = holdout_df.iloc[i]
        feats = {
            "gold_ratio": float(row["gold_ratio"]),
            "gold_diff": float(row["gold_diff"]) if "gold_diff" in holdout_df.columns else 0.0,
            "minute_bin": float(row["minute_bin"]),
            "tyrant_diff": float(row["tyrant_diff"]) if "tyrant_diff" in holdout_df.columns else 0.0,
            "dark_tyrant_diff": float(row["dark_tyrant_diff"]) if "dark_tyrant_diff" in holdout_df.columns else 0.0,
            "lord_diff": float(row["lord_diff"]) if "lord_diff" in holdout_df.columns else 0.0,
            "storm_diff": float(row["storm_diff"]) if "storm_diff" in holdout_df.columns else 0.0,
        }
        p2, _ = apply_gold_consistency_guard(float(p), feats, enabled=True, minute=int(feats["minute_bin"]))
        out.append(p2)
    return np.clip(np.asarray(out, dtype=float), 0.02, 0.98)


def choose_midlate_calibrator(holdout_df: pd.DataFrame, cal_early, early_clip: tuple[float, float]):
    """Choose midlate calibrator by holdout mid Brier; isotonic only if mid n>=30."""
    mid_rows = holdout_df[holdout_df["minute_bin"] >= 9]
    candidates: list[tuple[str, object | None]] = [("none", None)]
    cal_platt = (
        fit_calibrator(mid_rows["prob_midlate"].values, mid_rows["label"].values, method="platt", min_n=12)
        if len(mid_rows) >= 12
        else None
    )
    if cal_platt is not None:
        candidates.append(("platt", cal_platt))
    if len(mid_rows) >= 30:
        cal_iso = fit_calibrator(
            mid_rows["prob_midlate"].values, mid_rows["label"].values, method="isotonic", min_n=30
        )
        if cal_iso is not None:
            candidates.append(("isotonic", cal_iso))

    scores: dict[str, dict] = {}
    for name, cal in candidates:
        probs = finalize_holdout_probs(holdout_df, cal_early, cal, early_clip, use_gold_guard=True)
        metrics = evaluate_phase_metrics(holdout_df["minute"].values, holdout_df["label"].values, probs)
        scores[name] = {
            "mid_brier": metrics.get("mid", {}).get("brier", 1.0),
            "early_brier": metrics.get("early", {}).get("brier", 1.0),
            "all_brier": metrics.get("all", {}).get("brier", 1.0),
            "cal": cal,
        }

    # Plan: pick by mid Brier; early/all must not clearly regress
    viable = {
        k: v
        for k, v in scores.items()
        if v["early_brier"] <= 0.12 and v["all_brier"] <= 0.12
    } or scores
    best_name = min(viable, key=lambda k: (viable[k]["mid_brier"], viable[k]["all_brier"]))
    best = scores[best_name]
    return best["cal"], best_name, {k: {kk: vv for kk, vv in v.items() if kk != "cal"} for k, v in scores.items()}


def train_v9_model(verbose: bool = True, *, from_raw: bool = False) -> dict:
    """Train V11 phase-blend model; keep function name for monitor compatibility."""
    if verbose:
        print("=== KPL V11 Early/Midlate + Gold Guard Training ===")
        root = resolve_snapshot_root(from_raw=from_raw)
        print(f"snapshot root: {root} (from_raw={from_raw})")
    battles_path = PROCESSED_DIR / "battles.csv"
    players_path = PROCESSED_DIR / "players.csv"
    if not battles_path.exists() or not players_path.exists():
        raise FileNotFoundError("缺少 data/processed/battles.csv 或 players.csv")

    battles = pd.read_csv(battles_path)
    players = pd.read_csv(players_path)
    per_battle_wr, fallback_wr = build_causal_team_winrate_maps(battles)
    early_cols = load_early_feature_columns()

    labeled = list_labeled_real_battles(from_raw=from_raw)
    n_holdout = min(HOLDOUT_REAL_BATTLES, max(1, len(labeled) // 4)) if labeled else 0
    holdout_battles = labeled[-n_holdout:] if n_holdout else []
    holdout_ids = {b["battle_id"] for b in holdout_battles}
    train_labeled = [b for b in labeled if b["battle_id"] not in holdout_ids]
    comeback_ids, swing_ids = detect_swing_sets(train_labeled)
    if verbose:
        print(
            f"real labeled: {len(labeled)}, holdout: {len(holdout_ids)}, "
            f"comeback_train: {len(comeback_ids)}, swing_train: {len(swing_ids)}"
        )
        print(f"early features ({len(early_cols)}): {early_cols}")

    pos_df = aggregate_position_data(players)
    sim_df = simulate_from_final_battles(battles, pos_df, per_battle_wr, fallback_wr, exclude_ids=holdout_ids)
    real_df = extract_real_snapshots(per_battle_wr, fallback_wr, exclude_ids=holdout_ids, from_raw=from_raw)
    if verbose:
        print(f"sim snapshots: {len(sim_df):,}, real snapshots: {len(real_df):,}")

    if real_df.empty and sim_df.empty:
        raise RuntimeError("无可用训练样本")

    train_base = pd.concat([sim_df, real_df], ignore_index=True).fillna(0)
    missing = [c for c in V9_FEATURE_COLUMNS if c not in train_base.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    train_base = train_base[["battle_id", "label", "is_real", *V9_FEATURE_COLUMNS]].copy()
    train_aug = add_noise_augmentation(train_base, repeats=2)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=42)
    groups = train_aug["battle_id"]
    train_idx, test_idx = next(splitter.split(train_aug, train_aug["label"], groups))
    train = train_aug.iloc[train_idx].copy()
    test = train_aug.iloc[test_idx].copy()

    w_train = compute_sample_weights(train, comeback_ids, swing_ids)
    rng = np.random.default_rng(42)
    probs = w_train / max(w_train.sum(), 1e-9)
    resample_idx = rng.choice(len(train), size=len(train), replace=True, p=probs)
    train_fit = train.iloc[resample_idx].reset_index(drop=True)

    # --- early expert (RF-only after model_select_v11) ---
    early_mask = train_fit["minute_bin"] <= 8
    train_early = train_fit[early_mask] if early_mask.sum() >= 50 else train_fit
    x_early, y_early = train_early[early_cols], train_early["label"]
    model_early = RandomForestClassifier(
        n_estimators=220, max_depth=7, min_samples_leaf=12, random_state=42, n_jobs=-1
    )
    if verbose:
        print("training early expert (RF-only) ...")
    model_early.fit(x_early, y_early)

    # --- midlate expert (RF-strong voting after model_select_v11) ---
    x_train, y_train = train_fit[V9_FEATURE_COLUMNS], train_fit["label"]
    x_test, y_test = test[V9_FEATURE_COLUMNS], test["label"]
    models = {
        "LR": Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, C=0.35, random_state=42))]
        ),
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=8, random_state=42, n_jobs=-1
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=480, max_depth=11, min_samples_leaf=8, random_state=42, n_jobs=-1
        ),
        "GBDT": GradientBoostingClassifier(
            n_estimators=260, max_depth=4, learning_rate=0.045, subsample=0.82, min_samples_leaf=14, random_state=42
        ),
    }
    scores = {}
    fitted = []
    for name, model in models.items():
        if verbose:
            print(f"training midlate {name} ...")
        model.fit(x_train, y_train)
        scores[name] = evaluate(y_test, model.predict_proba(x_test)[:, 1])
        if verbose:
            print(" ", scores[name])
        fitted.append((name, model))
    weights = [max(0.2, scores[n]["auc"] * 2 + scores[n]["accuracy"] - scores[n]["brier"]) for n, _ in fitted]
    model_midlate = VotingClassifier(estimators=fitted, voting="soft", weights=weights, n_jobs=-1)
    model_midlate.fit(x_train, y_train)
    scores["VotingMidlate"] = evaluate(y_test, model_midlate.predict_proba(x_test)[:, 1])
    if verbose:
        print("VotingMidlate (mixed secondary)", scores["VotingMidlate"])

    # Holdout blend + phase calibrators (V11: Platt/isotonic chosen by mid Brier)
    holdout_raw = collect_holdout_blend_probs(
        model_early, early_cols, model_midlate, V9_FEATURE_COLUMNS, fallback_wr, holdout_battles
    )
    if not holdout_raw.empty:
        early_rows = holdout_raw[holdout_raw["minute_bin"] <= 8]
        cal_early = (
            fit_calibrator(early_rows["prob_early"].values, early_rows["label"].values, method="isotonic")
            if len(early_rows)
            else None
        )
        if cal_early is None:
            cal_early = fit_calibrator(holdout_raw["prob_early"].values, holdout_raw["label"].values, method="platt")

        # Early clip selection without midlate cal first
        p_tmp = finalize_holdout_probs(holdout_raw, cal_early, None, (0.02, 0.98), use_gold_guard=True)
        p_clip_tmp = finalize_holdout_probs(holdout_raw, cal_early, None, EARLY_PROB_CLIP, use_gold_guard=True)
        early_mask = holdout_raw["minute_bin"].values <= 8
        y_all = holdout_raw["label"].values
        b_no = float(brier_score_loss(y_all[early_mask], p_tmp[early_mask])) if early_mask.any() else 1.0
        b_yes = float(brier_score_loss(y_all[early_mask], p_clip_tmp[early_mask])) if early_mask.any() else 1.0
        use_early_clip = b_yes <= b_no
        early_clip = EARLY_PROB_CLIP if use_early_clip else (0.02, 0.98)

        cal_midlate, mid_cal_name, mid_cal_diag = choose_midlate_calibrator(holdout_raw, cal_early, early_clip)
        final_probs = finalize_holdout_probs(holdout_raw, cal_early, cal_midlate, early_clip, use_gold_guard=True)
        clip_diag = {
            "early_brier_no_clip": b_no,
            "early_brier_clip": b_yes,
            "chosen_clip": use_early_clip,
            "midlate_calibrator": mid_cal_name,
            "midlate_cal_diag": mid_cal_diag,
        }
        holdout_metrics = evaluate_phase_metrics(
            holdout_raw["minute"].values, holdout_raw["label"].values, final_probs
        )
        holdout_comeback, holdout_swing = detect_swing_sets(holdout_battles)
        if holdout_comeback:
            mask_c = holdout_raw["battle_id"].isin(holdout_comeback).values
            comeback_metrics = evaluate_phase_metrics(
                holdout_raw["minute"].values[mask_c],
                holdout_raw["label"].values[mask_c],
                final_probs[mask_c],
            )
        else:
            comeback_metrics = {}
        per_battle_acc = []
        for bid in holdout_raw["battle_id"].unique():
            m = holdout_raw["battle_id"].values == bid
            y = holdout_raw["label"].values[m]
            p = final_probs[m]
            ym = holdout_raw["minute_bin"].values[m]
            acc = float(((p >= 0.5) == y).mean()) if len(y) else 0.0
            late_acc = float(((p[ym >= 8] >= 0.5) == y[ym >= 8]).mean()) if np.any(ym >= 8) else acc
            per_battle_acc.append(
                {
                    "battle_id": str(bid),
                    "acc": acc,
                    "late_acc": late_acc,
                    "n": int(m.sum()),
                    "is_comeback": str(bid) in holdout_comeback,
                    "is_swing": str(bid) in holdout_swing,
                }
            )
        per_battle_acc.sort(key=lambda x: x["acc"])
    else:
        cal_early = cal_midlate = None
        early_clip = EARLY_PROB_CLIP
        use_early_clip = True
        holdout_metrics = {}
        comeback_metrics = {}
        per_battle_acc = []
        holdout_comeback = set()
        holdout_swing = set()
        clip_diag = {}
        mid_cal_name = "none"

    if verbose and holdout_metrics:
        print("Holdout real (headline):")
        for phase, m in holdout_metrics.items():
            print(
                f"  {phase:5s} n={m['n']:4d} Brier={m['brier']:.4f} ECE={m['ece']:.4f} "
                f"Acc={m['accuracy']:.3f} AUC={m['auc']:.3f}"
            )
        print(f"early_clip={early_clip} use={use_early_clip} cal_early={cal_early is not None} cal_mid={cal_midlate is not None} diag={clip_diag}")
        if comeback_metrics:
            print("comeback subset:", comeback_metrics.get("all"))
        if per_battle_acc:
            print("worst battles:", per_battle_acc[:3])

    artifact = {
        "version": "V11",
        "model_name": "VotingV11 EarlyRF + MidRFStrong",
        "model": model_midlate,  # legacy fallback
        "model_early": model_early,
        "model_midlate": model_midlate,
        "feature_columns": V9_FEATURE_COLUMNS,
        "early_feature_columns": early_cols,
        "calibrator": cal_midlate,  # legacy
        "calibrator_early": cal_early,
        "calibrator_midlate": cal_midlate,
        "midlate_calibrator_name": mid_cal_name if holdout_metrics else "none",
        "blend_start": BLEND_START,
        "blend_end": BLEND_END,
        "early_prob_clip": list(early_clip),
        "use_time_shrinkage": False,
        "use_gold_guard": True,
        "team_winrate": fallback_wr,
        "metrics": scores,
        "holdout_real": holdout_metrics,
        "holdout_comeback": comeback_metrics,
        "holdout_battles": sorted(holdout_ids),
        "comeback_train_battles": sorted(comeback_ids),
        "swing_train_battles": sorted(swing_ids),
        "worst_holdout_battles": per_battle_acc[:5],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "real_rows": int(len(real_df)),
        "sim_rows": int(len(sim_df)),
        "shrinkage_diag": clip_diag if holdout_metrics else {},
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "V11 gold consistency guard on mid/late clash.",
            "Causal FE: roll windows, momentum, lane crush, obj-tower convert, hurt concentration.",
            "Model select: early=RF-only(220,d7,leaf12); midlate=RF-strong voting (RF 500/d12).",
            "Dropped legacy lord(big_dragon) from objective_score; prophet/shadow used instead.",
            "Midlate calibrator chosen by holdout mid Brier (Platt/isotonic/none).",
            "Comeback + swing + mid-minute sample upweight; no early label-flip.",
        ],
    }
    out = MODEL_DIR / "v9_official_platform.joblib"
    joblib.dump(artifact, out)

    holdout_all = holdout_metrics.get("all", {})
    comeback_all = comeback_metrics.get("all", {})
    write_train_status(
        {
            "state": "ready",
            "reason": "manual_train",
            "version": "V11",
            "model_name": artifact["model_name"],
            "early_model": "RF-only",
            "trained_at": artifact["trained_at"],
            "auc": scores.get("VotingMidlate", {}).get("auc"),
            "accuracy": scores.get("VotingMidlate", {}).get("accuracy"),
            "holdout_real_brier": holdout_all.get("brier"),
            "holdout_real_ece": holdout_all.get("ece"),
            "holdout_early_brier": holdout_metrics.get("early", {}).get("brier"),
            "holdout_mid_brier": holdout_metrics.get("mid", {}).get("brier"),
            "holdout_late_brier": holdout_metrics.get("late", {}).get("brier"),
            "comeback_acc": comeback_all.get("accuracy"),
            "comeback_brier": comeback_all.get("brier"),
            "worst_battle_acc": per_battle_acc[0]["acc"] if per_battle_acc else None,
            "holdout_battles": artifact["holdout_battles"],
            "early_prob_clip": list(early_clip),
            "use_time_shrinkage": False,
            "use_gold_guard": True,
            "midlate_calibrator": artifact.get("midlate_calibrator_name"),
            "real_rows": artifact["real_rows"],
            "sim_rows": artifact["sim_rows"],
            "calibrator": f"early_isotonic+midlate_{artifact.get('midlate_calibrator_name', 'none')}",
        }
    )
    if verbose:
        print(f"saved: {out}")
    return artifact


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="use raw_snapshots instead of curated datasets/clean",
    )
    args = parser.parse_args()
    train_v9_model(verbose=True, from_raw=args.from_raw)


if __name__ == "__main__":
    main()
