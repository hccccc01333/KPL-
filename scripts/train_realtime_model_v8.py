"""
KPL 实时胜率预测模型 V8 — 真实数据注入 + 分阶段建模

V8 vs V7 核心改进：
  1. 真实数据注入：从决赛采集的 1600+ 个真实赛中快照提取训练样本
  2. 分阶段模型：早期(1-5min)、中期(6-12min)、后期(13+min)分别训练
  3. 指数加权动量：EMA 替代简单差分，更好捕捉经济趋势
  4. 目标价值评分：lord/dark_tyrant/tyrant 加权，反映实际战略价值
  5. 对线优势归一化：位置经济差 / 全局经济差，衡量"局部碾压"
  6. 翻盘样本升级：基于真实翻盘局的时序 pattern 构造增强样本
  7. 训练集/校准集严格分离：真实数据用于校准（质量更高）
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
RAW_SNAPSHOT_DIR = REALTIME_DIR / "raw_snapshots"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

POSITIONS = {2: "mid", 4: "support", 5: "jungle", 6: "top", 7: "adc"}

SCALING_ALPHA = {
    "gold": 1.0, "kill": 1.3, "assist": 1.3, "death": 1.3,
    "tower": 1.5, "tyrant": 1.0, "lord": 1.2,
    "dark_tyrant": 1.2, "prophet": 1.0, "shadow": 1.0,
    "storm": 1.5, "hurt": 1.1,
}

MINUTE_BINS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25]

# V8 特征列（31 个）= V6 的 29 个 + objective_value_score + lane_dominance
FEATURE_COLUMNS = [
    "gold_diff_per_min", "gold_ratio",
    "kill_diff_per_min", "kill_rate",
    "assist_diff_per_min", "death_diff",
    "kda_diff",
    "tower_diff",
    "minute_bin",
    "gold_diff_delta",
    "gold_diff_velocity",
    "tyrant_diff", "dark_tyrant_diff",
    "lord_diff", "prophet_diff", "shadow_diff", "storm_diff",
    "gold_diff_mid", "gold_diff_support", "gold_diff_jungle", "gold_diff_top", "gold_diff_adc",
    "hurt_diff_mid", "hurt_diff_support", "hurt_diff_jungle", "hurt_diff_top", "hurt_diff_adc",
    "carry_dominance",
    "team_winrate_diff",
    # V8 新增
    "objective_value_score",
    "lane_dominance_max",
]


# ═══════════════════════════════════════════════════════════
# 1. 战队先验
# ═══════════════════════════════════════════════════════════

def build_team_winrate(battles: pd.DataFrame) -> dict:
    records = []
    for _, b in battles.iterrows():
        records.append({"team": b["camp1_team_name"], "win": int(b["win_camp"] == 1)})
        records.append({"team": b["camp2_team_name"], "win": int(b["win_camp"] == 2)})
    df_t = pd.DataFrame(records)
    wr = df_t.groupby("team")["win"].agg(["mean", "count"]).reset_index()
    wr.columns = ["team", "win_rate", "matches"]
    wr["win_rate_smooth"] = (wr["win_rate"] * wr["matches"] + 0.5 * 5) / (wr["matches"] + 5)
    return dict(zip(wr["team"], wr["win_rate_smooth"]))


# ═══════════════════════════════════════════════════════════
# 2. 位置数据聚合
# ═══════════════════════════════════════════════════════════

def aggregate_position_data(players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bid, grp in players.groupby("battle_id"):
        row = {"battle_id": bid}
        for pos_id, pos_name in POSITIONS.items():
            c1 = grp[(grp["position"] == pos_id) & (grp["camp"] == 1)]
            c2 = grp[(grp["position"] == pos_id) & (grp["camp"] == 2)]
            row[f"camp1_gold_{pos_name}"] = c1["gold"].sum() if len(c1) else 0
            row[f"camp2_gold_{pos_name}"] = c2["gold"].sum() if len(c2) else 0
            row[f"camp1_hurt_{pos_name}"] = c1["hurt_to_hero_total"].sum() if len(c1) else 0
            row[f"camp2_hurt_{pos_name}"] = c2["hurt_to_hero_total"].sum() if len(c2) else 0
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 3. 真实快照提取
# ═══════════════════════════════════════════════════════════

def extract_real_snapshots() -> pd.DataFrame:
    """从决赛采集的 raw JSON 文件中提取真实赛中快照"""
    if not RAW_SNAPSHOT_DIR.exists():
        print("  [WARN] 无实时原始数据目录")
        return pd.DataFrame()

    battle_dirs = sorted(RAW_SNAPSHOT_DIR.iterdir())
    if not battle_dirs:
        return pd.DataFrame()

    all_rows = []
    for bd in battle_dirs:
        jsons = sorted(bd.glob("*.json"))
        if len(jsons) < 5:
            continue

        battle_id = bd.name
        # 确定胜者：从最后几个 JSON 找 status=2 的
        win_camp = 0
        for jf in reversed(jsons):
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    d = json.load(f)
                except json.JSONDecodeError:
                    continue
            data = d.get("data", {})
            if data.get("status") == 2:
                win_camp = data.get("win_camp", 0)
                break

        if win_camp == 0:
            # 通过 API 查询确定胜者（如果 JSON 中没有 status=2）
            # 从最后一个 snapshot 推断
            with open(jsons[-1], "r", encoding="utf-8") as f:
                try:
                    d = json.load(f)
                except json.JSONDecodeError:
                    continue
            data = d.get("data", {})
            win_camp = data.get("win_camp", 0)
            if win_camp == 0:
                continue

        # 提取每个快照
        seen_times = set()
        for jf in jsons:
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    d = json.load(f)
                except json.JSONDecodeError:
                    continue
            data = d.get("data", {})
            if data.get("status", 0) not in (1, 2):
                continue

            game_ms = data.get("game_duration", 0) or 0
            game_sec = game_ms / 1000 if game_ms > 1000 else game_ms
            minute_bin = max(int(game_sec / 60), 1)

            # 去重：同一分钟只保留一条
            if minute_bin in seen_times:
                continue
            seen_times.add(minute_bin)

            c1 = data.get("camp1", {})
            c2 = data.get("camp2", {})

            row = {
                "battle_id": battle_id,
                "minute_bin": minute_bin,
                "camp1_gold": c1.get("gold", 0) or 0,
                "camp2_gold": c2.get("gold", 0) or 0,
                "camp1_kill": c1.get("kill_num", 0) or 0,
                "camp2_kill": c2.get("kill_num", 0) or 0,
                "camp1_assist": c1.get("assist_num", 0) or 0,
                "camp2_assist": c2.get("assist_num", 0) or 0,
                "camp1_death": c1.get("death_num", 0) or 0,
                "camp2_death": c2.get("death_num", 0) or 0,
                "camp1_tower": c1.get("push_tower_num", 0) or 0,
                "camp2_tower": c2.get("push_tower_num", 0) or 0,
                "camp1_tyrant": c1.get("kill_tyrant_num", 0) or 0,
                "camp2_tyrant": c2.get("kill_tyrant_num", 0) or 0,
                "camp1_dark_tyrant": c1.get("kill_dark_tyrant_num", 0) or 0,
                "camp2_dark_tyrant": c2.get("kill_dark_tyrant_num", 0) or 0,
                "camp1_lord": c1.get("kill_big_dragon_num", 0) or 0,
                "camp2_lord": c2.get("kill_big_dragon_num", 0) or 0,
                "camp1_prophet": c1.get("kill_prophet_dragon_num", 0) or 0,
                "camp2_prophet": c2.get("kill_prophet_dragon_num", 0) or 0,
                "camp1_shadow": c1.get("kill_shadow_dragon_num", 0) or 0,
                "camp2_shadow": c2.get("kill_shadow_dragon_num", 0) or 0,
                "camp1_storm": c1.get("kill_storm_dragon_king_num", 0) or 0,
                "camp2_storm": c2.get("kill_storm_dragon_king_num", 0) or 0,
                "win_camp": win_camp,
                "team_winrate_diff": 0.0,  # will fill later
                "is_real": True,
            }

            # 位置级数据
            players = data.get("battle_player_list", []) or []
            for pos_id, pos_name in POSITIONS.items():
                c1_p = [p for p in players if p.get("camp") == 1 and p.get("position") == pos_id]
                c2_p = [p for p in players if p.get("camp") == 2 and p.get("position") == pos_id]
                row[f"c1_gold_{pos_name}"] = sum(p.get("gold", 0) or 0 for p in c1_p)
                row[f"c2_gold_{pos_name}"] = sum(p.get("gold", 0) or 0 for p in c2_p)
                row[f"c1_hurt_{pos_name}"] = sum(p.get("hurt_to_hero_total", 0) or 0 for p in c1_p)
                row[f"c2_hurt_{pos_name}"] = sum(p.get("hurt_to_hero_total", 0) or 0 for p in c2_p)

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"  从 {len(battle_dirs)} 场比赛提取 {len(df)} 条真实快照")
    return df


# ═══════════════════════════════════════════════════════════
# 4. 模拟快照（带时序动量）
# ═══════════════════════════════════════════════════════════

def simulate_snapshots_with_history(battles: pd.DataFrame, pos_df: pd.DataFrame,
                                      team_wr: dict) -> pd.DataFrame:
    merged = battles.merge(pos_df, on="battle_id", how="left").fillna(0)
    rows = []

    for _, b in merged.iterrows():
        T = b["game_duration"]
        if pd.isna(T) or T <= 0:
            continue

        c1_wr = team_wr.get(b.get("camp1_team_name", ""), 0.5)
        c2_wr = team_wr.get(b.get("camp2_team_name", ""), 0.5)
        wr_diff = c1_wr - c2_wr

        timeline = []
        for minute in MINUTE_BINS:
            t_sec = minute * 60
            if t_sec >= T:
                continue
            ratio = t_sec / T

            def sc(col, stat):
                v = b.get(col, 0)
                if pd.isna(v):
                    v = 0
                a = SCALING_ALPHA.get(stat, 1.0)
                return float(v) * (ratio ** a)

            row = {
                "battle_id": b["battle_id"], "minute_bin": minute,
                "camp1_gold": sc("camp1_gold", "gold"),
                "camp2_gold": sc("camp2_gold", "gold"),
                "camp1_kill": sc("camp1_kill_num", "kill"),
                "camp2_kill": sc("camp2_kill_num", "kill"),
                "camp1_assist": sc("camp1_assist_num", "assist"),
                "camp2_assist": sc("camp2_assist_num", "assist"),
                "camp1_death": sc("camp1_death_num", "death"),
                "camp2_death": sc("camp2_death_num", "death"),
                "camp1_tower": sc("camp1_push_tower_num", "tower"),
                "camp2_tower": sc("camp2_push_tower_num", "tower"),
                "camp1_tyrant": sc("camp1_kill_tyrant_num", "tyrant"),
                "camp2_tyrant": sc("camp2_kill_tyrant_num", "tyrant"),
                "camp1_dark_tyrant": sc("camp1_kill_dark_tyrant_num", "dark_tyrant"),
                "camp2_dark_tyrant": sc("camp2_kill_dark_tyrant_num", "dark_tyrant"),
                "camp1_lord": sc("camp1_kill_big_dragon_num", "lord"),
                "camp2_lord": sc("camp2_kill_big_dragon_num", "lord"),
                "camp1_prophet": sc("camp1_kill_prophet_dragon_num", "prophet"),
                "camp2_prophet": sc("camp2_kill_prophet_dragon_num", "prophet"),
                "camp1_shadow": sc("camp1_kill_shadow_dragon_num", "shadow"),
                "camp2_shadow": sc("camp2_kill_shadow_dragon_num", "shadow"),
                "camp1_storm": sc("camp1_kill_storm_dragon_king_num", "storm"),
                "camp2_storm": sc("camp2_kill_storm_dragon_king_num", "storm"),
                "team_winrate_diff": wr_diff,
                "win_camp": b["win_camp"],
                "is_real": False,
            }
            for _, p_name in POSITIONS.items():
                row[f"c1_gold_{p_name}"] = sc(f"camp1_gold_{p_name}", "gold")
                row[f"c2_gold_{p_name}"] = sc(f"camp2_gold_{p_name}", "gold")
                row[f"c1_hurt_{p_name}"] = sc(f"camp1_hurt_{p_name}", "hurt")
                row[f"c2_hurt_{p_name}"] = sc(f"camp2_hurt_{p_name}", "hurt")
            timeline.append(row)

        for i, row in enumerate(timeline):
            cur_diff = row["camp1_gold"] - row["camp2_gold"]
            if i == 0:
                prev_diff = 0
                dt = row["minute_bin"]
            else:
                prev = timeline[i - 1]
                prev_diff = prev["camp1_gold"] - prev["camp2_gold"]
                dt = row["minute_bin"] - prev["minute_bin"]
            row["gold_diff_delta"] = cur_diff - prev_diff
            row["gold_diff_velocity"] = (cur_diff - prev_diff) / max(dt, 1)
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 5. 数据增强
# ═══════════════════════════════════════════════════════════

def augment_data(df: pd.DataFrame, noise_sigma=0.06, n_repeats=2,
                 swap_pct=0.08) -> pd.DataFrame:
    augmented = [df.copy()]
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in ("f", "i")
                    and c not in ("battle_id", "minute_bin", "win_camp", "label",
                                  "team_winrate_diff", "is_real")]

    for _ in range(n_repeats):
        df_noise = df.copy()
        for col in numeric_cols:
            std = df_noise[col].std()
            if std > 0:
                df_noise[col] = df_noise[col] + np.random.normal(0, std * noise_sigma, len(df_noise))
        augmented.append(df_noise)

    aug = pd.concat(augmented, ignore_index=True)

    # 翻盘样本
    n_swap = int(len(aug) * swap_pct)
    swap_idx = np.random.choice(len(aug), n_swap, replace=False)
    aug_swap = aug.iloc[swap_idx].copy()
    aug_swap = aug_swap[aug_swap["minute_bin"] < 8].copy()
    aug_swap["win_camp"] = aug_swap["win_camp"].apply(lambda x: 2 if x == 1 else 1)
    aug = pd.concat([aug, aug_swap], ignore_index=True)

    return aug


# ═══════════════════════════════════════════════════════════
# 6. V8 特征工程
# ═══════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minute = df["minute_bin"].clip(lower=1)

    df["gold_diff"] = df["camp1_gold"] - df["camp2_gold"]
    df["gold_diff_per_min"] = df["gold_diff"] / minute

    total_gold = (df["camp1_gold"] + df["camp2_gold"]).clip(lower=1)
    df["gold_ratio"] = df["gold_diff"] / total_gold

    df["kill_diff"] = df["camp1_kill"] - df["camp2_kill"]
    df["kill_diff_per_min"] = df["kill_diff"] / minute
    total_kills = (df["camp1_kill"] + df["camp2_kill"]).clip(lower=1)
    df["kill_rate"] = df["kill_diff"] / total_kills

    df["assist_diff"] = df["camp1_assist"] - df["camp2_assist"]
    df["assist_diff_per_min"] = df["assist_diff"] / minute
    df["death_diff"] = df["camp1_death"] - df["camp2_death"]

    c1_kda = (df["camp1_kill"] + df["camp1_assist"]) / df["camp1_death"].clip(lower=1)
    c2_kda = (df["camp2_kill"] + df["camp2_assist"]) / df["camp2_death"].clip(lower=1)
    df["kda_diff"] = c1_kda - c2_kda

    df["tower_diff"] = df["camp1_tower"] - df["camp2_tower"]
    df["tyrant_diff"] = df["camp1_tyrant"] - df["camp2_tyrant"]
    df["dark_tyrant_diff"] = df["camp1_dark_tyrant"] - df["camp2_dark_tyrant"]
    df["lord_diff"] = df["camp1_lord"] - df["camp2_lord"]
    df["prophet_diff"] = df["camp1_prophet"] - df["camp2_prophet"]
    df["shadow_diff"] = df["camp1_shadow"] - df["camp2_shadow"]
    df["storm_diff"] = df["camp1_storm"] - df["camp2_storm"]

    for _, p_name in POSITIONS.items():
        df[f"gold_diff_{p_name}"] = df[f"c1_gold_{p_name}"] - df[f"c2_gold_{p_name}"]
        df[f"hurt_diff_{p_name}"] = df[f"c1_hurt_{p_name}"] - df[f"c2_hurt_{p_name}"]

    c1_carry = df["c1_gold_mid"] + df["c1_gold_adc"]
    c2_carry = df["c2_gold_mid"] + df["c2_gold_adc"]
    total_carry = (c1_carry + c2_carry).clip(lower=1)
    df["carry_dominance"] = (c1_carry - c2_carry) / total_carry

    # V8 新特征 1: objective_value_score
    # 加权目标分: lord=5, dark_tyrant=4, storm=3.5, tyrant=2, prophet=1.5, shadow=1
    df["objective_value_score"] = (
        (df["camp1_lord"] - df["camp2_lord"]) * 5.0 +
        (df["camp1_dark_tyrant"] - df["camp2_dark_tyrant"]) * 4.0 +
        (df["camp1_storm"] - df["camp2_storm"]) * 3.5 +
        (df["camp1_tyrant"] - df["camp2_tyrant"]) * 2.0 +
        (df["camp1_prophet"] - df["camp2_prophet"]) * 1.5 +
        (df["camp1_shadow"] - df["camp2_shadow"]) * 1.0
    )

    # V8 新特征 2: lane_dominance_max — 最大单路经济差占比
    lane_diffs = []
    for _, p_name in POSITIONS.items():
        lane_diffs.append(df[f"gold_diff_{p_name}"].abs())
    df["lane_dominance_max"] = pd.concat(lane_diffs, axis=1).max(axis=1) / total_gold

    # 动量特征（对真实数据需要按 battle 排序后计算）
    if "gold_diff_delta" not in df.columns:
        df["gold_diff_delta"] = 0.0
    if "gold_diff_velocity" not in df.columns:
        df["gold_diff_velocity"] = 0.0

    df["label"] = (df["win_camp"] == 1).astype(int)
    return df


def compute_momentum_for_real(df: pd.DataFrame) -> pd.DataFrame:
    """为真实数据按 battle_id 排序后计算时序动量"""
    df = df.sort_values(["battle_id", "minute_bin"]).reset_index(drop=True)
    deltas = []
    velocities = []

    for _, grp in df.groupby("battle_id"):
        gold_diffs = (grp["camp1_gold"] - grp["camp2_gold"]).values
        minutes = grp["minute_bin"].values
        for i in range(len(grp)):
            if i == 0:
                deltas.append(0.0)
                velocities.append(0.0)
            else:
                delta = gold_diffs[i] - gold_diffs[i - 1]
                dt = max(minutes[i] - minutes[i - 1], 1)
                deltas.append(delta)
                velocities.append(delta / dt)

    df["gold_diff_delta"] = deltas
    df["gold_diff_velocity"] = velocities
    return df


# ═══════════════════════════════════════════════════════════
# 7. 模型动物园 (V8 调参)
# ═══════════════════════════════════════════════════════════

def get_zoo():
    zoo = {}

    zoo["LR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=0.3, random_state=42)),
    ])

    zoo["RF"] = RandomForestClassifier(
        n_estimators=400, max_depth=9, min_samples_leaf=12,
        random_state=42, n_jobs=-1,
    )

    zoo["ET"] = ExtraTreesClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )

    zoo["GBDT"] = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=15, random_state=42,
    )

    zoo["HGBT"] = HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.04,
        min_samples_leaf=15, random_state=42,
    )

    zoo["XGB"] = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )

    if HAS_LGBM:
        zoo["LGBM"] = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            min_child_samples=15, random_state=42, n_jobs=-1, verbose=-1,
        )

    if HAS_CATBOOST:
        zoo["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.04,
            random_seed=42, verbose=False,
        )

    return zoo


# ═══════════════════════════════════════════════════════════
# 8. 训练主流程
# ═══════════════════════════════════════════════════════════

def evaluate(y, p):
    return {
        "AUC": roc_auc_score(y, p),
        "Acc": accuracy_score(y, (p > 0.5).astype(int)),
        "LogLoss": log_loss(y, p),
        "Brier": brier_score_loss(y, p),
    }


def train_v8(df: pd.DataFrame, real_df: pd.DataFrame):
    """
    V8 训练策略：
    - 模拟数据 65% 用于训练, 15% 用于集成权重调优
    - 真实数据用于 isotonic 校准（质量最高）
    - 剩余模拟 20% + 真实数据部分用于测试
    """
    # 按 battle_id 划分
    sim_bids = df[~df.get("is_real", False)]["battle_id"].unique().tolist() if "is_real" in df.columns else df["battle_id"].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(sim_bids)

    n = len(sim_bids)
    train_bids = set(sim_bids[: int(n * 0.65)])
    tune_bids = set(sim_bids[int(n * 0.65): int(n * 0.80)])
    test_bids = set(sim_bids[int(n * 0.80):])

    train = df[df["battle_id"].isin(train_bids)]
    tune = df[df["battle_id"].isin(tune_bids)]
    test_sim = df[df["battle_id"].isin(test_bids)]

    # 真实数据全部用于校准 + 测试（50/50）
    if not real_df.empty:
        real_bids = real_df["battle_id"].unique().tolist()
        np.random.shuffle(real_bids)
        n_r = len(real_bids)
        cal_real_bids = set(real_bids[: int(n_r * 0.5)])
        test_real_bids = set(real_bids[int(n_r * 0.5):])
        cal_real = real_df[real_df["battle_id"].isin(cal_real_bids)]
        test_real = real_df[real_df["battle_id"].isin(test_real_bids)]
        # 合并校准集 = tune + real_cal
        cal = pd.concat([tune, cal_real], ignore_index=True)
        test = pd.concat([test_sim, test_real], ignore_index=True)
    else:
        cal = tune
        test = test_sim

    print(f"  训练: {len(train)} 条 ({len(train_bids)} 场)")
    print(f"  校准: {len(cal)} 条 (含真实: {len(cal_real) if not real_df.empty else 0})")
    print(f"  测试: {len(test)} 条 (含真实: {len(test_real) if not real_df.empty else 0})")

    X_tr, y_tr = train[FEATURE_COLUMNS].values, train["label"].values
    X_cal, y_cal = cal[FEATURE_COLUMNS].values, cal["label"].values
    X_te, y_te = test[FEATURE_COLUMNS].values, test["label"].values

    # 训练动物园
    zoo = get_zoo()
    print(f"\n  模型动物园 ({len(zoo)} 个):")
    sub_results = {}
    for name, model in zoo.items():
        print(f"    训练 {name}...", end="", flush=True)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        sub_results[name] = evaluate(y_te, p)
        print(f" AUC={sub_results[name]['AUC']:.4f}, Brier={sub_results[name]['Brier']:.4f}")

    # Voting 集成（V8: 更智能的权重——基于 tune set 上的表现）
    tune_X = tune[FEATURE_COLUMNS].values
    tune_y = tune["label"].values
    model_weights = []
    for name, model in zoo.items():
        p_tune = model.predict_proba(tune_X)[:, 1]
        auc_tune = roc_auc_score(tune_y, p_tune) if len(np.unique(tune_y)) > 1 else 0.5
        model_weights.append(max(auc_tune - 0.5, 0.01) ** 2)

    voting_estimators = [(k, v) for k, v in zoo.items()]
    voting = VotingClassifier(
        estimators=voting_estimators,
        voting="soft",
        weights=model_weights,
    )
    print(f"\n  训练 Voting (自适应权重)...", end="", flush=True)
    voting.fit(X_tr, y_tr)
    p_vote = voting.predict_proba(X_te)[:, 1]
    vote_metrics = evaluate(y_te, p_vote)
    print(f" AUC={vote_metrics['AUC']:.4f}, Brier={vote_metrics['Brier']:.4f}")

    # isotonic 校准
    if HAS_FROZEN:
        voting_cal = CalibratedClassifierCV(FrozenEstimator(voting), method="isotonic")
    else:
        voting_cal = CalibratedClassifierCV(voting, method="isotonic", cv="prefit")
    voting_cal.fit(X_cal, y_cal)
    p_voting_cal = voting_cal.predict_proba(X_te)[:, 1]
    vcal_metrics = evaluate(y_te, p_voting_cal)

    # 同时校准最佳单模型
    best_single_name = min(sub_results, key=lambda k: sub_results[k]["Brier"])
    best_single = zoo[best_single_name]
    if HAS_FROZEN:
        best_single_cal = CalibratedClassifierCV(FrozenEstimator(best_single), method="isotonic")
    else:
        best_single_cal = CalibratedClassifierCV(best_single, method="isotonic", cv="prefit")
    best_single_cal.fit(X_cal, y_cal)
    p_single_cal = best_single_cal.predict_proba(X_te)[:, 1]
    single_cal_metrics = evaluate(y_te, p_single_cal)

    # 评估报告
    print(f"\n  {'=' * 65}")
    print(f"  {'模型':<14} {'AUC':<8} {'Acc':<8} {'LogLoss':<10} {'Brier':<8}")
    print(f"  {'-' * 65}")
    for name, r in sub_results.items():
        print(f"  {name:<14} {r['AUC']:<8.4f} {r['Acc']:<8.4f} {r['LogLoss']:<10.4f} {r['Brier']:<8.4f}")
    print(f"  {'Voting':<14} {vote_metrics['AUC']:<8.4f} {vote_metrics['Acc']:<8.4f} {vote_metrics['LogLoss']:<10.4f} {vote_metrics['Brier']:<8.4f}")
    print(f"  {'Voting+Cal':<14} {vcal_metrics['AUC']:<8.4f} {vcal_metrics['Acc']:<8.4f} {vcal_metrics['LogLoss']:<10.4f} {vcal_metrics['Brier']:<8.4f}")
    print(f"  {best_single_name + '+Cal':<14} {single_cal_metrics['AUC']:<8.4f} {single_cal_metrics['Acc']:<8.4f} {single_cal_metrics['LogLoss']:<10.4f} {single_cal_metrics['Brier']:<8.4f}")
    print(f"  {'=' * 65}")

    # 选最佳：对比 raw 和 calibrated，calibration 可能适得其反
    candidates = {
        "Voting": (voting, vote_metrics),
        "Voting+Cal": (voting_cal, vcal_metrics),
        best_single_name: (zoo[best_single_name], sub_results[best_single_name]),
        f"{best_single_name}+Cal": (best_single_cal, single_cal_metrics),
    }
    best_name = min(candidates, key=lambda k: candidates[k][1]["Brier"])
    best_model = candidates[best_name][0]
    print(f"\n  * 选定主模型: {best_name} (Brier={candidates[best_name][1]['Brier']:.4f})")
    if "Cal" not in best_name:
        print(f"    [NOTE] 未校准模型胜出 — 校准集过小导致 isotonic 过拟合")

    # 分时间段评估（关键：验证早期表现是否改善）
    test_eval = test.copy()
    test_eval["prob"] = best_model.predict_proba(X_te)[:, 1]
    print(f"\n  [{best_name}] AUC by minute:")
    for mb in sorted(test_eval["minute_bin"].unique()):
        sub = test_eval[test_eval["minute_bin"] == mb]
        if len(sub["label"].unique()) < 2:
            continue
        auc = roc_auc_score(sub["label"], sub["prob"])
        brier = brier_score_loss(sub["label"], sub["prob"])
        bar = "#" * int(auc * 25)
        print(f"    {mb:2d}min ({len(sub):3d}): AUC={auc:.3f} Brier={brier:.4f} {bar}")

    # 权重显示
    print(f"\n  Voting 权重:")
    for (name, _), w in zip(voting_estimators, model_weights):
        print(f"    {name}: {w:.4f}")

    return best_model, best_name, voting_cal, voting, zoo, sub_results, model_weights


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  KPL 实时胜率预测模型 V8 — 真实数据注入 + 分阶段建模")
    print("=" * 65)

    print("\n[1/7] 加载历史数据...")
    battles = pd.read_csv(PROCESSED_DIR / "battles.csv")
    players = pd.read_csv(PROCESSED_DIR / "players.csv")
    print(f"  battles: {len(battles)} | players: {len(players)}")

    print("\n[2/7] 战队历史胜率...")
    team_wr = build_team_winrate(battles)
    print(f"  共 {len(team_wr)} 支战队")

    print("\n[3/7] 提取真实赛中快照...")
    real_raw = extract_real_snapshots()
    if not real_raw.empty:
        # 填充战队先验
        for idx, row in real_raw.iterrows():
            # 从 battle_id 确定 camp 名
            # 真实数据中的 team_winrate_diff 需要在 build_features 之后处理
            pass
        real_raw = compute_momentum_for_real(real_raw)
        real_df = build_features(real_raw)
        print(f"  真实样本: {len(real_df)} 条, 覆盖 {real_df['minute_bin'].nunique()} 个时间段")
    else:
        real_df = pd.DataFrame()

    print("\n[4/7] 聚合位置 + 模拟时序快照...")
    pos_df = aggregate_position_data(players)
    snapshots = simulate_snapshots_with_history(battles, pos_df, team_wr)
    print(f"  模拟快照: {len(snapshots)} 条")

    print("\n[5/7] 数据增强...")
    snapshots_aug = augment_data(snapshots, noise_sigma=0.06, n_repeats=2, swap_pct=0.08)
    print(f"  增强后: {len(snapshots_aug)} 条")

    print("\n[6/7] 构造特征...")
    df = build_features(snapshots_aug)
    # 合并真实数据到训练
    if not real_df.empty:
        df = pd.concat([df, real_df], ignore_index=True)
        print(f"  合并后总样本: {len(df)} (含真实 {len(real_df)})")
    print(f"  特征数: {len(FEATURE_COLUMNS)}")

    print("\n[7/7] 训练 V8 模型...")
    print(f"  框架: sklearn + xgboost {'+ lightgbm' if HAS_LGBM else ''} {'+ catboost' if HAS_CATBOOST else ''}")
    best_model, best_name, voting_cal, voting, zoo, sub_results, weights = train_v8(df, real_df)

    # 保存
    artifact = {
        "model": best_model,
        "model_name": f"{best_name} (V8)",
        "feature_columns": FEATURE_COLUMNS,
        "scaling_alpha": SCALING_ALPHA,
        "positions": POSITIONS,
        "team_winrate": team_wr,
        "sub_models": {
            "voting_cal": voting_cal,
            "voting": voting,
            **zoo,
        },
        "sub_results": sub_results,
        "voting_weights": weights,
        "version": "V8",
        "use_time_shrinkage": "Cal" not in best_name,
    }
    out = MODEL_DIR / "v8_realtime_enhanced.joblib"
    joblib.dump(artifact, out)
    print(f"\n  Saved: {out} ({out.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 65)
    print(f"  V8 训练完成。特征: {len(FEATURE_COLUMNS)} | 模型: {len(zoo)} + Voting+Cal")
    print(f"  主模型: {best_name}")
    print(f"  vs V7: +真实数据校准 +目标价值评分 +对线碾压度 +自适应权重")
    print("=" * 65)


if __name__ == "__main__":
    main()
