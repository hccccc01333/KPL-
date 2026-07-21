"""
KPL 实时胜率预测模型 V6（决赛级）

V6 vs V5 的核心改进：
  1. 数据增强：高斯噪声注入 + 特征 dropout，模拟真实数据的"非完美"
  2. 概率校准：CalibratedClassifierCV (isotonic)，替换 V5 的硬编码 minute/8 衰减
  3. 共线性修复：去掉 exp_diff_per_min（与 gold_diff_per_min 重复）
  4. 时序窗口：训练时构造"上一时刻"的动量特征（gold_diff_delta）
  5. 战队先验：用历史胜率 logit 作为基线特征
  6. 翻盘样本生成：人工构造 swap 翻盘样本，让模型见过"反向"模式
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SCALING_ALPHA = {
    "gold": 1.0, "kill": 1.3, "assist": 1.3, "death": 1.3,
    "tower": 1.5, "tyrant": 1.0, "lord": 1.2,
    "dark_tyrant": 1.2, "prophet": 1.0, "shadow": 1.0,
    "storm": 1.5, "hurt": 1.1,
}

MINUTE_BINS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25]
POSITIONS = {2: "mid", 4: "support", 5: "jungle", 6: "top", 7: "adc"}

# 去掉 exp_diff_per_min（与 gold_diff_per_min 共线）
# 加入 gold_diff_delta（上一分钟到当前的动量）
FEATURE_COLUMNS = [
    "gold_diff_per_min", "gold_ratio",
    "kill_diff_per_min", "kill_rate",
    "assist_diff_per_min", "death_diff",
    "kda_diff",
    "tower_diff",
    "minute_bin",
    # 时序动量
    "gold_diff_delta",       # 上一分钟到当前的经济差变化
    "gold_diff_velocity",    # 经济差速度（标准化）
    # 野怪
    "tyrant_diff", "dark_tyrant_diff",
    "lord_diff", "prophet_diff", "shadow_diff", "storm_diff",
    # 位置
    "gold_diff_mid", "gold_diff_support", "gold_diff_jungle", "gold_diff_top", "gold_diff_adc",
    "hurt_diff_mid", "hurt_diff_support", "hurt_diff_jungle", "hurt_diff_top", "hurt_diff_adc",
    "carry_dominance",
    # 战队先验
    "team_winrate_diff",
]


# ═══════════════════════════════════════════════════════════
# 战队先验
# ═══════════════════════════════════════════════════════════

def build_team_winrate(battles: pd.DataFrame) -> dict:
    """计算每支战队的历史胜率"""
    records = []
    for _, b in battles.iterrows():
        records.append({"team": b["camp1_team_name"], "win": int(b["win_camp"] == 1)})
        records.append({"team": b["camp2_team_name"], "win": int(b["win_camp"] == 2)})
    df_t = pd.DataFrame(records)
    wr = df_t.groupby("team")["win"].agg(["mean", "count"]).reset_index()
    wr.columns = ["team", "win_rate", "matches"]
    # 贝叶斯平滑：少于 5 场的用 0.5
    wr["win_rate_smooth"] = (wr["win_rate"] * wr["matches"] + 0.5 * 5) / (wr["matches"] + 5)
    return dict(zip(wr["team"], wr["win_rate_smooth"]))


# ═══════════════════════════════════════════════════════════
# 位置聚合
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
# 模拟（带"上一分钟"用于动量特征）
# ═══════════════════════════════════════════════════════════

def simulate_snapshots_with_history(battles: pd.DataFrame, pos_df: pd.DataFrame,
                                      team_wr: dict) -> pd.DataFrame:
    merged = battles.merge(pos_df, on="battle_id", how="left").fillna(0)
    rows = []

    for _, b in merged.iterrows():
        T = b["game_duration"]
        if pd.isna(T) or T <= 0:
            continue

        # 战队胜率先验
        c1_wr = team_wr.get(b.get("camp1_team_name", ""), 0.5)
        c2_wr = team_wr.get(b.get("camp2_team_name", ""), 0.5)
        wr_diff = c1_wr - c2_wr

        # 先生成所有时间点的"绝对值"
        timeline = []
        for minute in MINUTE_BINS:
            t_sec = minute * 60
            if t_sec >= T:
                continue
            ratio = t_sec / T

            def sc(col, stat):
                v = b.get(col, 0)
                if pd.isna(v): v = 0
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
            }
            for _, p_name in POSITIONS.items():
                row[f"c1_gold_{p_name}"] = sc(f"camp1_gold_{p_name}", "gold")
                row[f"c2_gold_{p_name}"] = sc(f"camp2_gold_{p_name}", "gold")
                row[f"c1_hurt_{p_name}"] = sc(f"camp1_hurt_{p_name}", "hurt")
                row[f"c2_hurt_{p_name}"] = sc(f"camp2_hurt_{p_name}", "hurt")
            timeline.append(row)

        # 计算动量特征：当前 minute 的 gold_diff - 上一时刻的 gold_diff
        for i, row in enumerate(timeline):
            cur_diff = row["camp1_gold"] - row["camp2_gold"]
            if i == 0:
                prev_diff = 0
                dt = row["minute_bin"]
            else:
                prev = timeline[i - 1]
                prev_diff = prev["camp1_gold"] - prev["camp2_gold"]
                dt = row["minute_bin"] - prev["minute_bin"]
            row["_prev_gold_diff"] = prev_diff
            row["_dt"] = max(dt, 1)
            row["gold_diff_delta"] = cur_diff - prev_diff
            row["gold_diff_velocity"] = (cur_diff - prev_diff) / max(dt, 1)
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 数据增强：噪声 + 翻盘样本
# ═══════════════════════════════════════════════════════════

def augment_data(df: pd.DataFrame, noise_sigma=0.08, n_repeats=2,
                 swap_pct=0.10) -> pd.DataFrame:
    """
    数据增强：
      1. 复制 n_repeats 倍 + 加高斯噪声（模拟真实数据的扰动）
      2. 翻盘样本：随机选 swap_pct 的样本，把 minute < 8 的快照保留原局势但 label 翻转
         （模拟"前期领先后期被翻盘"）
    """
    augmented = []

    # 1. 噪声增强
    augmented.append(df.copy())
    for _ in range(n_repeats):
        df_noise = df.copy()
        # 给所有 *_diff 字段加噪声（按列内 std）
        for col in df_noise.columns:
            if col in ("battle_id", "minute_bin", "win_camp", "label",
                      "team_winrate_diff", "_prev_gold_diff", "_dt"):
                continue
            if df_noise[col].dtype.kind not in ("f", "i"):
                continue
            std = df_noise[col].std()
            if std > 0:
                df_noise[col] = df_noise[col] + np.random.normal(0, std * noise_sigma, len(df_noise))
        augmented.append(df_noise)

    aug = pd.concat(augmented, ignore_index=True)

    # 2. 翻盘样本：模拟"早期领先后期翻盘"
    n_swap = int(len(aug) * swap_pct)
    swap_idx = np.random.choice(len(aug), n_swap, replace=False)
    aug_swap = aug.iloc[swap_idx].copy()
    # 只取早期（minute < 8）的样本
    aug_swap = aug_swap[aug_swap["minute_bin"] < 8].copy()
    # 翻转 label：label 原本是 (win_camp==1)，现在反转为对方胜
    aug_swap["win_camp"] = aug_swap["win_camp"].apply(lambda x: 2 if x == 1 else 1)
    # 把这些样本的胜率衰减信号（在 LR 中表现为接近 0.5）
    aug = pd.concat([aug, aug_swap], ignore_index=True)

    return aug


# ═══════════════════════════════════════════════════════════
# 特征工程
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

    df["label"] = (df["win_camp"] == 1).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# 训练 + 校准
# ═══════════════════════════════════════════════════════════

def train_with_calibration(df: pd.DataFrame):
    """训练 + isotonic 概率校准"""
    bids = df["battle_id"].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(bids)

    split = int(len(bids) * 0.70)
    cal_split = int(len(bids) * 0.85)

    train_bids = set(bids[:split])
    cal_bids = set(bids[split:cal_split])
    test_bids = set(bids[cal_split:])

    train = df[df["battle_id"].isin(train_bids)]
    cal = df[df["battle_id"].isin(cal_bids)]
    test = df[df["battle_id"].isin(test_bids)]

    print(f"  训练: {len(train)} | 校准: {len(cal)} | 测试: {len(test)}")
    print(f"  对应比赛: {len(train_bids)} / {len(cal_bids)} / {len(test_bids)}")

    X_tr, y_tr = train[FEATURE_COLUMNS].values, train["label"].values
    X_cal, y_cal = cal[FEATURE_COLUMNS].values, cal["label"].values
    X_te, y_te = test[FEATURE_COLUMNS].values, test["label"].values

    # ── LR ──
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=0.5, random_state=42)),
    ])
    lr_pipe.fit(X_tr, y_tr)

    # ── GBDT ──
    gbdt = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.06,
        subsample=0.85, min_samples_leaf=20, random_state=42,
    )
    gbdt.fit(X_tr, y_tr)

    # ── RF ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    # ── Voting ──
    voting = VotingClassifier(
        estimators=[("lr", lr_pipe), ("gbdt", gbdt), ("rf", rf)],
        voting="soft", weights=[1, 2, 1],
    )
    voting.fit(X_tr, y_tr)

    # ── isotonic 校准（用校准集）──
    if HAS_FROZEN:
        # sklearn >= 1.6: 用 FrozenEstimator 包装已训练模型
        voting_cal = CalibratedClassifierCV(FrozenEstimator(voting), method="isotonic")
    else:
        voting_cal = CalibratedClassifierCV(voting, method="isotonic", cv="prefit")
    voting_cal.fit(X_cal, y_cal)

    # ── 评估 ──
    print(f"\n  {'模型':<14} {'AUC':<8} {'Acc':<8} {'LogLoss':<10} {'Brier':<8}")
    print(f"  {'─' * 55}")
    for name, m in [("LR", lr_pipe), ("GBDT", gbdt), ("RF", rf),
                    ("Voting", voting), ("Voting+Calib", voting_cal)]:
        p = m.predict_proba(X_te)[:, 1]
        print(f"  {name:<14} {roc_auc_score(y_te, p):<8.4f} "
              f"{accuracy_score(y_te, (p>0.5).astype(int)):<8.4f} "
              f"{log_loss(y_te, p):<10.4f} {brier_score_loss(y_te, p):<8.4f}")

    # 分时间段评估
    test_eval = test.copy()
    test_eval["prob"] = voting_cal.predict_proba(X_te)[:, 1]
    print(f"\n  [Voting+Calib] AUC by minute:")
    for mb in sorted(test_eval["minute_bin"].unique()):
        sub = test_eval[test_eval["minute_bin"] == mb]
        if len(sub["label"].unique()) < 2:
            continue
        auc = roc_auc_score(sub["label"], sub["prob"])
        bar = "█" * int(auc * 25)
        print(f"    {mb:2d}min ({len(sub):3d} 条): AUC={auc:.3f} {bar}")

    return voting_cal, {"voting": voting, "lr": lr_pipe, "gbdt": gbdt, "rf": rf}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  KPL 实时胜率预测模型 V6 — 决赛级")
    print("═" * 60)

    print("\n[1/6] 加载历史数据...")
    battles = pd.read_csv(PROCESSED_DIR / "battles.csv")
    players = pd.read_csv(PROCESSED_DIR / "players.csv")
    print(f"  battles: {len(battles)} | players: {len(players)}")

    print("\n[2/6] 计算战队历史胜率（贝叶斯平滑）...")
    team_wr = build_team_winrate(battles)
    print(f"  共 {len(team_wr)} 支战队")
    # 显示几个战队
    sample = sorted(team_wr.items(), key=lambda x: -x[1])[:5]
    print(f"  Top 5 胜率: {sample}")

    print("\n[3/6] 聚合位置 + 模拟时序快照（含动量）...")
    pos_df = aggregate_position_data(players)
    snapshots = simulate_snapshots_with_history(battles, pos_df, team_wr)
    print(f"  原始快照: {len(snapshots)}")

    print("\n[4/6] 数据增强（噪声 + 翻盘样本）...")
    snapshots_aug = augment_data(snapshots, noise_sigma=0.08, n_repeats=2, swap_pct=0.10)
    print(f"  增强后: {len(snapshots_aug)} (×{len(snapshots_aug)/len(snapshots):.2f})")

    print("\n[5/6] 构造模型特征...")
    df = build_features(snapshots_aug)
    print(f"  特征数: {len(FEATURE_COLUMNS)}")

    print("\n[6/6] 训练 + isotonic 校准...")
    model, sub_models = train_with_calibration(df)

    # ── 保存 ──
    artifact = {
        "model": model,
        "model_name": "Voting+Calibrated (V6)",
        "feature_columns": FEATURE_COLUMNS,
        "scaling_alpha": SCALING_ALPHA,
        "positions": POSITIONS,
        "team_winrate": team_wr,
        "sub_models": sub_models,
    }
    out = MODEL_DIR / "v6_realtime_calibrated.joblib"
    joblib.dump(artifact, out)
    print(f"\n  ✅ 保存: {out} ({out.stat().st_size/1024:.1f} KB)")

    print("\n" + "═" * 60)
    print("  V6 训练完成。改进点:")
    print("   - 数据增强（高斯噪声 + 翻盘样本）")
    print("   - isotonic 概率校准（替换硬编码衰减）")
    print("   - 移除 exp_diff_per_min 共线特征")
    print("   - 加入 gold_diff_delta + velocity 时序动量")
    print("   - 战队历史胜率作为先验特征")
    print("═" * 60)


if __name__ == "__main__":
    main()
