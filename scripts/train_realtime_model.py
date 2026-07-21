"""
KPL 实时胜率预测模型 V5（完整版）

特征体系：
  ─ 队伍级（17）：经济/人头/助攻/死亡/塔/5种野怪/KDA/比赛时长
  ─ 位置级（10）：5个位置的经济差 + 5个位置的英雄伤害差
  ─ 共 27 个特征

模型策略：
  ─ LR        强基线，可解释性最好
  ─ GBDT      捕捉非线性交互
  ─ RF        多树集成，稳定性好
  ─ Voting    软投票集成（最终用这个）

数据来源：
  battles.csv  → 队伍级终局数据
  players.csv  → 玩家级终局数据（聚合到位置级）
  幂律插值     → 模拟各分钟的中间快照
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
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 幂律插值的指数（不同指标增长节奏不同）
SCALING_ALPHA = {
    "gold": 1.0,        # 经济近似线性
    "kill": 1.3,        # 击杀前期少后期多
    "assist": 1.3,      # 助攻同击杀
    "death": 1.3,       # 死亡同击杀
    "tower": 1.5,       # 推塔显著后期化
    "tyrant": 1.0,      # 暴君刷新固定
    "lord": 1.2,        # 主宰偏后期
    "dark_tyrant": 1.2, # 黑暗暴君偏后期
    "prophet": 1.0,     # 先知主宰固定刷新
    "shadow": 1.0,      # 暗影主宰固定刷新
    "storm": 1.5,       # 风暴龙王 15 分钟才刷
    "hurt": 1.1,        # 伤害近线性偏后期
}

# 时间切片：3 ~ 25 分钟
MINUTE_BINS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25]

# 5 个位置（KPL 标准）
POSITIONS = {2: "mid", 4: "support", 5: "jungle", 6: "top", 7: "adc"}

# 完整特征列表
FEATURE_COLUMNS = [
    # ── 队伍级核心（10）──
    "gold_diff_per_min", "gold_ratio",
    "kill_diff_per_min", "kill_rate",
    "assist_diff_per_min", "death_diff",
    "kda_diff",
    "tower_diff",
    "minute_bin",
    "exp_diff_per_min",  # 经济差速度（gold_diff / minute）
    # ── 野怪资源（6）──
    "tyrant_diff", "dark_tyrant_diff",
    "lord_diff", "prophet_diff", "shadow_diff", "storm_diff",
    # ── 位置经济差（5）──
    "gold_diff_mid", "gold_diff_support", "gold_diff_jungle", "gold_diff_top", "gold_diff_adc",
    # ── 位置伤害差（5）──
    "hurt_diff_mid", "hurt_diff_support", "hurt_diff_jungle", "hurt_diff_top", "hurt_diff_adc",
    # ── 衍生（1）──
    "carry_dominance",  # 后排（中路+发育）经济差占比
]


# ═══════════════════════════════════════════════════════════
# 数据加载与聚合
# ═══════════════════════════════════════════════════════════

def aggregate_position_data(players: pd.DataFrame) -> pd.DataFrame:
    """把玩家级数据聚合成 battle 级的位置差值"""
    rows = []
    for bid, grp in players.groupby("battle_id"):
        row = {"battle_id": bid}
        for pos_id, pos_name in POSITIONS.items():
            c1 = grp[(grp["position"] == pos_id) & (grp["camp"] == 1)]
            c2 = grp[(grp["position"] == pos_id) & (grp["camp"] == 2)]
            if len(c1) > 0 and len(c2) > 0:
                row[f"camp1_gold_{pos_name}"] = c1["gold"].sum()
                row[f"camp2_gold_{pos_name}"] = c2["gold"].sum()
                row[f"camp1_hurt_{pos_name}"] = c1["hurt_to_hero_total"].sum()
                row[f"camp2_hurt_{pos_name}"] = c2["hurt_to_hero_total"].sum()
            else:
                row[f"camp1_gold_{pos_name}"] = 0
                row[f"camp2_gold_{pos_name}"] = 0
                row[f"camp1_hurt_{pos_name}"] = 0
                row[f"camp2_hurt_{pos_name}"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 模拟快照
# ═══════════════════════════════════════════════════════════

def simulate_snapshots(battles: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    """对每场比赛 × 每个时间切片生成一条模拟快照"""
    merged = battles.merge(pos_df, on="battle_id", how="left").fillna(0)
    rows = []

    for _, b in merged.iterrows():
        T = b["game_duration"]
        if pd.isna(T) or T <= 0:
            continue

        for minute in MINUTE_BINS:
            t_sec = minute * 60
            if t_sec >= T:
                continue
            ratio = t_sec / T

            def sc(col, stat_type):
                val = b.get(col, 0)
                if pd.isna(val): val = 0
                a = SCALING_ALPHA.get(stat_type, 1.0)
                return max(0, float(val) * (ratio ** a))

            # 队伍级
            c1_gold = round(sc("camp1_gold", "gold"))
            c2_gold = round(sc("camp2_gold", "gold"))
            c1_kill = round(sc("camp1_kill_num", "kill"))
            c2_kill = round(sc("camp2_kill_num", "kill"))
            c1_assist = round(sc("camp1_assist_num", "assist"))
            c2_assist = round(sc("camp2_assist_num", "assist"))
            c1_death = round(sc("camp1_death_num", "death"))
            c2_death = round(sc("camp2_death_num", "death"))
            c1_tower = round(sc("camp1_push_tower_num", "tower"))
            c2_tower = round(sc("camp2_push_tower_num", "tower"))

            # 野怪
            c1_tyrant = round(sc("camp1_kill_tyrant_num", "tyrant"))
            c2_tyrant = round(sc("camp2_kill_tyrant_num", "tyrant"))
            c1_dark = round(sc("camp1_kill_dark_tyrant_num", "dark_tyrant"))
            c2_dark = round(sc("camp2_kill_dark_tyrant_num", "dark_tyrant"))
            c1_lord = round(sc("camp1_kill_big_dragon_num", "lord"))
            c2_lord = round(sc("camp2_kill_big_dragon_num", "lord"))
            c1_proph = round(sc("camp1_kill_prophet_dragon_num", "prophet"))
            c2_proph = round(sc("camp2_kill_prophet_dragon_num", "prophet"))
            c1_shdw = round(sc("camp1_kill_shadow_dragon_num", "shadow"))
            c2_shdw = round(sc("camp2_kill_shadow_dragon_num", "shadow"))
            c1_storm = round(sc("camp1_kill_storm_dragon_king_num", "storm"))
            c2_storm = round(sc("camp2_kill_storm_dragon_king_num", "storm"))

            # 位置级
            pos_data = {}
            for _, p_name in POSITIONS.items():
                pos_data[f"c1_gold_{p_name}"] = round(sc(f"camp1_gold_{p_name}", "gold"))
                pos_data[f"c2_gold_{p_name}"] = round(sc(f"camp2_gold_{p_name}", "gold"))
                pos_data[f"c1_hurt_{p_name}"] = round(sc(f"camp1_hurt_{p_name}", "hurt"))
                pos_data[f"c2_hurt_{p_name}"] = round(sc(f"camp2_hurt_{p_name}", "hurt"))

            row = {
                "battle_id": b["battle_id"], "minute_bin": minute,
                "camp1_gold": c1_gold, "camp2_gold": c2_gold,
                "camp1_kill": c1_kill, "camp2_kill": c2_kill,
                "camp1_assist": c1_assist, "camp2_assist": c2_assist,
                "camp1_death": c1_death, "camp2_death": c2_death,
                "camp1_tower": c1_tower, "camp2_tower": c2_tower,
                "camp1_tyrant": c1_tyrant, "camp2_tyrant": c2_tyrant,
                "camp1_dark_tyrant": c1_dark, "camp2_dark_tyrant": c2_dark,
                "camp1_lord": c1_lord, "camp2_lord": c2_lord,
                "camp1_prophet": c1_proph, "camp2_prophet": c2_proph,
                "camp1_shadow": c1_shdw, "camp2_shadow": c2_shdw,
                "camp1_storm": c1_storm, "camp2_storm": c2_storm,
                "win_camp": b["win_camp"],
                **pos_data,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 特征工程
# ═══════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始快照构造模型特征"""
    df = df.copy()
    minute = df["minute_bin"].clip(lower=1)

    # ── 队伍级 ──
    df["gold_diff"] = df["camp1_gold"] - df["camp2_gold"]
    df["gold_diff_per_min"] = df["gold_diff"] / minute
    df["exp_diff_per_min"] = df["gold_diff"] / minute  # 经济差作为发育速度代理

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

    # ── 野怪 ──
    df["tyrant_diff"] = df["camp1_tyrant"] - df["camp2_tyrant"]
    df["dark_tyrant_diff"] = df["camp1_dark_tyrant"] - df["camp2_dark_tyrant"]
    df["lord_diff"] = df["camp1_lord"] - df["camp2_lord"]
    df["prophet_diff"] = df["camp1_prophet"] - df["camp2_prophet"]
    df["shadow_diff"] = df["camp1_shadow"] - df["camp2_shadow"]
    df["storm_diff"] = df["camp1_storm"] - df["camp2_storm"]

    # ── 位置级 ──
    for _, p_name in POSITIONS.items():
        df[f"gold_diff_{p_name}"] = df[f"c1_gold_{p_name}"] - df[f"c2_gold_{p_name}"]
        df[f"hurt_diff_{p_name}"] = df[f"c1_hurt_{p_name}"] - df[f"c2_hurt_{p_name}"]

    # ── 衍生：核心 carry 经济统治力（中路+发育路 vs 对方）──
    c1_carry = df["c1_gold_mid"] + df["c1_gold_adc"]
    c2_carry = df["c2_gold_mid"] + df["c2_gold_adc"]
    total_carry = (c1_carry + c2_carry).clip(lower=1)
    df["carry_dominance"] = (c1_carry - c2_carry) / total_carry

    df["label"] = (df["win_camp"] == 1).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# 训练 + 评估
# ═══════════════════════════════════════════════════════════

def evaluate(name, y_true, y_prob):
    return {
        "name": name,
        "AUC": roc_auc_score(y_true, y_prob),
        "Acc": accuracy_score(y_true, (y_prob > 0.5).astype(int)),
        "LogLoss": log_loss(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
    }


def train_and_evaluate(df: pd.DataFrame):
    """训练 LR + GBDT + RF + Ensemble，按比赛切分"""
    bids = df["battle_id"].unique()
    np.random.seed(42)
    np.random.shuffle(bids)

    split = int(len(bids) * 0.75)
    train_bids = set(bids[:split])
    test_bids = set(bids[split:])

    train_mask = df["battle_id"].isin(train_bids)
    test_mask = df["battle_id"].isin(test_bids)

    X_train = df.loc[train_mask, FEATURE_COLUMNS].values
    y_train = df.loc[train_mask, "label"].values
    X_test = df.loc[test_mask, FEATURE_COLUMNS].values
    y_test = df.loc[test_mask, "label"].values

    print(f"\n  训练集: {len(X_train)} 条 ({len(train_bids)} 场)")
    print(f"  测试集: {len(X_test)} 条 ({len(test_bids)} 场)")

    # ── 1. LR（带 scaler）──
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=0.5, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    prob_lr = lr_pipe.predict_proba(X_test)[:, 1]

    # ── 2. GBDT ──
    gbdt = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.06,
        subsample=0.85, min_samples_leaf=20, random_state=42,
    )
    gbdt.fit(X_train, y_train)
    prob_gbdt = gbdt.predict_proba(X_test)[:, 1]

    # ── 3. RF ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]

    # ── 4. Ensemble（软投票）──
    voting = VotingClassifier(
        estimators=[("lr", lr_pipe), ("gbdt", gbdt), ("rf", rf)],
        voting="soft",
        weights=[1, 2, 1],  # GBDT 权重略高
    )
    voting.fit(X_train, y_train)
    prob_voting = voting.predict_proba(X_test)[:, 1]

    # ── 评估 ──
    results = {
        "LR": evaluate("LR", y_test, prob_lr),
        "GBDT": evaluate("GBDT", y_test, prob_gbdt),
        "RF": evaluate("RF", y_test, prob_rf),
        "Voting": evaluate("Voting", y_test, prob_voting),
    }

    print(f"\n  {'模型':<8} {'AUC':<8} {'Acc':<8} {'LogLoss':<10} {'Brier':<8}")
    print(f"  {'─' * 45}")
    for k, r in results.items():
        print(f"  {k:<8} {r['AUC']:<8.4f} {r['Acc']:<8.4f} {r['LogLoss']:<10.4f} {r['Brier']:<8.4f}")

    # 各模型 → 测试集分时间段 AUC
    df_test = df.loc[test_mask].copy()
    df_test["prob_voting"] = prob_voting
    print(f"\n  [Voting] AUC by minute_bin:")
    for mb in sorted(df_test["minute_bin"].unique()):
        sub = df_test[df_test["minute_bin"] == mb]
        if len(sub["label"].unique()) < 2:
            continue
        auc_mb = roc_auc_score(sub["label"], sub["prob_voting"])
        bar = "█" * int(auc_mb * 25)
        print(f"    {mb:2d}min ({len(sub):3d} 条): AUC={auc_mb:.3f} {bar}")

    return {
        "LR": lr_pipe,
        "GBDT": gbdt,
        "RF": rf,
        "Voting": voting,
    }, results


def show_feature_importance(models):
    """打印 GBDT 和 RF 的特征重要性"""
    for name in ["GBDT", "RF"]:
        model = models[name]
        importances = model.feature_importances_
        ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
        print(f"\n  [{name}] Top 15 特征重要性:")
        for fname, imp in ranked[:15]:
            bar = "█" * int(imp * 100)
            print(f"    {fname:<22} {imp:.4f} {bar}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  KPL 实时胜率预测模型 V5 — 完整特征 + 多模型集成")
    print("═" * 60)

    print("\n[1/5] 加载历史数据...")
    battles = pd.read_csv(PROCESSED_DIR / "battles.csv")
    players = pd.read_csv(PROCESSED_DIR / "players.csv")
    print(f"  battles: {len(battles)} 场")
    print(f"  players: {len(players)} 行（每场 10 人）")

    print("\n[2/5] 聚合玩家数据到位置级...")
    pos_df = aggregate_position_data(players)
    print(f"  位置聚合: {pos_df.shape}")

    print("\n[3/5] 模拟时序快照（幂律插值）...")
    snapshots = simulate_snapshots(battles, pos_df)
    print(f"  生成 {len(snapshots)} 条快照（{snapshots['battle_id'].nunique()} 场）")
    print(f"  分钟分布:\n{snapshots['minute_bin'].value_counts().sort_index().to_string()}")

    print("\n[4/5] 构造模型特征...")
    df = build_features(snapshots)
    print(f"  特征数: {len(FEATURE_COLUMNS)}")
    print(f"  特征列: {FEATURE_COLUMNS}")

    print("\n[5/5] 训练 LR + GBDT + RF + Voting 集成...")
    models, results = train_and_evaluate(df)

    show_feature_importance(models)

    # ── 保存 Voting 集成模型 ──
    artifact = {
        "model": models["Voting"],
        "model_name": "Voting (LR+GBDT+RF)",
        "feature_columns": FEATURE_COLUMNS,
        "scaling_alpha": SCALING_ALPHA,
        "positions": POSITIONS,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "name"} for k, v in results.items()},
        "sub_models": {
            "LR": models["LR"],
            "GBDT": models["GBDT"],
            "RF": models["RF"],
        },
    }
    out_path = MODEL_DIR / "v5_realtime_voting.joblib"
    joblib.dump(artifact, out_path)
    print(f"\n  ✅ 已保存: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

    print("\n" + "═" * 60)
    print("  训练完成。可以启动实时预测了。")
    print("═" * 60)


if __name__ == "__main__":
    main()
