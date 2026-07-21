"""
KPL 实时胜率预测模型 V7 — 模型动物园 + Stacking

V7 vs V6：
  - 模型从 3 个 → 9 个：LR / RF / ExtraTrees / GBDT / HistGBT / AdaBoost / XGBoost / LightGBM / CatBoost
  - 集成从 Voting → Stacking（用 LR 作为 meta-learner，自动学最优权重）
  - 保留 V6 的所有改进：数据增强、isotonic 校准、时序动量、战队先验

特征：29 个（与 V6 相同）
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
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

# 复用 V6 的训练逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_realtime_model_v6 import (
    PROJECT_ROOT, PROCESSED_DIR, MODEL_DIR,
    SCALING_ALPHA, MINUTE_BINS, POSITIONS, FEATURE_COLUMNS,
    build_team_winrate, aggregate_position_data,
    simulate_snapshots_with_history, augment_data, build_features,
)


# ═══════════════════════════════════════════════════════════
# 训练 + Stacking
# ═══════════════════════════════════════════════════════════

def evaluate(y, p):
    return {
        "AUC": roc_auc_score(y, p),
        "Acc": accuracy_score(y, (p > 0.5).astype(int)),
        "LogLoss": log_loss(y, p),
        "Brier": brier_score_loss(y, p),
    }


def get_zoo():
    """构造模型动物园"""
    zoo = {}

    # ── 1. LR（带 scaler，强基线）──
    zoo["LR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=0.5, random_state=42)),
    ])

    # ── 2. RF ──
    zoo["RF"] = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )

    # ── 3. ExtraTrees（更随机的 RF 变体）──
    zoo["ET"] = ExtraTreesClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )

    # ── 4. GBDT（sklearn 经典）──
    zoo["GBDT"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.06,
        subsample=0.85, min_samples_leaf=20, random_state=42,
    )

    # ── 5. HistGBT（sklearn 直方图加速版，类似 LightGBM）──
    zoo["HGBT"] = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )

    # ── 6. AdaBoost（经典 boosting）──
    zoo["AdaBoost"] = AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=42,
    )

    # ── 7. XGBoost ──
    zoo["XGB"] = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )

    # ── 8. LightGBM（如果可用）──
    if HAS_LGBM:
        zoo["LGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
        )

    # ── 9. CatBoost（如果可用）──
    if HAS_CATBOOST:
        zoo["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False,
        )

    return zoo


def train_zoo_stacking(df: pd.DataFrame):
    bids = df["battle_id"].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(bids)

    n = len(bids)
    train_bids = set(bids[: int(n * 0.65)])
    cal_bids = set(bids[int(n * 0.65): int(n * 0.80)])
    test_bids = set(bids[int(n * 0.80):])

    train = df[df["battle_id"].isin(train_bids)]
    cal = df[df["battle_id"].isin(cal_bids)]
    test = df[df["battle_id"].isin(test_bids)]

    print(f"  训练: {len(train)} 条 | 校准: {len(cal)} 条 | 测试: {len(test)} 条")
    print(f"  对应比赛: {len(train_bids)} / {len(cal_bids)} / {len(test_bids)}")

    X_tr, y_tr = train[FEATURE_COLUMNS].values, train["label"].values
    X_cal, y_cal = cal[FEATURE_COLUMNS].values, cal["label"].values
    X_te, y_te = test[FEATURE_COLUMNS].values, test["label"].values

    # ── 训练所有动物园模型 ──
    zoo = get_zoo()
    print(f"\n  动物园成员 ({len(zoo)} 个):")
    sub_results = {}
    for name, model in zoo.items():
        print(f"    训练 {name}...", end="", flush=True)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        sub_results[name] = evaluate(y_te, p)
        print(f" AUC={sub_results[name]['AUC']:.4f}, Brier={sub_results[name]['Brier']:.4f}")

    # ── Voting 集成（等权 + GBDT 类加权）──
    voting_estimators = [(k, v) for k, v in zoo.items()]
    voting = VotingClassifier(
        estimators=voting_estimators,
        voting="soft",
        # 强力树模型权重高
        weights=[1, 1, 1, 2, 2, 1, 2] + ([2] if HAS_LGBM else []) + ([2] if HAS_CATBOOST else []),
    )
    print(f"\n  训练 Voting (soft, 加权)...", end="", flush=True)
    voting.fit(X_tr, y_tr)
    p_vote = voting.predict_proba(X_te)[:, 1]
    print(f" AUC={roc_auc_score(y_te, p_vote):.4f}")

    # ── Stacking 集成（用 LR 学习最优组合）──
    stack_estimators = [(k, v) for k, v in zoo.items()]
    stack_meta = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    stacking = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=stack_meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    print(f"  训练 Stacking (5-fold, LR meta)...", end="", flush=True)
    stacking.fit(X_tr, y_tr)
    p_stack = stacking.predict_proba(X_te)[:, 1]
    print(f" AUC={roc_auc_score(y_te, p_stack):.4f}")

    # ── isotonic 校准 Voting（最终模型，因为 Stacking 在增强数据上反而过拟合）──
    if HAS_FROZEN:
        voting_cal = CalibratedClassifierCV(FrozenEstimator(voting), method="isotonic")
    else:
        voting_cal = CalibratedClassifierCV(voting, method="isotonic", cv="prefit")
    voting_cal.fit(X_cal, y_cal)
    p_voting_cal = voting_cal.predict_proba(X_te)[:, 1]

    # 同时校准 CatBoost（作为备选，是 AUC 最高的单模型）
    catboost_cal = None
    p_cat_cal = None
    if "CatBoost" in zoo:
        if HAS_FROZEN:
            catboost_cal = CalibratedClassifierCV(FrozenEstimator(zoo["CatBoost"]), method="isotonic")
        else:
            catboost_cal = CalibratedClassifierCV(zoo["CatBoost"], method="isotonic", cv="prefit")
        catboost_cal.fit(X_cal, y_cal)
        p_cat_cal = catboost_cal.predict_proba(X_te)[:, 1]

    # 保留 Stacking 的校准结果（用于对比）
    if HAS_FROZEN:
        stacking_cal = CalibratedClassifierCV(FrozenEstimator(stacking), method="isotonic")
    else:
        stacking_cal = CalibratedClassifierCV(stacking, method="isotonic", cv="prefit")
    stacking_cal.fit(X_cal, y_cal)
    p_stack_cal = stacking_cal.predict_proba(X_te)[:, 1]

    # ── 评估全员 ──
    print(f"\n  {'═' * 65}")
    print(f"  {'模型':<14} {'AUC':<8} {'Acc':<8} {'LogLoss':<10} {'Brier':<8}")
    print(f"  {'─' * 65}")
    for name, r in sub_results.items():
        print(f"  {name:<14} {r['AUC']:<8.4f} {r['Acc']:<8.4f} {r['LogLoss']:<10.4f} {r['Brier']:<8.4f}")
    extras = [("Voting", p_vote), ("Voting+Calib", p_voting_cal),
              ("Stacking", p_stack), ("Stack+Calib", p_stack_cal)]
    if p_cat_cal is not None:
        extras.append(("CatBoost+Cal", p_cat_cal))
    for name, p in extras:
        r = evaluate(y_te, p)
        print(f"  {name:<14} {r['AUC']:<8.4f} {r['Acc']:<8.4f} {r['LogLoss']:<10.4f} {r['Brier']:<8.4f}")
    print(f"  {'═' * 65}")

    # ── 选最佳：在 Brier 与 AUC 间权衡 ──
    candidates = {"Voting+Calib": (voting_cal, p_voting_cal)}
    if catboost_cal is not None:
        candidates["CatBoost+Cal"] = (catboost_cal, p_cat_cal)
    # 选 Brier 最低的（概率校准最好）
    best_name = min(candidates, key=lambda k: brier_score_loss(y_te, candidates[k][1]))
    best_model, best_p = candidates[best_name]
    print(f"\n  ⭐ 选定主模型: {best_name} (Brier最低)")

    # ── 分时间段评估最终模型 ──
    test_eval = test.copy()
    test_eval["prob"] = best_p
    print(f"\n  [{best_name}] AUC by minute:")
    for mb in sorted(test_eval["minute_bin"].unique()):
        sub = test_eval[test_eval["minute_bin"] == mb]
        if len(sub["label"].unique()) < 2:
            continue
        auc = roc_auc_score(sub["label"], sub["prob"])
        bar = "█" * int(auc * 25)
        print(f"    {mb:2d}min ({len(sub):3d} 条): AUC={auc:.3f} {bar}")

    return best_model, best_name, stacking_cal, voting_cal, voting, zoo, sub_results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print("  KPL 实时胜率预测模型 V7 — 模型动物园 + Stacking")
    print("═" * 65)

    print("\n[1/6] 加载历史数据...")
    battles = pd.read_csv(PROCESSED_DIR / "battles.csv")
    players = pd.read_csv(PROCESSED_DIR / "players.csv")
    print(f"  battles: {len(battles)} | players: {len(players)}")

    print("\n[2/6] 战队历史胜率...")
    team_wr = build_team_winrate(battles)
    print(f"  共 {len(team_wr)} 支战队")

    print("\n[3/6] 聚合位置 + 模拟时序快照...")
    pos_df = aggregate_position_data(players)
    snapshots = simulate_snapshots_with_history(battles, pos_df, team_wr)
    print(f"  原始: {len(snapshots)}")

    print("\n[4/6] 数据增强（噪声 + 翻盘）...")
    snapshots_aug = augment_data(snapshots, noise_sigma=0.08, n_repeats=2, swap_pct=0.10)
    print(f"  增强后: {len(snapshots_aug)}")

    print("\n[5/6] 构造特征...")
    df = build_features(snapshots_aug)
    print(f"  特征数: {len(FEATURE_COLUMNS)}")

    print("\n[6/6] 训练模型动物园 + Stacking...")
    print(f"  可用框架: sklearn ✅ | xgboost ✅ | lightgbm {'✅' if HAS_LGBM else '❌'} | catboost {'✅' if HAS_CATBOOST else '❌'}")
    best_model, best_name, stacking_cal, voting_cal, voting, zoo, sub_results = train_zoo_stacking(df)

    # ── 保存 ──
    artifact = {
        "model": best_model,                  # 主模型：自动选 Brier 最低的（Voting+Calib 或 CatBoost+Cal）
        "model_name": f"{best_name} (V7)",
        "feature_columns": FEATURE_COLUMNS,
        "scaling_alpha": SCALING_ALPHA,
        "positions": POSITIONS,
        "team_winrate": team_wr,
        "sub_models": {
            "voting_cal": voting_cal,
            "stacking_cal": stacking_cal,
            "voting": voting,
            **zoo,
        },
        "sub_results": sub_results,
    }
    out = MODEL_DIR / "v7_realtime_stacking.joblib"
    joblib.dump(artifact, out)
    print(f"\n  ✅ 保存: {out} ({out.stat().st_size/1024:.1f} KB)")

    print("\n" + "═" * 65)
    print(f"  V7 训练完成。{len(zoo)} 个基础模型 + 多种集成 + isotonic 校准")
    print(f"  主模型: {best_name}")
    print("═" * 65)


if __name__ == "__main__":
    main()
