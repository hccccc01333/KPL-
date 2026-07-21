"""用 V8 模型回测决赛所有局，对比 V7 预测文件"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "output" / "models"
RAW_DIR = PROJECT_ROOT / "data" / "realtime" / "raw_snapshots"
POSITIONS = {2: "mid", 4: "support", 5: "jungle", 6: "top", 7: "adc"}

# Load V8 model
model_path = MODEL_DIR / "v8_realtime_enhanced.joblib"
art = joblib.load(model_path)
feat_cols = art["feature_columns"]
print(f"Model: {art['model_name']} | Features: {len(feat_cols)}")
print(f"Time shrinkage: {art.get('use_time_shrinkage', False)}")


def parse_snapshot(data, battle_id):
    game_ms = data.get("game_duration", 0) or 0
    game_sec = game_ms / 1000 if game_ms > 1000 else game_ms
    minute_bin = max(int(game_sec / 60), 1)
    c1 = data.get("camp1", {})
    c2 = data.get("camp2", {})
    snap = {
        "battle_id": battle_id,
        "snapshot_time_sec": int(game_sec),
        "minute_bin": minute_bin,
        "camp1_team": c1.get("team_name", "Camp1"),
        "camp2_team": c2.get("team_name", "Camp2"),
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
        "status": data.get("status", 0),
        "win_camp": data.get("win_camp", 0),
    }
    players = data.get("battle_player_list", []) or []
    for pos_id, pos_name in POSITIONS.items():
        c1_p = [p for p in players if p.get("camp") == 1 and p.get("position") == pos_id]
        c2_p = [p for p in players if p.get("camp") == 2 and p.get("position") == pos_id]
        snap[f"c1_gold_{pos_name}"] = sum(p.get("gold", 0) or 0 for p in c1_p)
        snap[f"c2_gold_{pos_name}"] = sum(p.get("gold", 0) or 0 for p in c2_p)
        snap[f"c1_hurt_{pos_name}"] = sum(p.get("hurt_to_hero_total", 0) or 0 for p in c1_p)
        snap[f"c2_hurt_{pos_name}"] = sum(p.get("hurt_to_hero_total", 0) or 0 for p in c2_p)
    return snap


def compute_features(snap, history, team_wr):
    minute = max(snap["minute_bin"], 1)
    gold_diff = snap["camp1_gold"] - snap["camp2_gold"]
    kill_diff = snap["camp1_kill"] - snap["camp2_kill"]
    assist_diff = snap["camp1_assist"] - snap["camp2_assist"]
    death_diff = snap["camp1_death"] - snap["camp2_death"]
    tower_diff = snap["camp1_tower"] - snap["camp2_tower"]
    total_gold = max(snap["camp1_gold"] + snap["camp2_gold"], 1)
    total_kills = max(snap["camp1_kill"] + snap["camp2_kill"], 1)
    c1_kda = (snap["camp1_kill"] + snap["camp1_assist"]) / max(snap["camp1_death"], 1)
    c2_kda = (snap["camp2_kill"] + snap["camp2_assist"]) / max(snap["camp2_death"], 1)

    cur_t = snap["snapshot_time_sec"]
    prev_snap = None
    for h in reversed(history):
        if h["snapshot_time_sec"] < cur_t:
            prev_snap = h
            break
    if prev_snap:
        prev_gold_diff = prev_snap["camp1_gold"] - prev_snap["camp2_gold"]
        dt = max(cur_t - prev_snap["snapshot_time_sec"], 1) / 60.0
        gold_diff_delta = gold_diff - prev_gold_diff
        gold_diff_velocity = gold_diff_delta / max(dt, 0.1)
    else:
        gold_diff_delta = 0.0
        gold_diff_velocity = 0.0

    c1_wr = team_wr.get(snap.get("camp1_team", ""), 0.5)
    c2_wr = team_wr.get(snap.get("camp2_team", ""), 0.5)

    feats = {
        "gold_diff_per_min": gold_diff / minute,
        "gold_ratio": gold_diff / total_gold,
        "kill_diff_per_min": kill_diff / minute,
        "kill_rate": kill_diff / total_kills,
        "assist_diff_per_min": assist_diff / minute,
        "death_diff": death_diff,
        "kda_diff": c1_kda - c2_kda,
        "tower_diff": tower_diff,
        "minute_bin": minute,
        "gold_diff_delta": gold_diff_delta,
        "gold_diff_velocity": gold_diff_velocity,
        "tyrant_diff": snap["camp1_tyrant"] - snap["camp2_tyrant"],
        "dark_tyrant_diff": snap["camp1_dark_tyrant"] - snap["camp2_dark_tyrant"],
        "lord_diff": snap["camp1_lord"] - snap["camp2_lord"],
        "prophet_diff": snap["camp1_prophet"] - snap["camp2_prophet"],
        "shadow_diff": snap["camp1_shadow"] - snap["camp2_shadow"],
        "storm_diff": snap["camp1_storm"] - snap["camp2_storm"],
        "team_winrate_diff": c1_wr - c2_wr,
    }
    for _, p_name in POSITIONS.items():
        feats[f"gold_diff_{p_name}"] = snap[f"c1_gold_{p_name}"] - snap[f"c2_gold_{p_name}"]
        feats[f"hurt_diff_{p_name}"] = snap[f"c1_hurt_{p_name}"] - snap[f"c2_hurt_{p_name}"]

    c1_carry = snap["c1_gold_mid"] + snap["c1_gold_adc"]
    c2_carry = snap["c2_gold_mid"] + snap["c2_gold_adc"]
    feats["carry_dominance"] = (c1_carry - c2_carry) / max(c1_carry + c2_carry, 1)

    feats["objective_value_score"] = (
        (snap["camp1_lord"] - snap["camp2_lord"]) * 5.0 +
        (snap["camp1_dark_tyrant"] - snap["camp2_dark_tyrant"]) * 4.0 +
        (snap["camp1_storm"] - snap["camp2_storm"]) * 3.5 +
        (snap["camp1_tyrant"] - snap["camp2_tyrant"]) * 2.0 +
        (snap["camp1_prophet"] - snap["camp2_prophet"]) * 1.5 +
        (snap["camp1_shadow"] - snap["camp2_shadow"]) * 1.0
    )

    lane_abs = [abs(feats.get(f"gold_diff_{p}", 0)) for _, p in POSITIONS.items()]
    feats["lane_dominance_max"] = max(lane_abs) / total_gold if lane_abs else 0
    feats["exp_diff_per_min"] = gold_diff / minute

    return feats


def predict_with_shrinkage(art, features):
    X = np.array([[features.get(c, 0) for c in art["feature_columns"]]])
    p_raw = art["model"].predict_proba(X)[0, 1]
    minute = features["minute_bin"]

    if art.get("use_time_shrinkage", False):
        if minute <= 2:
            conf = 0.4
        elif minute <= 5:
            conf = 0.5 + (minute - 2) * 0.1
        elif minute <= 8:
            conf = 0.8 + (minute - 5) * 0.067
        else:
            conf = 1.0
        p_main = 0.5 + (p_raw - 0.5) * conf
    else:
        p_main = p_raw

    return float(np.clip(p_main, 0.02, 0.98)), float(np.clip(p_raw, 0.02, 0.98))


# Run backtest
team_wr = art.get("team_winrate", {})
battle_dirs = sorted(RAW_DIR.iterdir())

all_correct = []
all_brier = []
game_results = []

for bd in battle_dirs:
    jsons = sorted(bd.glob("*.json"))
    if len(jsons) < 10:
        continue

    battle_id = bd.name
    # Determine winner
    win_camp = 0
    for jf in reversed(jsons):
        with open(jf, "r", encoding="utf-8") as f:
            try:
                d = json.load(f)
            except:
                continue
        data = d.get("data", {})
        if data.get("status") == 2:
            win_camp = data.get("win_camp", 0)
            break
    if win_camp == 0:
        continue

    # Process snapshots
    history = deque(maxlen=20)
    seen_minutes = {}
    t1_name = None

    for jf in jsons:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                d = json.load(f)
            except:
                continue
        data = d.get("data", {})
        if data.get("status", 0) not in (1, 2):
            continue

        snap = parse_snapshot(data, battle_id)
        if t1_name is None:
            t1_name = snap["camp1_team"]

        mb = snap["minute_bin"]
        if mb in seen_minutes:
            continue
        seen_minutes[mb] = True

        feats = compute_features(snap, history, team_wr)
        p_main, p_raw = predict_with_shrinkage(art, feats)

        label = 1 if win_camp == 1 else 0
        correct = (p_main > 0.5 and win_camp == 1) or (p_main < 0.5 and win_camp == 2)
        brier = (p_main - label) ** 2
        all_correct.append(correct)
        all_brier.append(brier)

        history.append(snap)

    t2_name = snap["camp2_team"] if snap else "?"
    winner = t1_name if win_camp == 1 else t2_name
    n_correct = sum(1 for c in all_correct[-len(seen_minutes):] if c)
    n_total_g = len(seen_minutes)
    game_results.append((battle_id, t1_name, t2_name, winner, n_correct, n_total_g))

# Report
print(f"\n{'='*60}")
print(f"V8 回测报告（决赛全部局）")
print(f"{'='*60}")
for bid, t1, t2, winner, nc, nt in game_results:
    acc = nc / nt * 100 if nt > 0 else 0
    print(f"  {t1} vs {t2} | winner={winner} | {nc}/{nt}={acc:.0f}%")

n_total = len(all_correct)
n_ok = sum(all_correct)
avg_brier = np.mean(all_brier)
print(f"\n{'='*60}")
print(f"  总方向正确率: {n_ok}/{n_total} = {n_ok/n_total*100:.1f}%")
print(f"  平均 Brier Score: {avg_brier:.4f}")
print(f"{'='*60}")
