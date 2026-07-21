"""用第 2 局真实历史 JSON 验证 V6 链路（含时序动量）"""
import sys, os; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_live import (
    parse_battle_data, compute_features, predict_all_models, load_model
)

art = load_model()
team_wr = art.get("team_winrate", {})

raw_dir = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\raw_snapshots')
target = "1373651472_21_1779526666"  # 第 2 局
target_dir = raw_dir / target
jsons = sorted(target_dir.glob('*.json'))
print(f"\n=== 第 2 局回放：{len(jsons)} 个快照 ===\n")

# 测试关键时间点（早期 / 翻盘前 / 翻盘后 / 结束）
key_indices = [
    int(len(jsons) * 0.10),  # 早期 ~ 4min
    int(len(jsons) * 0.35),  # 翻盘前 ~ 10min  
    int(len(jsons) * 0.55),  # 翻盘后 ~ 14min
    int(len(jsons) * 0.85),  # 后期 ~ 18min
    len(jsons) - 1,           # 结束
]

# 维护历史 buffer 模拟实时
history = deque(maxlen=30)

print(f"{'时间':<10} {'经济差':<10} {'动量':<10} {'LR':<8} {'GBDT':<8} {'RF':<8} {'Voting':<10}")
print("─" * 70)

# 简化：每隔 N 个快照预测一次（模拟 10s 间隔）
step = max(1, len(jsons) // 25)
for idx in range(0, len(jsons), step):
    with open(jsons[idx], encoding='utf-8') as f:
        data = json.load(f)
    snap = parse_battle_data(data.get('data', {}), target)
    if snap["snapshot_time_sec"] == 0:
        continue

    feats = compute_features(snap, history, team_wr)
    probs = predict_all_models(art, feats)

    g_sec = snap["snapshot_time_sec"]
    gold_diff = snap["camp1_gold"] - snap["camp2_gold"]
    velocity = feats["gold_diff_velocity"]
    
    print(f"{g_sec//60}:{g_sec%60:02d}      "
          f"{gold_diff:+6d}    "
          f"{velocity:+7.1f}   "
          f"{probs.get('LR',0)*100:5.1f}%  "
          f"{probs.get('GBDT',0)*100:5.1f}%  "
          f"{probs.get('RF',0)*100:5.1f}%  "
          f"{probs['Voting']*100:5.1f}%")
    
    history.append(snap)

# 终局
print("\n[终局信息]")
with open(jsons[-1], encoding='utf-8') as f:
    data = json.load(f)
snap = parse_battle_data(data.get('data', {}), target)
print(f"实际获胜: camp{snap['win_camp']} ({snap['camp1_team'] if snap['win_camp']==1 else snap['camp2_team']})")
feats = compute_features(snap, history, team_wr)
probs = predict_all_models(art, feats)
print(f"V6 终局预测: Voting={probs['Voting']:.1%}")
ok = (probs["Voting"] > 0.5 and snap["win_camp"] == 1) or \
     (probs["Voting"] < 0.5 and snap["win_camp"] == 2)
print(f"{'✅ 正确' if ok else '❌ 错误'}")
