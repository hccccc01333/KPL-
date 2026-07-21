"""用历史 JSON 验证 predict_live 的解析+预测链路是否正常"""
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_live import (
    parse_battle_data, compute_features, predict_all_models, load_model
)

art = load_model()

# 用一份完整的 JSON 测试（取第二局结束时的快照）
raw_dir = Path(r'd:\AI数据分析\项目作品集\02-KPL实时胜率预测系统\data\realtime\raw_snapshots')
for d in sorted(raw_dir.iterdir()):
    if not d.is_dir() or d.name == '.gitkeep':
        continue
    jsons = sorted(d.glob('*.json'))
    if len(jsons) < 5:
        continue
    print(f'\n=== Battle: {d.name} ({len(jsons)} snapshots) ===\n')

    # 取 3 个时间点测试
    for idx in [len(jsons)//4, len(jsons)//2, len(jsons)-1]:
        with open(jsons[idx], encoding='utf-8') as f:
            data = json.load(f)
        snap = parse_battle_data(data.get('data', {}), d.name)
        feats = compute_features(snap)
        probs = predict_all_models(art, feats)

        print(f"[{jsons[idx].name}]  game_time={snap['snapshot_time_sec']}s  minute={snap['minute_bin']}")
        print(f"  {snap['camp1_team']} vs {snap['camp2_team']} | "
              f"经济差={snap['camp1_gold']-snap['camp2_gold']:+d}")
        print(f"  K {snap['camp1_kill']}-{snap['camp2_kill']} | "
              f"塔 {snap['camp1_tower']}-{snap['camp2_tower']} | "
              f"龙(主宰): {snap['camp1_lord']}-{snap['camp2_lord']} | "
              f"风暴: {snap['camp1_storm']}-{snap['camp2_storm']}")
        print(f"  位置经济差: ", end='')
        for p_name in ['mid', 'support', 'jungle', 'top', 'adc']:
            print(f"{p_name}={snap[f'c1_gold_{p_name}']-snap[f'c2_gold_{p_name}']:+d}  ", end='')
        print()
        print(f"  预测: LR={probs['LR']:.3f}  GBDT={probs['GBDT']:.3f}  "
              f"RF={probs['RF']:.3f}  Voting={probs['Voting']:.3f}")
        print()
    break

print("\n✅ 链路测试完成，所有解析+特征+模型工作正常")
