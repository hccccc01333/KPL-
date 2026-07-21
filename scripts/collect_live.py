"""
KPL 实时赛中数据采集脚本

用法：
  1. 自动发现正在进行的比赛并采集：
     python scripts/collect_live.py

  2. 指定 battle_id 采集：
     python scripts/collect_live.py --battle_id 736117264_39_1778333000

  3. 指定 match_id，自动轮询该 match 下所有 battle：
     python scripts/collect_live.py --match_id 2026052301

  4. 调整采集间隔（默认 30 秒）：
     python scripts/collect_live.py --interval 20
"""

import argparse
import json
import time
import random
import sys
import os
from datetime import datetime
from pathlib import Path

# Windows 终端编码修复 + 强制行缓冲
if sys.platform == "win32":
    os.system("")  # enable ANSI escape
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import requests
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
RAW_SNAPSHOT_DIR = REALTIME_DIR / "raw_snapshots"

REALTIME_DIR.mkdir(parents=True, exist_ok=True)
RAW_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL_BATTLE = "https://prod.comp.smoba.qq.com/leaguesite/battle/open"
BASE_URL_LEAGUE = "https://prod.comp.smoba.qq.com/leaguesite/matches/open"
BASE_URL_MATCH = "https://prod.comp.smoba.qq.com/leaguesite/match/battles/open"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://pvp.qq.com/",
    "Origin": "https://pvp.qq.com",
}

LEAGUE_ID = 20260003  # 2026 KPL夏季赛

SNAPSHOT_COLUMNS = [
    "battle_id", "collected_at", "snapshot_time_sec", "minute_bin",
    "camp1_team", "camp2_team",
    "camp1_gold", "camp2_gold", "gold_diff",
    "camp1_kill_num", "camp2_kill_num", "kill_diff",
    "camp1_push_tower_num", "camp2_push_tower_num", "tower_diff",
    "camp1_tyrant", "camp2_tyrant", "tyrant_diff",
    "camp1_lord", "camp2_lord", "lord_diff",
    "status", "win_camp", "is_simulated",
]


# ──────────────────────────────────────────────────────────────
# 网络请求封装
# ──────────────────────────────────────────────────────────────

def safe_get(url: str, params: dict = None, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 200:
                return data
            print(f"  [API] 业务错误: code={data.get('code')}, msg={data.get('message')}")
            return None
        except requests.Timeout:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [超时] 第 {attempt+1} 次，等待 {wait:.1f}s 后重试...")
            time.sleep(wait)
        except requests.ConnectionError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [连接失败] 第 {attempt+1} 次，等待 {wait:.1f}s 后重试...")
            time.sleep(wait)
        except requests.RequestException as e:
            print(f"  [请求异常] {e}")
            return None
    print(f"  [失败] {max_retries} 次重试均失败")
    return None


# ──────────────────────────────────────────────────────────────
# 赛程发现：找到正在进行的比赛
# ──────────────────────────────────────────────────────────────

def discover_live_matches() -> list[dict]:
    """从赛程 API 找到当前正在进行（status != 2）的 match"""
    print("=" * 60)
    print("正在查询赛程，寻找进行中的比赛...")
    print("=" * 60)

    data = safe_get(BASE_URL_LEAGUE, {"league_id": LEAGUE_ID})
    if not data:
        print("  赛程接口请求失败")
        return []

    matches = data.get("results", [])
    if not matches:
        print("  未找到任何 match")
        return []

    live = [m for m in matches if m.get("status") != 2]
    completed = [m for m in matches if m.get("status") == 2]

    print(f"  总共 {len(matches)} 场 match，已完赛 {len(completed)} 场")

    if live:
        print(f"\n  🔴 发现 {len(live)} 场未完赛的 match：")
        for m in live:
            camp1 = m.get("camp1", {})
            camp2 = m.get("camp2", {})
            t1 = camp1.get("team_name", "?") if isinstance(camp1, dict) else "?"
            t2 = camp2.get("team_name", "?") if isinstance(camp2, dict) else "?"
            print(f"    match_id={m['match_id']} | {t1} vs {t2} | "
                  f"status={m.get('status')} | start={m.get('start_time')}")
    else:
        print("\n  当前没有正在进行的 match。")
        print("  显示最近 3 场已完赛的 match：")
        for m in completed[-3:]:
            camp1 = m.get("camp1", {})
            camp2 = m.get("camp2", {})
            t1 = camp1.get("team_name", "?") if isinstance(camp1, dict) else "?"
            t2 = camp2.get("team_name", "?") if isinstance(camp2, dict) else "?"
            print(f"    match_id={m['match_id']} | {t1} vs {t2} | end={m.get('end_time')}")

    return live


def get_battle_ids_for_match(match_id: str) -> list[str]:
    """获取某个 match 下所有 battle_id"""
    data = safe_get(BASE_URL_MATCH, {"match_id": match_id})
    if not data:
        return []
    results = data.get("results", [])
    return [r["battle_id"] for r in results if "battle_id" in r]


def find_live_battle(battle_ids: list[str]) -> str | None:
    """从 battle_id 列表中找到正在进行的那一场（status != 2）"""
    for bid in battle_ids:
        data = safe_get(BASE_URL_BATTLE, {"battle_id": bid})
        if not data:
            continue
        battle = data.get("data", {})
        status = battle.get("status", 0)
        if status != 2:
            return bid
        time.sleep(0.5)
    return None


# ──────────────────────────────────────────────────────────────
# 解析快照
# ──────────────────────────────────────────────────────────────

def parse_snapshot(raw_data: dict, battle_id: str) -> dict:
    """从 API data 字段解析一行快照"""
    game_ms = raw_data.get("game_duration", 0)
    game_sec = game_ms / 1000 if game_ms > 1000 else game_ms
    minute_bin = int(game_sec / 60)

    camp1 = raw_data.get("camp1", {})
    camp2 = raw_data.get("camp2", {})

    c1_gold = camp1.get("gold", 0) or 0
    c2_gold = camp2.get("gold", 0) or 0
    c1_kill = camp1.get("kill_num", 0) or 0
    c2_kill = camp2.get("kill_num", 0) or 0
    c1_tower = camp1.get("push_tower_num", 0) or 0
    c2_tower = camp2.get("push_tower_num", 0) or 0
    c1_tyrant = camp1.get("kill_tyrant_num", 0) or 0
    c2_tyrant = camp2.get("kill_tyrant_num", 0) or 0
    c1_lord = camp1.get("kill_big_dragon_num", 0) or 0
    c2_lord = camp2.get("kill_big_dragon_num", 0) or 0

    return {
        "battle_id": battle_id,
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_time_sec": int(game_sec),
        "minute_bin": minute_bin,
        "camp1_team": camp1.get("team_name", ""),
        "camp2_team": camp2.get("team_name", ""),
        "camp1_gold": c1_gold,
        "camp2_gold": c2_gold,
        "gold_diff": c1_gold - c2_gold,
        "camp1_kill_num": c1_kill,
        "camp2_kill_num": c2_kill,
        "kill_diff": c1_kill - c2_kill,
        "camp1_push_tower_num": c1_tower,
        "camp2_push_tower_num": c2_tower,
        "tower_diff": c1_tower - c2_tower,
        "camp1_tyrant": c1_tyrant,
        "camp2_tyrant": c2_tyrant,
        "tyrant_diff": c1_tyrant - c2_tyrant,
        "camp1_lord": c1_lord,
        "camp2_lord": c2_lord,
        "lord_diff": c1_lord - c2_lord,
        "status": raw_data.get("status", 0),
        "win_camp": raw_data.get("win_camp", 0),
        "is_simulated": False,
    }


# ──────────────────────────────────────────────────────────────
# 采集主循环
# ──────────────────────────────────────────────────────────────

def collect_battle(battle_id: str, interval_sec: int = 30, max_rounds: int = 80):
    """
    对单场 battle 持续采集，直到比赛结束或达到最大轮次。

    采集产出：
      - data/realtime/raw_snapshots/{battle_id}/{timestamp}.json  （每次原始响应）
      - data/realtime/live_snapshots.csv                          （追加式结构化数据）
    """
    battle_dir = RAW_SNAPSHOT_DIR / battle_id
    battle_dir.mkdir(parents=True, exist_ok=True)
    csv_path = REALTIME_DIR / "live_snapshots.csv"

    print("\n" + "=" * 60)
    print(f"🎮 开始采集 battle_id = {battle_id}")
    print(f"   间隔: {interval_sec}s | 最大轮次: {max_rounds}")
    print(f"   原始 JSON: {battle_dir}")
    print(f"   CSV 输出:  {csv_path}")
    print("=" * 60 + "\n")

    snapshots = []
    prev_game_sec = -1

    for round_i in range(max_rounds):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 调 API
        data = safe_get(BASE_URL_BATTLE, {"battle_id": battle_id})
        if not data:
            print(f"  [{ts}] ⚠️  API 无响应，{interval_sec}s 后重试...")
            time.sleep(interval_sec)
            continue

        battle_data = data.get("data", {})

        # 保存原始 JSON
        json_path = battle_dir / f"{ts}.json"
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 解析快照
        snap = parse_snapshot(battle_data, battle_id)
        snapshots.append(snap)

        # 打印实时状态
        game_sec = snap["snapshot_time_sec"]
        status_emoji = "🔴 进行中" if snap["status"] != 2 else "🏁 已结束"
        gold_sign = "+" if snap["gold_diff"] >= 0 else ""

        print(
            f"  [{ts}] {status_emoji} | "
            f"{snap['camp1_team']} vs {snap['camp2_team']} | "
            f"{game_sec // 60}:{game_sec % 60:02d} | "
            f"经济差: {gold_sign}{snap['gold_diff']} | "
            f"人头: {snap['camp1_kill_num']}-{snap['camp2_kill_num']} | "
            f"塔: {snap['camp1_push_tower_num']}-{snap['camp2_push_tower_num']} | "
            f"暴君: {snap['camp1_tyrant']}-{snap['camp2_tyrant']}"
        )

        # 比赛结束
        if snap["status"] == 2:
            winner = snap["win_camp"]
            winner_team = snap["camp1_team"] if winner == 1 else snap["camp2_team"]
            print(f"\n  🏆 比赛结束！获胜方: {winner_team} (camp{winner})")
            print(f"  📊 本场共采集 {len(snapshots)} 个快照点")
            break

        # 检测数据是否有变化（避免比赛还没开始时的无效轮询）
        if game_sec == prev_game_sec and game_sec == 0:
            print(f"       ⏳ 比赛似乎还未开始，等待中...")
        prev_game_sec = game_sec

        time.sleep(interval_sec)

    else:
        print(f"\n  ⚠️  达到最大轮次 {max_rounds}，停止采集")

    # 写入 CSV
    if snapshots:
        df_new = pd.DataFrame(snapshots)
        if csv_path.exists():
            df_new.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ 已追加 {len(df_new)} 行到 {csv_path}")
        print(f"     CSV 总行数: {len(pd.read_csv(csv_path))}")

    return snapshots


def collect_match(match_id: str, interval_sec: int = 30):
    """
    对一个 match 下的所有 battle 依次采集。

    逻辑：
      持续轮询 match 的 battle 列表，遇到新 battle 就检查状态：
      - 已完赛 → 跳过
      - 进行中/未开始 → 开始采集
      BO5 可能有 3~5 局，每局之间有 5-10 分钟间隔。
    """
    print(f"\n正在获取 match_id={match_id} 的 battle 列表...")
    collected_battles = set()
    max_wait_rounds = 60  # 最多等 60 分钟（60 × 60s）

    for wait_round in range(max_wait_rounds):
        battle_ids = get_battle_ids_for_match(match_id)

        if not battle_ids:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] 尚无 battle，60s 后重查...")
            time.sleep(60)
            continue

        # 找到还没采集过的 battle
        new_battles = [b for b in battle_ids if b not in collected_battles]

        if not new_battles:
            # 检查 match 整体是否结束了
            schedule_data = safe_get(BASE_URL_LEAGUE, {"league_id": LEAGUE_ID})
            if schedule_data:
                matches = schedule_data.get("results", [])
                this_match = next((m for m in matches if str(m.get("match_id")) == str(match_id)), None)
                if this_match and this_match.get("status") == 2:
                    print(f"\n✅ match {match_id} 整场结束！共采集 {len(collected_battles)} 局")
                    return

            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"已采集 {len(collected_battles)} 局，等待新 battle... (30s)")
            time.sleep(30)
            continue

        print(f"\n  发现 {len(battle_ids)} 个 battle（新增 {len(new_battles)}）: {battle_ids}")

        for bid in new_battles:
            print(f"\n{'─' * 40}")
            print(f"检查 battle: {bid}")

            data = safe_get(BASE_URL_BATTLE, {"battle_id": bid})
            if not data:
                print(f"  请求失败，稍后重试")
                continue

            battle_data = data.get("data", {})
            status = battle_data.get("status", 0)

            if status == 2:
                # 已完赛，保存最终数据但不轮询
                print(f"  已完赛，保存最终快照")
                snap = parse_snapshot(battle_data, bid)
                df_snap = pd.DataFrame([snap])
                csv_path = REALTIME_DIR / "live_snapshots.csv"
                if csv_path.exists():
                    df_snap.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
                else:
                    df_snap.to_csv(csv_path, index=False, encoding="utf-8-sig")
                # 同时保存原始 JSON
                battle_dir = RAW_SNAPSHOT_DIR / bid
                battle_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                (battle_dir / f"{ts}_final.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                collected_battles.add(bid)
                continue

            # 正在进行或即将开始，开始采集
            collect_battle(bid, interval_sec=interval_sec)
            collected_battles.add(bid)

            # 采完后短暂等待，让下一局有时间出现
            print(f"\n  等待 20s 后检查下一局...")
            time.sleep(20)

    print(f"\n⚠️  等待超时（{max_wait_rounds} 轮），退出。已采集 {len(collected_battles)} 局")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KPL 实时赛中数据采集")
    parser.add_argument("--battle_id", type=str, help="直接指定 battle_id 采集")
    parser.add_argument("--match_id", type=str, help="指定 match_id，自动轮询所有 battle")
    parser.add_argument("--interval", type=int, default=30, help="采集间隔（秒），默认 30")
    parser.add_argument("--discover", action="store_true", help="自动发现正在进行的比赛")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  KPL 实时赛中数据采集器 v1.0")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # 模式 1：直接指定 battle_id
    if args.battle_id:
        collect_battle(args.battle_id, interval_sec=args.interval)
        return

    # 模式 2：指定 match_id
    if args.match_id:
        collect_match(args.match_id, interval_sec=args.interval)
        return

    # 模式 3：自动发现
    live_matches = discover_live_matches()

    if not live_matches:
        print("\n没有发现正在进行的比赛。")
        print("你可以手动指定：")
        print("  python scripts/collect_live.py --battle_id <battle_id>")
        print("  python scripts/collect_live.py --match_id <match_id>")

        # 交互式选择
        user_input = input("\n或者直接输入 battle_id / match_id（回车退出）: ").strip()
        if not user_input:
            return
        if "_" in user_input:
            collect_battle(user_input, interval_sec=args.interval)
        else:
            collect_match(user_input, interval_sec=args.interval)
        return

    # 自动选第一个 live match
    match = live_matches[0]
    match_id = match["match_id"]
    camp1 = match.get("camp1", {})
    camp2 = match.get("camp2", {})
    t1 = camp1.get("team_name", "?") if isinstance(camp1, dict) else "?"
    t2 = camp2.get("team_name", "?") if isinstance(camp2, dict) else "?"

    print(f"\n自动选择: {t1} vs {t2} (match_id={match_id})")
    collect_match(str(match_id), interval_sec=args.interval)


if __name__ == "__main__":
    main()
