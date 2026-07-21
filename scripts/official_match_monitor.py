"""
KPL 官方赛程自动监控与实时采集脚本

能力：
- 自动从官方赛程接口识别进行中或最近一场比赛；
- 自动发现当前 battle_id；
- 多副本探测获取最新局内数据；
- 保存 raw snapshots、实时预测 CSV、monitor_status.json；
- 一场 BO 全部结束后，等待下一场开赛时间并自动切换预测；
- 当日全部比赛结束后统一重训一次模型，并热加载新模型（非每场/每局都训练）。

运行示例：
    python scripts/official_match_monitor.py --interval 15
    python scripts/official_match_monitor.py --no-auto-train
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from kpl_official_core import (
    LEAGUE_ID,
    MONOTONIC_TOLERANCE_SEC,
    PREDICTION_CSV_FIELDS,
    PREDICTION_DIR,
    RAW_DIR,
    REALTIME_DIR,
    ScheduleCenter,
    ScheduleMatch,
    adaptive_poll_interval,
    compute_lag_sec,
    fetch_freshest_battle,
    is_monotonic_snapshot,
    load_model,
    parse_snapshot,
    predict_probability,
    refresh_schedule_archive,
    sanitize_lag_sec,
)
from train_realtime_model_v9 import train_v9_model


STATUS_FILE = REALTIME_DIR / "monitor_status.json"
TRAIN_STATUS_FILE = REALTIME_DIR / "train_status.json"
_SAVE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kpl-save")
_TRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kpl-train")


@dataclass
class PipelineContext:
    artifact: dict
    auto_train: bool = True
    train_after_matches: int = 0
    trained_session: bool = False
    training: bool = False
    last_data_sig: tuple[int, float] | None = None
    battle_anchors: dict[str, tuple[datetime, int]] = field(default_factory=dict)
    last_velocity_by_battle: dict[str, float] = field(default_factory=dict)
    last_snapshot_sec: dict[str, int] = field(default_factory=dict)
    last_lag_sec: dict[str, float] = field(default_factory=dict)


def write_status(payload: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_raw_snapshot(battle_id: str, payload: dict) -> Path:
    out_dir = RAW_DIR / battle_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_raw_snapshot_async(battle_id: str, payload: dict) -> None:
    _SAVE_EXECUTOR.submit(save_raw_snapshot, battle_id, payload)


def append_prediction(battle_id: str, row: dict):
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTION_DIR / f"{battle_id}.csv"
    exists = path.exists()
    full_row = {col: row.get(col, "") for col in PREDICTION_CSV_FIELDS}
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


def next_poll_interval(ctx: PipelineContext, battle_id: str | None, args) -> int:
    if not args.adaptive or not battle_id:
        return args.interval
    velocity = ctx.last_velocity_by_battle.get(battle_id, 0.0)
    return adaptive_poll_interval(
        velocity,
        interval_calm=args.interval,
        interval_hot=args.interval_hot,
        hot_velocity_threshold=args.hot_velocity,
    )


def ensure_battle_anchor(ctx: PipelineContext, battle_id: str, snap: dict) -> None:
    if battle_id in ctx.battle_anchors:
        return
    game_sec = int(snap.get("time_sec", 0) or snap.get("snapshot_time_sec", 0) or 0)
    if game_sec > 0:
        ctx.battle_anchors[battle_id] = (datetime.now(), game_sec)


def realtime_data_signature() -> tuple[int, float]:
    """Cheap signature: battle-dir count + newest mtime (no full JSON glob)."""
    if not RAW_DIR.exists():
        return 0, 0.0
    dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if not dirs:
        return 0, 0.0
    latest = 0.0
    for d in dirs:
        try:
            mt = d.stat().st_mtime
            if mt > latest:
                latest = mt
            for child in d.iterdir():
                if child.suffix == ".json":
                    try:
                        cmt = child.stat().st_mtime
                        if cmt > latest:
                            latest = cmt
                            break  # one file peek is enough for "data changed" signal
                    except OSError:
                        continue
        except OSError:
            continue
    return len(dirs), latest


def write_train_status(payload: dict):
    TRAIN_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    TRAIN_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def today_match_progress(schedule: ScheduleCenter, date: str | None = None) -> tuple[int, int, list[ScheduleMatch]]:
    """Return finished_count, total_count, today's matches."""
    if date:
        today = [
            m for m in schedule.fetch_matches()
            if m.start_time and m.start_time.strftime("%Y-%m-%d") == date
        ]
    else:
        today = schedule.today_matches()
    finished = [m for m in today if m.status == 2]
    return len(finished), len(today), today


def matches_needed_for_train(schedule: ScheduleCenter, date: str | None, train_after_matches: int) -> int:
    _, total, _ = today_match_progress(schedule, date)
    if train_after_matches > 0:
        return min(train_after_matches, total) if total else train_after_matches
    return total if total else 3


def _train_job(ctx: PipelineContext, reason: str, finished_count: int, needed: int, sig: tuple[int, float]) -> None:
    try:
        artifact = train_v9_model(verbose=False)
        ctx.artifact = artifact
        ctx.last_data_sig = sig
        ctx.trained_session = True
        metrics = (
            artifact.get("metrics", {}).get("VotingMidlate")
            or artifact.get("metrics", {}).get("VotingV9")
            or {}
        )
        holdout = artifact.get("holdout_real", {}).get("all", {})
        write_train_status(
            {
                "state": "ready",
                "reason": reason,
                "finished_matches": finished_count,
                "needed_matches": needed,
                "snapshot_dirs": sig[0],
                "real_rows": artifact.get("real_rows", 0),
                "sim_rows": artifact.get("sim_rows", 0),
                "model_name": artifact.get("model_name", "Unknown"),
                "version": artifact.get("version", "V9"),
                "trained_at": artifact.get("trained_at", ""),
                "auc": metrics.get("auc"),
                "accuracy": metrics.get("accuracy"),
                "holdout_real_brier": holdout.get("brier"),
                "holdout_real_ece": holdout.get("ece"),
                "holdout_early_brier": artifact.get("holdout_real", {}).get("early", {}).get("brier"),
                "holdout_mid_brier": artifact.get("holdout_real", {}).get("mid", {}).get("brier"),
                "holdout_late_brier": artifact.get("holdout_real", {}).get("late", {}).get("brier"),
                "comeback_acc": artifact.get("holdout_comeback", {}).get("all", {}).get("accuracy"),
                "use_time_shrinkage": artifact.get("use_time_shrinkage"),
                "calibrator": "phase_isotonic" if artifact.get("calibrator_early") is not None else (
                    "isotonic" if artifact.get("calibrator") is not None else "none"
                ),
            }
        )
        write_status(
            {
                "state": "day_complete",
                "message": (
                    f"重训完成 holdout_brier={holdout.get('brier', float('nan')):.4f} "
                    f"(mixed AUC={metrics.get('auc', 0):.3f} 仅供参考)"
                ),
                "model_name": artifact.get("model_name", "Unknown"),
            }
        )
        print(
            f"[TRAIN] 完成 holdout_brier={holdout.get('brier', float('nan')):.4f} "
            f"mixed_auc={metrics.get('auc', 0):.3f} "
            f"real={artifact.get('real_rows', 0)} sim={artifact.get('sim_rows', 0)}"
        )
    except Exception as exc:
        write_train_status({"state": "error", "reason": reason, "message": str(exc)})
        write_status({"state": "train_error", "message": f"重训失败: {exc}"})
        print(f"[TRAIN] 失败: {exc}")
    finally:
        ctx.training = False


def try_retrain_once(ctx: PipelineContext, schedule: ScheduleCenter, date: str | None, reason: str) -> bool:
    """当日比赛打满后异步重训一次；主循环不阻塞采集。"""
    if not ctx.auto_train or ctx.training or ctx.trained_session:
        return False

    finished_count, total_count, _ = today_match_progress(schedule, date)
    needed = matches_needed_for_train(schedule, date, ctx.train_after_matches)
    if finished_count < needed:
        print(f"[TRAIN] 今日已完成 {finished_count}/{needed} 场，暂不训练")
        return False

    sig = realtime_data_signature()
    if sig[0] == 0:
        return False

    ctx.training = True
    write_status({"state": "training", "message": f"今日 {finished_count} 场比赛已结束，后台重训中（采集不中断）"})
    write_train_status(
        {
            "state": "training",
            "reason": reason,
            "finished_matches": finished_count,
            "needed_matches": needed,
            "snapshot_dirs": sig[0],
        }
    )
    print(f"[TRAIN] 后台启动 · 今日 {finished_count}/{needed} 场已结束 · dirs={sig[0]}")
    _TRAIN_EXECUTOR.submit(_train_job, ctx, reason, finished_count, needed, sig)
    return True


def maybe_train_after_batch(
    ctx: PipelineContext,
    schedule: ScheduleCenter,
    date: str | None,
    reason: str,
) -> None:
    finished_count, total_count, _ = today_match_progress(schedule, date)
    needed = matches_needed_for_train(schedule, date, ctx.train_after_matches)
    if finished_count >= needed:
        try_retrain_once(ctx, schedule, date, reason)


def seconds_until(start_time: datetime) -> int:
    return max(0, int((start_time - datetime.now()).total_seconds()))


def wait_until_start(
    match: ScheduleMatch,
    poll_sec: int,
    ctx: PipelineContext,
    label: str = "waiting_start",
):
    if not match.start_time:
        return
    last_print = 0.0
    while True:
        wait_sec = seconds_until(match.start_time)
        if wait_sec <= 0:
            return
        write_status(
            {
                "state": label,
                "match_id": match.match_id,
                "match": match.display_name,
                "team1": match.team1,
                "team2": match.team2,
                "start_time": match.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds_until_start": wait_sec,
                "model_name": ctx.artifact.get("model_name", "Unknown"),
                "message": f"等待开赛，距离开赛还有 {wait_sec // 60} 分 {wait_sec % 60} 秒",
            }
        )
        now_mono = time.monotonic()
        if wait_sec > 600 and now_mono - last_print >= 300:
            print(
                f"[WAIT] {match.team1} vs {match.team2} "
                f"开赛倒计时 {wait_sec // 60:02d}:{wait_sec % 60:02d}"
            )
            last_print = now_mono
        elif wait_sec <= 600 and now_mono - last_print >= 60:
            print(
                f"[WAIT] {match.team1} vs {match.team2} "
                f"开赛倒计时 {wait_sec // 60:02d}:{wait_sec % 60:02d}"
            )
            last_print = now_mono
        time.sleep(min(poll_sec, wait_sec, 60))


def run_once(
    args,
    artifact,
    history_by_battle: dict[str, list[dict]],
    match_id: str,
    match: ScheduleMatch | None = None,
    ctx: PipelineContext | None = None,
) -> tuple[str, int]:
    schedule = ScheduleCenter(args.league_id)
    match = match or schedule.get_match_by_id(match_id)
    if not match_id:
        write_status({"state": "waiting_schedule", "message": "未发现可监控赛程"})
        print("[WAIT] 未发现可监控赛程")
        return "waiting_schedule", args.interval

    if match and match.status == 2:
        print(f"[DONE] match={match_id} 已结束 · {match.display_name}")
        return "match_finished", args.interval

    battle_id = args.battle_id or schedule.fetch_live_battle_id(match_id)
    if not battle_id:
        match_info = schedule.get_match_by_id(match_id) or match
        if match_info and match_info.status == 2:
            print(f"[DONE] match={match_id} 已结束 · {match_info.display_name}")
            return "match_finished", args.interval
        if match_info and match_info.status == 1:
            write_status(
                {
                    "state": "waiting_battle",
                    "match_id": match_id,
                    "match": match.display_name if match else match_id,
                    "message": "局间休息，等待下一局 battle_id",
                }
            )
            print(f"[WAIT] match={match_id} 局间休息，等待下一局")
            return "waiting_battle", min(5, args.interval)
        write_status(
            {
                "state": "waiting_battle",
                "match_id": match_id,
                "match": match.display_name if match else match_id,
                "message": "比赛尚未生成 battle_id",
            }
        )
        print(f"[WAIT] match={match_id} 尚未生成 battle_id")
        return "waiting_battle", args.interval

    history = history_by_battle.setdefault(battle_id, [])
    min_game_sec = 0
    if history:
        min_game_sec = int(
            history[-1].get("time_sec", 0) or history[-1].get("snapshot_time_sec", 0) or 0
        )

    raw = fetch_freshest_battle(
        battle_id,
        n_probes=args.probes,
        gap_sec=args.probe_gap,
        parallel=not args.sequential_probes,
        min_game_sec=min_game_sec,
    )
    if not raw:
        write_status({"state": "api_error", "match_id": match_id, "battle_id": battle_id, "message": "battle API 不可用"})
        print(f"[ERR] battle API unavailable: {battle_id}")
        return "error", args.interval

    snap = parse_snapshot(raw.get("data", {}), battle_id)
    history = history_by_battle.setdefault(battle_id, [])

    if not is_monotonic_snapshot(snap, history):
        prev_t = int(history[-1].get("time_sec", 0) or history[-1].get("snapshot_time_sec", 0) or 0)
        cur_t = int(snap.get("time_sec", 0) or snap.get("snapshot_time_sec", 0) or 0)
        print(
            f"[SKIP] {battle_id} 快照倒退 {cur_t}s < {prev_t}s "
            f"(tol={MONOTONIC_TOLERANCE_SEC}s)，丢弃"
        )
        write_status(
            {
                "state": "stale_snapshot",
                "match_id": match_id,
                "battle_id": battle_id,
                "message": f"API 副本倒退 ({cur_t}s < {prev_t}s)，已跳过",
            }
        )
        return "stale_snapshot", next_poll_interval(ctx, battle_id, args) if ctx else args.interval

    if (
        int(snap.get("status", 0) or 0) == 2
        and history
        and int(history[-1].get("status", 0) or 0) == 2
    ):
        match_info = schedule.get_match_by_id(match_id)
        if match_info and match_info.status == 2:
            return "match_finished", args.interval
        return "battle_finished", args.interval

    if ctx is not None:
        ensure_battle_anchor(ctx, battle_id, snap)
        anchor = ctx.battle_anchors.get(battle_id)
        raw_lag = compute_lag_sec(snap, anchor[0], anchor[1]) if anchor else None
        prev_lag = ctx.last_lag_sec.get(battle_id)
        lag_sec, clock_jump = sanitize_lag_sec(raw_lag, prev_lag)
        if lag_sec is not None:
            ctx.last_lag_sec[battle_id] = lag_sec
    else:
        lag_sec, clock_jump = None, False

    snap_sec = int(snap.get("time_sec", 0) or snap.get("snapshot_time_sec", 0) or 0)
    if ctx is not None and ctx.last_snapshot_sec.get(battle_id) == snap_sec and snap_sec > 0:
        poll_sec = next_poll_interval(ctx, battle_id, args)
        write_status(
            {
                "state": "running",
                "match_id": match_id,
                "battle_id": battle_id,
                "match": match.display_name if match else match_id,
                "message": f"同秒快照已跳过 ({snap_sec}s)",
                "snapshot_time_sec": snap_sec,
                "lag_sec": lag_sec if lag_sec is not None else "",
                "clock_jump": clock_jump,
                "poll_interval_sec": poll_sec,
                "model_name": artifact.get("model_name", "Unknown"),
            }
        )
        return "duplicate_sec", poll_sec

    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob, feats, explain = predict_probability(artifact, snap, history)
    history.append(snap)
    if len(history) > 40:
        history[:] = history[-40:]

    if ctx is not None:
        ctx.last_velocity_by_battle[battle_id] = float(feats.get("gold_diff_velocity", 0.0))
        ctx.last_snapshot_sec[battle_id] = snap_sec

    row = {
        "collected_at": collected_at,
        "match_id": match_id,
        "battle_id": battle_id,
        "minute": round(float(snap.get("minute", 0)), 2),
        "snapshot_time_sec": snap_sec,
        "lag_sec": lag_sec if lag_sec is not None else "",
        "camp1_team": snap.get("camp1_team"),
        "camp2_team": snap.get("camp2_team"),
        "camp1_win_prob": round(prob, 5),
        "camp2_win_prob": round(1 - prob, 5),
        "gold_diff": snap.get("camp1_gold", 0) - snap.get("camp2_gold", 0),
        "kill_diff": snap.get("camp1_kill", 0) - snap.get("camp2_kill", 0),
        "tower_diff": snap.get("camp1_tower", 0) - snap.get("camp2_tower", 0),
        "objective_value_score": round(float(feats.get("objective_value_score", 0)), 4),
        "tempo_swing_score": round(float(feats.get("tempo_swing_score", 0)), 4),
        "top_factor": explain[0]["factor"] if explain else "",
        "status": snap.get("status"),
        "win_camp": snap.get("win_camp"),
    }
    append_prediction(battle_id, row)

    battle_state = "finished" if int(snap.get("status", 0) or 0) == 2 else "running"
    match_info = schedule.get_match_by_id(match_id)
    if match_info and match_info.status == 2:
        battle_state = "finished"

    poll_sec = next_poll_interval(ctx, battle_id, args) if ctx else args.interval
    write_status(
        {
            "state": battle_state,
            "match_id": match_id,
            "battle_id": battle_id,
            "match": match.display_name if match else match_id,
            "camp1_team": snap.get("camp1_team"),
            "camp2_team": snap.get("camp2_team"),
            "minute": row["minute"],
            "snapshot_time_sec": row["snapshot_time_sec"],
            "lag_sec": row["lag_sec"],
            "clock_jump": clock_jump,
            "low_confidence": bool(feats.get("_low_confidence", 0)),
            "confidence": feats.get("_confidence", "high"),
            "camp1_win_prob": row["camp1_win_prob"],
            "camp2_win_prob": row["camp2_win_prob"],
            "gold_diff": row["gold_diff"],
            "top_factor": row["top_factor"],
            "poll_interval_sec": poll_sec,
            "prediction_path": str(PREDICTION_DIR / f"{battle_id}.csv"),
            "model_name": artifact.get("model_name", "Unknown"),
        }
    )
    save_raw_snapshot_async(battle_id, raw)
    lag_note = f" lag={lag_sec:+.0f}s" if isinstance(lag_sec, (int, float)) else (" lag=clock_jump" if clock_jump else "")
    print(
        f"[{battle_state.upper()}] {battle_id} {row['minute']:.1f}min "
        f"{row['camp1_team']} {row['camp1_win_prob']:.1%} vs {row['camp2_team']} {row['camp2_win_prob']:.1%}"
        f"{lag_note} · next={poll_sec}s"
    )
    if match_info and match_info.status == 2:
        return "match_finished", poll_sec
    if battle_state == "finished":
        history_by_battle.pop(battle_id, None)
        if ctx is not None:
            ctx.battle_anchors.pop(battle_id, None)
            ctx.last_velocity_by_battle.pop(battle_id, None)
            ctx.last_snapshot_sec.pop(battle_id, None)
            ctx.last_lag_sec.pop(battle_id, None)
        return "battle_finished", poll_sec
    return "running", poll_sec


def advance_to_next_match(
    schedule: ScheduleCenter,
    finished_match_id: str,
    date: str | None,
    prestart_poll: int,
    ctx: PipelineContext,
) -> tuple[str | None, ScheduleMatch | None]:
    next_match = schedule.next_match_after(finished_match_id, date=date)
    if not next_match:
        write_status(
            {
                "state": "day_complete",
                "finished_match_id": finished_match_id,
                "message": "今日赛程已全部结束",
            }
        )
        print("[WAIT] 今日赛程已全部结束")
        maybe_train_after_batch(ctx, schedule, date, reason="day_complete")
        return None, None

    write_status(
        {
            "state": "waiting_next_match",
            "finished_match_id": finished_match_id,
            "next_match_id": next_match.match_id,
            "next_match": next_match.display_name,
            "team1": next_match.team1,
            "team2": next_match.team2,
            "start_time": next_match.start_time.strftime("%Y-%m-%d %H:%M:%S") if next_match.start_time else "",
            "message": f"上一场已结束，等待 {next_match.team1} vs {next_match.team2}",
        }
    )
    print(f"[NEXT] 上一场结束，下一场 {next_match.team1} vs {next_match.team2}")
    if next_match.start_time and datetime.now() < next_match.start_time:
        wait_until_start(next_match, prestart_poll, ctx, label="waiting_next_match")
    return next_match.match_id, next_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default=None, help="指定 match_id；为空则自动从赛程选择")
    parser.add_argument("--battle-id", default=None, help="指定 battle_id；为空则从 match_id 自动发现")
    parser.add_argument("--league-id", type=int, default=LEAGUE_ID)
    parser.add_argument("--date", default=None, help="限定当天赛程，格式 YYYY-MM-DD")
    parser.add_argument("--interval", type=int, default=10, help="平稳期轮询间隔秒（默认 10）")
    parser.add_argument("--interval-hot", type=int, default=4, help="团战期轮询间隔秒（默认 4）")
    parser.add_argument("--hot-velocity", type=float, default=150.0, help="判定团战期的 gold_diff_velocity 阈值")
    parser.add_argument("--adaptive", action="store_true", default=True, help="自适应轮询（默认开启）")
    parser.add_argument("--no-adaptive", action="store_true", help="禁用自适应轮询")
    parser.add_argument("--prestart-poll", type=int, default=30, help="等待开赛时的检查间隔秒")
    parser.add_argument("--probes", type=int, default=2, help="每次采集并行探测 API 副本次数")
    parser.add_argument("--probe-gap", type=float, default=0.2, help="串行探测时的间隔秒")
    parser.add_argument("--sequential-probes", action="store_true", help="使用串行探测（默认并行）")
    parser.add_argument("--once", action="store_true", help="只运行一次")
    parser.add_argument("--stop-when-finished", action="store_true", help="当前局结束后停止")
    parser.add_argument("--auto-next", action="store_true", help="一场结束后自动等待并切换到下一场")
    parser.add_argument("--no-auto-next", action="store_true", help="禁用自动切换下一场")
    parser.add_argument("--auto-train", action="store_true", help="当日比赛打满后自动重训一次（默认开启）")
    parser.add_argument("--no-auto-train", action="store_true", help="禁用自动重训")
    parser.add_argument(
        "--train-after-matches",
        type=int,
        default=0,
        help="打满几场后训练；0 表示今日赛程全部结束后再训练",
    )
    args = parser.parse_args()
    if args.no_adaptive:
        args.adaptive = False

    auto_next = not args.no_auto_next and (args.auto_next or not args.match_id)
    auto_train = not args.no_auto_train

    artifact = load_model()
    if artifact is None:
        raise RuntimeError("未找到可用模型，请先运行 train_realtime_model_v9.py 或 V8 训练脚本")

    ctx = PipelineContext(
        artifact=artifact,
        auto_train=auto_train,
        train_after_matches=args.train_after_matches,
        last_data_sig=realtime_data_signature(),
    )

    write_status(
        {
            "state": "starting",
            "model_name": artifact.get("model_name", "Unknown"),
            "auto_next": auto_next,
            "auto_train": auto_train,
            "train_after_matches": args.train_after_matches,
        }
    )
    history_by_battle: dict[str, list[dict]] = {}
    schedule = ScheduleCenter(args.league_id)
    current_match_id = args.match_id
    current_match: ScheduleMatch | None = None
    last_schedule_refresh = 0.0

    try:
        n = refresh_schedule_archive(args.league_id)
        print(f"[INIT] 已刷新赛程缓存 {n} 场")
        last_schedule_refresh = time.monotonic()
    except Exception as exc:
        print(f"[WARN] 赛程缓存刷新失败: {exc}")

    while True:
        if current_match_id:
            current_match = schedule.get_match_by_id(current_match_id)
            if not current_match:
                current_match = schedule.resolve_monitor_match(None, date=args.date)
                if current_match:
                    current_match_id = current_match.match_id
        else:
            current_match = schedule.resolve_monitor_match(None, date=args.date)
            if current_match:
                current_match_id = current_match.match_id

        if not current_match:
            write_status({"state": "waiting_schedule", "message": "暂无可监控比赛"})
            print("[WAIT] 暂无可监控比赛")
            maybe_train_after_batch(ctx, schedule, args.date, reason="waiting_schedule")
            if args.once:
                break
            time.sleep(args.prestart_poll)
            continue

        current_match_id = current_match.match_id
        if current_match.start_time and datetime.now() < current_match.start_time:
            wait_until_start(current_match, args.prestart_poll, ctx)

        cycle_args = argparse.Namespace(
            league_id=args.league_id,
            match_id=current_match_id,
            battle_id=None,
            date=args.date,
            probes=args.probes,
            probe_gap=args.probe_gap,
            sequential_probes=args.sequential_probes,
            interval=args.interval,
            interval_hot=args.interval_hot,
            hot_velocity=args.hot_velocity,
            adaptive=args.adaptive,
        )
        outcome, poll_sec = run_once(
            cycle_args, ctx.artifact, history_by_battle, current_match_id, current_match, ctx
        )

        if outcome == "match_finished":
            maybe_train_after_batch(ctx, schedule, args.date, reason="match_finished")
            if time.monotonic() - last_schedule_refresh > 1800:
                try:
                    refresh_schedule_archive(args.league_id)
                    last_schedule_refresh = time.monotonic()
                except Exception:
                    pass

        if args.once or (outcome in {"battle_finished", "match_finished"} and args.stop_when_finished):
            break

        if outcome == "match_finished" and auto_next:
            next_match_id, next_match = advance_to_next_match(
                schedule, current_match_id, args.date, args.prestart_poll, ctx
            )
            if next_match_id:
                current_match_id = next_match_id
                current_match = next_match
                args.battle_id = None
                continue
            if args.once:
                break
            time.sleep(args.prestart_poll)
            current_match_id = None
            continue

        if outcome in {"waiting_schedule", "waiting_battle", "error", "battle_finished", "stale_snapshot"}:
            time.sleep(poll_sec)
            continue

        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
