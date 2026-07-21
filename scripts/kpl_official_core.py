"""
KPL official intelligence platform core.

This module centralizes schedule discovery, snapshot parsing, feature building,
replay loading, event detection and model inference so the dashboard and live
scripts do not drift apart.
"""

from __future__ import annotations

import ast
import json
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "output" / "models"
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
RAW_DIR = REALTIME_DIR / "raw_snapshots"
PREDICTION_DIR = REALTIME_DIR / "predictions"
DATASETS_DIR = REALTIME_DIR / "datasets"
CLEAN_DIR = DATASETS_DIR / "clean"
QUARANTINE_DIR = DATASETS_DIR / "quarantine"
MONITOR_STATUS_FILE = REALTIME_DIR / "monitor_status.json"
SCHEDULE_ARCHIVE_PATH = REALTIME_DIR / "schedule_archive.csv"
KPL_KNOWLEDGE_PATH = PROJECT_ROOT / "docs" / "kpl_knowledge.json"

BASE_URL_BATTLE = "https://prod.comp.smoba.qq.com/leaguesite/battle/open"
BASE_URL_LEAGUES = "https://prod.comp.smoba.qq.com/leaguesite/leagues/open"
BASE_URL_LEAGUE = "https://prod.comp.smoba.qq.com/leaguesite/matches/open"
BASE_URL_MATCH = "https://prod.comp.smoba.qq.com/leaguesite/match/battles/open"
LEAGUE_ID = 20260003  # 2026 KPL夏季赛（进行中）；挑战者杯=20260002

HEADERS_API = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://pvp.qq.com/",
    "Origin": "https://pvp.qq.com",
}

POSITIONS = {2: "mid", 4: "support", 5: "jungle", 6: "top", 7: "adc"}
POSITION_CN = {
    "mid": "中路",
    "support": "游走",
    "jungle": "打野",
    "top": "对抗路",
    "adc": "发育路",
}

STATUS_MAP = {0: "未开始", 1: "进行中", 2: "已结束"}
MONOTONIC_TOLERANCE_SEC = 30


def api_get(url: str, params: dict[str, Any] | None = None, timeout: int = 12) -> dict[str, Any] | None:
    """Return a valid KPL API response, or None when the API is unavailable."""
    try:
        resp = requests.get(url, params=params, headers=HEADERS_API, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 200:
            return data
    except Exception:
        return None
    return None


def load_model() -> dict[str, Any] | None:
    candidates = [
        MODEL_DIR / "v9_official_platform.joblib",
        MODEL_DIR / "v8_realtime_enhanced.joblib",
        MODEL_DIR / "v7_realtime_stacking.joblib",
        MODEL_DIR / "v6_realtime_calibrated.joblib",
        MODEL_DIR / "v5_realtime_voting.joblib",
    ]
    path = next((p for p in candidates if p.exists()), None)
    return joblib.load(path) if path else None


def get_seconds(raw_duration: Any) -> int:
    value = raw_duration or 0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0
    return int(value / 1000 if value > 1000 else value)


def parse_start_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        ts = float(text)
        if ts > 10_000_000_000:
            ts /= 1000
        return datetime.fromtimestamp(ts)
    except (TypeError, ValueError, OSError):
        return None


def status_text(status: Any) -> str:
    try:
        return STATUS_MAP.get(int(status), str(status))
    except (TypeError, ValueError):
        return str(status or "未知")


def safe_literal(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if not text or text == "nan":
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def extract_team_name(value: Any) -> str:
    parsed = safe_literal(value)
    if isinstance(parsed, dict):
        return str(parsed.get("team_name") or parsed.get("team_abbreviation") or "")
    return str(value or "")


def extract_battle_ids(value: Any) -> list[str]:
    parsed = safe_literal(value)
    if not isinstance(parsed, list):
        return []
    battle_ids = []
    for item in parsed:
        if isinstance(item, dict) and item.get("battle_id"):
            battle_ids.append(str(item["battle_id"]))
    return battle_ids


def normalize_schedule_archive_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["league_id"] = df.get("league_id", "").astype(str)
    out["match_id"] = df.get("match_id", "").astype(str)
    out["start_time"] = df.get("start_time", "").astype(str)
    starts = pd.to_datetime(out["start_time"], errors="coerce")
    out["date"] = starts.dt.strftime("%Y-%m-%d").fillna("")
    out["year"] = starts.dt.year.fillna("").astype(str)
    out["team1"] = df.get("team1", df.get("camp1", "")).apply(extract_team_name)
    out["team2"] = df.get("team2", df.get("camp2", "")).apply(extract_team_name)
    out["status"] = df.get("status", "")
    out["status_text"] = out["status"].apply(status_text)
    out["stage"] = df.get("match_stage_desc", df.get("stage", df.get("match_stage_name", ""))).astype(str)
    out["round_name"] = df.get("match_stage_name", df.get("round_name", "")).astype(str)
    out["bo_type"] = df.get("bo", df.get("bo_type", "")).astype(str)
    battle_lists = df.get("match_battle_video_list", "").apply(extract_battle_ids)
    out["battle_ids"] = battle_lists.apply(lambda xs: "|".join(xs))
    out["battle_count"] = battle_lists.apply(len)
    out["cc_match_id"] = df.get("cc_match_id", "").astype(str)
    return out.fillna("")


@dataclass
class ScheduleMatch:
    match_id: str
    start_time: datetime | None
    team1: str
    team2: str
    status: int
    stage: str
    round_name: str
    bo_type: str
    raw: dict[str, Any]

    @property
    def status_text(self) -> str:
        return status_text(self.status)

    @property
    def display_name(self) -> str:
        date = self.start_time.strftime("%m-%d %H:%M") if self.start_time else "时间待定"
        return f"{date} · {self.team1} vs {self.team2} · {self.status_text}"


class ScheduleCenter:
    """Fetch league schedule and battle IDs from the official KPL API."""

    def __init__(self, league_id: int = LEAGUE_ID):
        self.league_id = league_id

    def fetch_matches(self) -> list[ScheduleMatch]:
        data = api_get(BASE_URL_LEAGUE, {"league_id": self.league_id})
        rows = data.get("results", []) if data else []
        return self.parse_match_rows(rows, self.league_id)

    @staticmethod
    def parse_match_rows(rows: list[dict[str, Any]], league_id: int | str) -> list[ScheduleMatch]:
        matches: list[ScheduleMatch] = []
        for row in rows:
            team1 = (
                row.get("camp1_team_name")
                or row.get("team1_name")
                or row.get("camp1", {}).get("team_name", "")
                or row.get("team_name_a", "")
            )
            team2 = (
                row.get("camp2_team_name")
                or row.get("team2_name")
                or row.get("camp2", {}).get("team_name", "")
                or row.get("team_name_b", "")
            )
            start = parse_start_time(row.get("start_time") or row.get("match_time") or row.get("time"))
            try:
                status = int(row.get("status", 0) or 0)
            except (TypeError, ValueError):
                status = 0
            match_id = str(row.get("match_id") or row.get("id") or "")
            if not match_id:
                continue
            stage = row.get("match_stage") or row.get("stage_name") or row.get("group_name") or "未知赛段"
            round_name = row.get("round_name") or row.get("match_round") or row.get("round") or "未知轮次"
            bo_type = row.get("bo") or row.get("bo_type") or row.get("match_type") or "BO?"
            matches.append(
                ScheduleMatch(
                    match_id=match_id,
                    start_time=start,
                    team1=team1 or "待定",
                    team2=team2 or "待定",
                    status=status,
                    stage=str(stage),
                    round_name=str(round_name),
                    bo_type=str(bo_type),
                    raw={**row, "league_id": str(row.get("league_id") or league_id)},
                )
            )
        return sorted(matches, key=lambda m: m.start_time or datetime.max)

    @staticmethod
    def candidate_league_ids(start_year: int = 2018, end_year: int | None = None, event_codes: range = range(1, 13)) -> list[int]:
        """Generate likely KPL league_id values: YYYY + 4-digit event code."""
        end_year = end_year or datetime.now().year
        return [int(f"{year}{code:04d}") for year in range(start_year, end_year + 1) for code in event_codes]

    def fetch_available_leagues(self) -> list[dict[str, Any]]:
        """Try the official league list endpoint. Fallback callers can use candidate ids."""
        data = api_get(BASE_URL_LEAGUES)
        rows = data.get("results", []) if data else []
        return rows if isinstance(rows, list) else []

    def fetch_matches_by_league_id(self, league_id: int | str) -> list[ScheduleMatch]:
        data = api_get(BASE_URL_LEAGUE, {"league_id": league_id})
        rows = data.get("results", []) if data else []
        return self.parse_match_rows(rows, league_id)

    def build_schedule_archive(
        self,
        league_ids: list[int | str] | None = None,
        start_year: int = 2018,
        end_year: int | None = None,
        event_codes: range = range(1, 13),
        sleep_sec: float = 0.15,
        save_path: Path = SCHEDULE_ARCHIVE_PATH,
    ) -> pd.DataFrame:
        """
        Crawl multi-season schedules and cache them.

        Notebook finding: league schedule is available from
        `/leaguesite/matches/open?league_id=...`; every row contains match_id,
        start_time, teams, status and match_battle_video_list with battle_id list.
        """
        if league_ids is None:
            api_leagues = self.fetch_available_leagues()
            league_ids = [
                x.get("league_id") or x.get("id")
                for x in api_leagues
                if str(x.get("league_id") or x.get("id") or "").isdigit()
            ]
            if not league_ids:
                league_ids = self.candidate_league_ids(start_year, end_year, event_codes)

        archive_rows: list[dict[str, Any]] = []
        seen = set()
        for lid in league_ids:
            matches = self.fetch_matches_by_league_id(lid)
            if not matches:
                continue
            for match in matches:
                key = (str(match.raw.get("league_id") or lid), str(match.match_id))
                if key in seen:
                    continue
                seen.add(key)
                battle_ids = extract_battle_ids(match.raw.get("match_battle_video_list"))
                archive_rows.append(
                    {
                        "league_id": str(match.raw.get("league_id") or lid),
                        "match_id": match.match_id,
                        "start_time": match.start_time.strftime("%Y-%m-%d %H:%M:%S") if match.start_time else "",
                        "date": match.start_time.strftime("%Y-%m-%d") if match.start_time else "",
                        "year": match.start_time.year if match.start_time else "",
                        "team1": match.team1,
                        "team2": match.team2,
                        "status": match.status,
                        "status_text": match.status_text,
                        "stage": match.stage,
                        "round_name": match.round_name,
                        "bo_type": match.bo_type,
                        "battle_ids": "|".join(battle_ids),
                        "battle_count": len(battle_ids),
                        "cc_match_id": match.raw.get("cc_match_id", ""),
                    }
                )
            time.sleep(sleep_sec)

        df = pd.DataFrame(archive_rows)
        if not df.empty:
            df = df.sort_values(["start_time", "match_id"]).reset_index(drop=True)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return df

    def load_schedule_archive(self, path: Path = SCHEDULE_ARCHIVE_PATH) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path, dtype={"league_id": str, "match_id": str}).fillna("")
        processed = PROJECT_ROOT / "data" / "processed" / "schedule.csv"
        if processed.exists():
            df = pd.read_csv(processed, dtype={"league_id": str, "match_id": str}).fillna("")
            return normalize_schedule_archive_frame(df)
        return pd.DataFrame()

    def query_schedule_archive(
        self,
        date: str | None = None,
        team_keyword: str | None = None,
        stage_keyword: str | None = None,
        path: Path = SCHEDULE_ARCHIVE_PATH,
    ) -> pd.DataFrame:
        df = self.load_schedule_archive(path)
        if df.empty:
            return df
        out = df.copy()
        if date:
            out = out[out["date"].astype(str) == date]
        if team_keyword:
            kw = team_keyword.strip()
            out = out[out["team1"].astype(str).str.contains(kw, case=False, na=False) | out["team2"].astype(str).str.contains(kw, case=False, na=False)]
        if stage_keyword:
            kw = stage_keyword.strip()
            out = out[out["stage"].astype(str).str.contains(kw, case=False, na=False) | out["round_name"].astype(str).str.contains(kw, case=False, na=False)]
        return out.reset_index(drop=True)

    def match_ids_for_date(self, date: str | None = None) -> list[str]:
        date = date or datetime.now().strftime("%Y-%m-%d")
        df = self.query_schedule_archive(date=date)
        if df.empty:
            return [m.match_id for m in self.today_matches()]
        return df["match_id"].astype(str).drop_duplicates().tolist()

    def today_matches(self, now: datetime | None = None) -> list[ScheduleMatch]:
        now = now or datetime.now()
        return [m for m in self.fetch_matches() if m.start_time and m.start_time.date() == now.date()]

    def get_match_by_id(self, match_id: str) -> ScheduleMatch | None:
        for match in self.fetch_matches():
            if match.match_id == str(match_id):
                return match
        return None

    def resolve_monitor_match(
        self,
        explicit_match_id: str | None = None,
        date: str | None = None,
    ) -> ScheduleMatch | None:
        """Pick the match that should be monitored right now."""
        if explicit_match_id:
            return self.get_match_by_id(explicit_match_id)
        today = self.today_matches() if date is None else [
            m for m in self.fetch_matches()
            if m.start_time and m.start_time.strftime("%Y-%m-%d") == date
        ]
        if not today:
            return self.active_or_next_match()
        active = [m for m in today if m.status == 1]
        if active:
            return sorted(active, key=lambda m: m.start_time or datetime.max)[0]
        now = datetime.now()
        upcoming = [
            m for m in today
            if m.status == 0 and m.start_time and m.start_time >= now - timedelta(minutes=30)
        ]
        if upcoming:
            return sorted(upcoming, key=lambda m: m.start_time or datetime.max)[0]
        pending = [m for m in today if m.status != 2]
        return sorted(pending, key=lambda m: m.start_time or datetime.max)[0] if pending else None

    def next_match_after(self, match_id: str, date: str | None = None) -> ScheduleMatch | None:
        """Return the next scheduled match on the same day after match_id."""
        today = self.today_matches() if date is None else [
            m for m in self.fetch_matches()
            if m.start_time and m.start_time.strftime("%Y-%m-%d") == date
        ]
        today = sorted(today, key=lambda m: m.start_time or datetime.max)
        found = False
        for match in today:
            if found:
                return match
            if match.match_id == str(match_id):
                found = True
        return None

    def active_or_next_match(self) -> ScheduleMatch | None:
        matches = self.fetch_matches()
        active = [m for m in matches if m.status == 1]
        if active:
            return active[0]
        now = datetime.now()
        upcoming = [m for m in matches if m.start_time and m.start_time >= now - timedelta(hours=1)]
        return upcoming[0] if upcoming else (matches[-1] if matches else None)

    def fetch_battles(self, match_id: str) -> list[dict[str, Any]]:
        data = api_get(BASE_URL_MATCH, {"match_id": match_id})
        return data.get("results", []) if data else []

    def fetch_active_battle_id(self, match_id: str) -> str | None:
        battles = self.fetch_battles(match_id)
        active = [b for b in battles if int(b.get("status", 0) or 0) == 1 and b.get("battle_id")]
        return str(active[-1]["battle_id"]) if active else None

    def fetch_current_battle_id(self, match_id: str) -> str | None:
        """Return active battle_id, or the latest battle_id for replay/backfill only."""
        active_id = self.fetch_active_battle_id(match_id)
        if active_id:
            return active_id
        battles = self.fetch_battles(match_id)
        if not battles:
            return None
        with_ids = [b for b in battles if b.get("battle_id")]
        return str(with_ids[-1]["battle_id"]) if with_ids else None

    def fetch_live_battle_id(self, match_id: str) -> str | None:
        """Only return a battle that is actively running (status=1)."""
        return self.fetch_active_battle_id(match_id)


def fetch_freshest_battle(
    battle_id: str,
    n_probes: int = 3,
    gap_sec: float = 0.6,
    parallel: bool = True,
    min_game_sec: int = 0,
) -> dict[str, Any] | None:
    """Probe replicas and return the battle response with max game_duration."""
    params = {"battle_id": battle_id}
    candidates: list[tuple[int, int, dict[str, Any]]] = []

    if parallel and n_probes > 1:
        with ThreadPoolExecutor(max_workers=n_probes) as pool:
            futures = [pool.submit(api_get, BASE_URL_BATTLE, params) for _ in range(n_probes)]
            for fut in as_completed(futures):
                data = fut.result()
                if data:
                    bd = data.get("data", {})
                    sec = get_seconds(bd.get("game_duration"))
                    gold = int((bd.get("camp1", {}) or {}).get("gold", 0) or 0) + int(
                        (bd.get("camp2", {}) or {}).get("gold", 0) or 0
                    )
                    candidates.append((sec, gold, data))
    else:
        for i in range(n_probes):
            data = api_get(BASE_URL_BATTLE, params)
            if data:
                bd = data.get("data", {})
                sec = get_seconds(bd.get("game_duration"))
                gold = int((bd.get("camp1", {}) or {}).get("gold", 0) or 0) + int(
                    (bd.get("camp2", {}) or {}).get("gold", 0) or 0
                )
                candidates.append((sec, gold, data))
            if i < n_probes - 1:
                time.sleep(gap_sec)

    if not candidates:
        return None
    if min_game_sec > 0:
        fresh = [c for c in candidates if c[0] + MONOTONIC_TOLERANCE_SEC >= min_game_sec]
        if fresh:
            candidates = fresh
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def is_monotonic_snapshot(
    snap: dict[str, Any],
    history: list[dict[str, Any]] | None,
    tolerance_sec: int = MONOTONIC_TOLERANCE_SEC,
) -> bool:
    """Reject API replicas that roll game_duration far backward (stale snapshot)."""
    if int(snap.get("status", 0) or 0) == 2:
        return True
    if not history:
        return True
    cur_t = int(snap.get("time_sec", 0) or snap.get("snapshot_time_sec", 0) or 0)
    prev_t = int(history[-1].get("time_sec", 0) or history[-1].get("snapshot_time_sec", 0) or 0)
    if prev_t <= 0:
        return True
    return cur_t + tolerance_sec >= prev_t


def refresh_schedule_archive(league_id: int | None = None) -> int:
    """Refresh cached schedule_archive.csv from live API."""
    center = ScheduleCenter(league_id or LEAGUE_ID)
    archive = center.build_schedule_archive(league_ids=[league_id or LEAGUE_ID])
    return len(archive)


def compute_lag_sec(
    snap: dict[str, Any],
    anchor_wall: datetime | None,
    anchor_game_sec: int | None,
) -> float | None:
    """Wall-clock elapsed minus in-game elapsed since battle anchor."""
    if anchor_wall is None or anchor_game_sec is None:
        return None
    game_sec = int(snap.get("time_sec", 0) or snap.get("snapshot_time_sec", 0) or 0)
    if game_sec <= 0:
        return None
    wall_elapsed = (datetime.now() - anchor_wall).total_seconds()
    game_elapsed = game_sec - anchor_game_sec
    return round(wall_elapsed - game_elapsed, 1)


def sanitize_lag_sec(
    lag_sec: float | None,
    prev_lag_sec: float | None = None,
    *,
    abs_limit: float = 120.0,
    jump_limit: float = 60.0,
) -> tuple[float | None, bool]:
    """Clamp noisy lag; return (clean_lag_or_None, clock_jump_flag)."""
    if lag_sec is None:
        return None, False
    try:
        lag = float(lag_sec)
    except (TypeError, ValueError):
        return None, True
    if abs(lag) > abs_limit:
        return None, True
    if prev_lag_sec is not None:
        try:
            if abs(lag - float(prev_lag_sec)) > jump_limit:
                return None, True
        except (TypeError, ValueError):
            pass
    return round(lag, 1), False


def phase_of_minute(minute: float | int) -> str:
    m = float(minute)
    if m <= 8:
        return "early"
    if m <= 15:
        return "mid"
    return "late"


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width probability bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / len(p)) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def evaluate_prediction_arrays(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Shared metrics for train/backtest (AUC secondary; Brier/ECE primary)."""
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"n": 0, "auc": 0.5, "accuracy": 0.0, "brier": 0.0, "ece": 0.0, "direction_acc": 0.0}
    pred = (p >= 0.5).astype(int)
    direction = ((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0)) | (np.isclose(p, 0.5))
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "direction_acc": float(np.mean(direction)),
    }


def evaluate_phase_metrics(minutes: np.ndarray, y_true: np.ndarray, prob: np.ndarray) -> dict[str, dict[str, float]]:
    minutes = np.asarray(minutes, dtype=float)
    y = np.asarray(y_true)
    p = np.asarray(prob)
    out: dict[str, dict[str, float]] = {"all": evaluate_prediction_arrays(y, p)}
    for phase in ("early", "mid", "late"):
        mask = np.array([phase_of_minute(m) == phase for m in minutes])
        out[phase] = evaluate_prediction_arrays(y[mask], p[mask])
    return out


def prediction_confidence(minute: float | int, prob: float, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Product-facing confidence badge for early / near-coin-flip predictions."""
    m = float(minute)
    p = float(prob)
    edge = abs(p - 0.5)
    low = m <= 5 or edge < 0.06
    reason = []
    if m <= 5:
        reason.append("开局信息不足")
    if edge < 0.06:
        reason.append("胜率接近 50%")
    return {
        "low_confidence": low,
        "confidence": "low" if low else ("medium" if m <= 8 or edge < 0.12 else "high"),
        "reason": "；".join(reason) if reason else "局势信号较充分",
        "use_time_shrinkage": bool((artifact or {}).get("use_time_shrinkage", False)),
    }


def adaptive_poll_interval(
    gold_diff_velocity: float,
    *,
    interval_calm: int = 10,
    interval_hot: int = 4,
    hot_velocity_threshold: float = 150.0,
) -> int:
    """Shorter polling during high-tempo moments (team fights / swings)."""
    if abs(gold_diff_velocity) >= hot_velocity_threshold:
        return max(3, interval_hot)
    return max(3, interval_calm)


PREDICTION_CSV_FIELDS = [
    "collected_at",
    "match_id",
    "battle_id",
    "minute",
    "snapshot_time_sec",
    "lag_sec",
    "camp1_team",
    "camp2_team",
    "camp1_win_prob",
    "camp2_win_prob",
    "gold_diff",
    "kill_diff",
    "tower_diff",
    "objective_value_score",
    "tempo_swing_score",
    "top_factor",
    "status",
    "win_camp",
]


def parse_snapshot(raw_data: dict[str, Any], battle_id: str = "") -> dict[str, Any]:
    game_sec = get_seconds(raw_data.get("game_duration"))
    c1 = raw_data.get("camp1", {}) or {}
    c2 = raw_data.get("camp2", {}) or {}
    snap: dict[str, Any] = {
        "battle_id": battle_id,
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_sec": int(game_sec),
        "snapshot_time_sec": int(game_sec),
        "minute": game_sec / 60,
        "minute_bin": max(int(game_sec / 60), 1),
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
        "status": raw_data.get("status", 0) or 0,
        "win_camp": raw_data.get("win_camp", 0) or 0,
    }
    players = raw_data.get("battle_player_list", []) or []
    snap["players"] = players
    for pos_id, pos_name in POSITIONS.items():
        c1_p = [p for p in players if p.get("camp") == 1 and p.get("position") == pos_id]
        c2_p = [p for p in players if p.get("camp") == 2 and p.get("position") == pos_id]
        snap[f"c1_gold_{pos_name}"] = sum(p.get("gold", 0) or 0 for p in c1_p)
        snap[f"c2_gold_{pos_name}"] = sum(p.get("gold", 0) or 0 for p in c2_p)
        snap[f"c1_hurt_{pos_name}"] = sum(p.get("hurt_to_hero_total", 0) or 0 for p in c1_p)
        snap[f"c2_hurt_{pos_name}"] = sum(p.get("hurt_to_hero_total", 0) or 0 for p in c2_p)
        snap[f"c1_behurt_{pos_name}"] = sum(p.get("be_hurt_by_hero_total", 0) or 0 for p in c1_p)
        snap[f"c2_behurt_{pos_name}"] = sum(p.get("be_hurt_by_hero_total", 0) or 0 for p in c2_p)
        snap[f"c1_kill_{pos_name}"] = sum(p.get("kill_num", 0) or 0 for p in c1_p)
        snap[f"c2_kill_{pos_name}"] = sum(p.get("kill_num", 0) or 0 for p in c2_p)
        snap[f"c1_death_{pos_name}"] = sum(p.get("death_num", 0) or 0 for p in c1_p)
        snap[f"c2_death_{pos_name}"] = sum(p.get("death_num", 0) or 0 for p in c2_p)
        snap[f"c1_assist_{pos_name}"] = sum(p.get("assist_num", 0) or 0 for p in c1_p)
        snap[f"c2_assist_{pos_name}"] = sum(p.get("assist_num", 0) or 0 for p in c2_p)
        snap[f"c1_level_{pos_name}"] = max([p.get("level", 0) or 0 for p in c1_p], default=0)
        snap[f"c2_level_{pos_name}"] = max([p.get("level", 0) or 0 for p in c2_p], default=0)
    return snap


class FeatureBuilder:
    """Build model features and official explanation metrics from snapshots."""

    def __init__(self, team_winrate: dict[str, float] | None = None):
        self.team_winrate = team_winrate or {}

    def build(self, snap: dict[str, Any], history: list[dict[str, Any]] | deque | None = None) -> dict[str, float]:
        history = list(history or [])
        minute = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
        gold_diff = snap.get("camp1_gold", 0) - snap.get("camp2_gold", 0)
        total_gold = max(snap.get("camp1_gold", 0) + snap.get("camp2_gold", 0), 1)
        kill_diff = snap.get("camp1_kill", 0) - snap.get("camp2_kill", 0)
        total_kills = max(snap.get("camp1_kill", 0) + snap.get("camp2_kill", 0), 1)
        c1_kda = (snap.get("camp1_kill", 0) + snap.get("camp1_assist", 0)) / max(snap.get("camp1_death", 0), 1)
        c2_kda = (snap.get("camp2_kill", 0) + snap.get("camp2_assist", 0)) / max(snap.get("camp2_death", 0), 1)

        prev = self._previous_snapshot(snap, history)
        if prev:
            prev_gold_diff = prev.get("camp1_gold", 0) - prev.get("camp2_gold", 0)
            dt_min = max((snap.get("time_sec", 0) - prev.get("time_sec", 0)) / 60, 0.1)
            gold_delta = gold_diff - prev_gold_diff
            gold_velocity = gold_delta / dt_min
        else:
            gold_delta = 0.0
            gold_velocity = 0.0

        c1_wr = self.team_winrate.get(str(snap.get("camp1_team", "")), 0.5)
        c2_wr = self.team_winrate.get(str(snap.get("camp2_team", "")), 0.5)

        feats: dict[str, float] = {
            "gold_diff": float(gold_diff),
            "gold_diff_per_min": gold_diff / minute,
            "gold_ratio": gold_diff / total_gold,
            "kill_diff_per_min": kill_diff / minute,
            "kill_rate": kill_diff / total_kills,
            "assist_diff_per_min": (snap.get("camp1_assist", 0) - snap.get("camp2_assist", 0)) / minute,
            "death_diff": snap.get("camp1_death", 0) - snap.get("camp2_death", 0),
            "kda_diff": c1_kda - c2_kda,
            "tower_diff": snap.get("camp1_tower", 0) - snap.get("camp2_tower", 0),
            "minute_bin": float(minute),
            "gold_diff_delta": gold_delta,
            "gold_diff_velocity": gold_velocity,
            "tyrant_diff": snap.get("camp1_tyrant", 0) - snap.get("camp2_tyrant", 0),
            "dark_tyrant_diff": snap.get("camp1_dark_tyrant", 0) - snap.get("camp2_dark_tyrant", 0),
            # lord (= big_dragon) kept for compat but excluded from objective_score (legacy overlap)
            "lord_diff": snap.get("camp1_lord", 0) - snap.get("camp2_lord", 0),
            "prophet_diff": snap.get("camp1_prophet", 0) - snap.get("camp2_prophet", 0),
            "shadow_diff": snap.get("camp1_shadow", 0) - snap.get("camp2_shadow", 0),
            "storm_diff": snap.get("camp1_storm", 0) - snap.get("camp2_storm", 0),
            "team_winrate_diff": c1_wr - c2_wr,
        }

        for _, pos_name in POSITIONS.items():
            feats[f"gold_diff_{pos_name}"] = snap.get(f"c1_gold_{pos_name}", 0) - snap.get(f"c2_gold_{pos_name}", 0)
            feats[f"hurt_diff_{pos_name}"] = snap.get(f"c1_hurt_{pos_name}", 0) - snap.get(f"c2_hurt_{pos_name}", 0)
            feats[f"behurt_diff_{pos_name}"] = snap.get(f"c1_behurt_{pos_name}", 0) - snap.get(f"c2_behurt_{pos_name}", 0)
            feats[f"kill_diff_{pos_name}"] = snap.get(f"c1_kill_{pos_name}", 0) - snap.get(f"c2_kill_{pos_name}", 0)
            feats[f"death_diff_{pos_name}"] = snap.get(f"c1_death_{pos_name}", 0) - snap.get(f"c2_death_{pos_name}", 0)
            feats[f"level_diff_{pos_name}"] = snap.get(f"c1_level_{pos_name}", 0) - snap.get(f"c2_level_{pos_name}", 0)

        c1_carry = snap.get("c1_gold_mid", 0) + snap.get("c1_gold_adc", 0)
        c2_carry = snap.get("c2_gold_mid", 0) + snap.get("c2_gold_adc", 0)
        c1_hurt = sum(snap.get(f"c1_hurt_{pos}", 0) for pos in POSITIONS.values())
        c2_hurt = sum(snap.get(f"c2_hurt_{pos}", 0) for pos in POSITIONS.values())
        c1_behurt = sum(snap.get(f"c1_behurt_{pos}", 0) for pos in POSITIONS.values())
        c2_behurt = sum(snap.get(f"c2_behurt_{pos}", 0) for pos in POSITIONS.values())
        # Prefer prophet/shadow over legacy big_dragon(lord)
        objective_score = (
            feats["dark_tyrant_diff"] * 4
            + feats["storm_diff"] * 6
            + feats["tyrant_diff"] * 2
            + feats["prophet_diff"] * 2.0
            + feats["shadow_diff"] * 2.5
            + feats["tower_diff"] * 3.5
        )
        lane_abs = [abs(feats.get(f"gold_diff_{pos}", 0)) for pos in POSITIONS.values()]

        # --- derived: momentum / structure / windows (all causal) ---
        gold_accel = self._gold_diff_accel(snap, history, gold_diff, gold_velocity, prev)
        kill_mom = self._kill_momentum_diff(snap, history, kill_diff, minute)
        lane_crush = self._lane_crush_count(feats, thresh=800.0)
        hurt_conc = self._max_share_diff(
            [snap.get(f"c1_hurt_{p}", 0) for p in POSITIONS.values()],
            [snap.get(f"c2_hurt_{p}", 0) for p in POSITIONS.values()],
        )
        behurt_conc = self._max_share_diff(
            [snap.get(f"c1_behurt_{p}", 0) for p in POSITIONS.values()],
            [snap.get(f"c2_behurt_{p}", 0) for p in POSITIONS.values()],
        )
        roll4 = self._rolling_team_diff(snap, history, lookback_min=4, key1="camp1_gold", key2="camp2_gold")
        roll10 = self._rolling_team_diff(snap, history, lookback_min=10, key1="camp1_gold", key2="camp2_gold")
        roll4_jg = self._rolling_pos_gold_diff(snap, history, lookback_min=4, pos="jungle")
        roll4_adc = self._rolling_pos_gold_diff(snap, history, lookback_min=4, pos="adc")
        win35 = self._interval_team_diffs(snap, history, start_min=3, end_min=5)
        win911 = self._interval_team_diffs(snap, history, start_min=9, end_min=11)
        obj_convert = self._obj_tower_convert(snap, history, lookback_min=3)

        feats.update(
            {
                "carry_dominance": (c1_carry - c2_carry) / max(c1_carry + c2_carry, 1),
                "objective_value_score": objective_score,
                "lane_dominance_max": max(lane_abs) / total_gold if lane_abs else 0,
                "exp_diff_per_min": gold_diff / minute,
                "map_pressure_index": (feats["tower_diff"] * 0.45 + objective_score * 0.35 + feats["gold_ratio"] * 20) / 3,
                "resource_control_rate": objective_score / max(minute, 1),
                "carry_gold_share_diff": self._carry_share_diff(snap),
                "damage_conversion_diff": (c1_hurt / max(snap.get("camp1_gold", 0), 1)) - (c2_hurt / max(snap.get("camp2_gold", 0), 1)),
                "tempo_swing_score": gold_velocity / 1000 + feats["kill_diff_per_min"] * 0.6 + feats["tower_diff"] * 0.35,
                "late_game_scaling_proxy": (c1_carry - c2_carry) / max(total_gold, 1) + feats["storm_diff"] * 0.25,
                # new real features
                "gold_diff_accel": gold_accel,
                "kill_momentum_diff": kill_mom,
                "lane_crush_count": float(lane_crush),
                "hurt_conc_diff": hurt_conc,
                "behurt_conc_diff": behurt_conc,
                "gold_diff_roll4": roll4["delta"],
                "gold_diff_roll4_per_min": roll4["per_min"],
                "gold_diff_roll10": roll10["delta"],
                "gold_diff_roll10_per_min": roll10["per_min"],
                "gold_diff_jungle_roll4": roll4_jg["delta"],
                "gold_diff_adc_roll4": roll4_adc["delta"],
                "win35_kill_diff": win35["kill_diff"],
                "win35_death_diff": win35["death_diff"],
                "win35_hurt_diff": win35["hurt_diff"],
                "win911_kill_diff": win911["kill_diff"],
                "win911_death_diff": win911["death_diff"],
                "win911_hurt_diff": win911["hurt_diff"],
                "obj_tower_convert": obj_convert,
                "kill_diff_jungle": feats["kill_diff_jungle"],
                "kill_diff_adc": feats["kill_diff_adc"],
                "death_diff_jungle": feats["death_diff_jungle"],
                "death_diff_adc": feats["death_diff_adc"],
                "behurt_diff_top": feats["behurt_diff_top"],
                "behurt_diff_support": feats["behurt_diff_support"],
            }
        )
        return feats

    def explain(self, snap: dict[str, Any], feats: dict[str, float]) -> list[dict[str, Any]]:
        """Return business-readable factors for dashboard explanation panels."""
        factors = [
            ("经济节奏", feats.get("gold_ratio", 0), "双方总经济中的领先比例"),
            ("资源控制", feats.get("objective_value_score", 0) / 12, "暴君/主宰/防御塔带来的地图价值"),
            ("核心发育", feats.get("carry_dominance", 0), "中路与发育路的经济领先"),
            ("地图压力", feats.get("map_pressure_index", 0), "推塔、资源和经济形成的推进压力"),
            ("近期动量", feats.get("tempo_swing_score", 0), "最近一段时间优势扩大的速度"),
            ("伤害转化", feats.get("damage_conversion_diff", 0), "每单位经济转化为输出的效率"),
            ("金差加速", feats.get("gold_diff_accel", 0) / 2000, "经济领先是在扩大还是收窄"),
            ("击杀动量", feats.get("kill_momentum_diff", 0), "近窗击杀节奏相对全场均值"),
        ]
        rows = []
        for name, value, desc in factors:
            signed_value = float(np.clip(value, -1.0, 1.0))
            leader = snap.get("camp1_team") if signed_value >= 0 else snap.get("camp2_team")
            rows.append(
                {
                    "factor": name,
                    "value": signed_value,
                    "abs_value": abs(signed_value),
                    "leader": leader,
                    "description": desc,
                }
            )
        return sorted(rows, key=lambda x: x["abs_value"], reverse=True)

    @staticmethod
    def _previous_snapshot(snap: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any] | None:
        cur_t = snap.get("time_sec", 0)
        for item in reversed(history):
            if item.get("time_sec", 0) < cur_t:
                return item
        return None

    @staticmethod
    def _snap_at_or_before(history: list[dict[str, Any]], target_minute: float) -> dict[str, Any] | None:
        """Latest history snap with minute_bin <= target_minute."""
        best = None
        for item in history:
            m = float(item.get("minute_bin") or item.get("minute") or 0)
            if m <= target_minute + 1e-6:
                if best is None or m >= float(best.get("minute_bin") or best.get("minute") or 0):
                    best = item
        return best

    @classmethod
    def _gold_diff_accel(
        cls,
        snap: dict[str, Any],
        history: list[dict[str, Any]],
        gold_diff: float,
        gold_velocity: float,
        prev: dict[str, Any] | None,
    ) -> float:
        if not prev:
            return 0.0
        prev2 = cls._previous_snapshot(prev, history)
        if not prev2:
            return 0.0
        prev_gd = prev.get("camp1_gold", 0) - prev.get("camp2_gold", 0)
        prev2_gd = prev2.get("camp1_gold", 0) - prev2.get("camp2_gold", 0)
        dt0 = max((prev.get("time_sec", 0) - prev2.get("time_sec", 0)) / 60, 0.1)
        v0 = (prev_gd - prev2_gd) / dt0
        return float(gold_velocity - v0)

    @classmethod
    def _kill_momentum_diff(
        cls,
        snap: dict[str, Any],
        history: list[dict[str, Any]],
        kill_diff: float,
        minute: int,
    ) -> float:
        if minute < 3:
            return 0.0
        past = cls._snap_at_or_before(history, minute - 3)
        if past is None:
            return 0.0
        past_kd = past.get("camp1_kill", 0) - past.get("camp2_kill", 0)
        dt = max(minute - float(past.get("minute_bin") or past.get("minute") or 1), 0.5)
        recent_rate = (kill_diff - past_kd) / dt
        cum_rate = kill_diff / max(minute, 1)
        return float(recent_rate - cum_rate)

    @staticmethod
    def _lane_crush_count(feats: dict[str, float], thresh: float = 800.0) -> int:
        c1 = c2 = 0
        for pos in POSITIONS.values():
            gd = float(feats.get(f"gold_diff_{pos}", 0.0))
            if gd >= thresh:
                c1 += 1
            elif gd <= -thresh:
                c2 += 1
        return c1 - c2

    @staticmethod
    def _max_share_diff(c1_vals: list[float], c2_vals: list[float]) -> float:
        s1 = float(sum(c1_vals))
        s2 = float(sum(c2_vals))
        m1 = (max(c1_vals) / s1) if s1 > 0 else 0.0
        m2 = (max(c2_vals) / s2) if s2 > 0 else 0.0
        return float(m1 - m2)

    @classmethod
    def _rolling_team_diff(
        cls,
        snap: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        lookback_min: int,
        key1: str,
        key2: str,
    ) -> dict[str, float]:
        minute = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
        cur = float(snap.get(key1, 0) or 0) - float(snap.get(key2, 0) or 0)
        if minute < 2:
            return {"delta": 0.0, "per_min": 0.0}
        past = cls._snap_at_or_before(history, max(minute - lookback_min, 1))
        if past is None:
            # fall back to earliest history or zero
            if not history:
                return {"delta": cur, "per_min": cur / max(minute, 1)}
            past = history[0]
        past_v = float(past.get(key1, 0) or 0) - float(past.get(key2, 0) or 0)
        dt = max(minute - float(past.get("minute_bin") or past.get("minute") or 1), 0.5)
        delta = cur - past_v
        return {"delta": float(delta), "per_min": float(delta / dt)}

    @classmethod
    def _rolling_pos_gold_diff(
        cls,
        snap: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        lookback_min: int,
        pos: str,
    ) -> dict[str, float]:
        return cls._rolling_team_diff(
            snap,
            history,
            lookback_min=lookback_min,
            key1=f"c1_gold_{pos}",
            key2=f"c2_gold_{pos}",
        )

    @classmethod
    def _interval_team_diffs(
        cls,
        snap: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        start_min: int,
        end_min: int,
    ) -> dict[str, float]:
        """Diffs accrued inside [start_min, end_min]; 0 if current minute < end_min."""
        minute = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
        empty = {"kill_diff": 0.0, "death_diff": 0.0, "hurt_diff": 0.0}
        if minute < end_min:
            return empty
        # end anchor: snap at end_min (or current if we're past and want closed window at end_min)
        end_snap = cls._snap_at_or_before(history + [snap], float(end_min))
        start_snap = cls._snap_at_or_before(history, float(start_min))
        if end_snap is None:
            return empty
        if start_snap is None:
            start_snap = {"camp1_kill": 0, "camp2_kill": 0, "camp1_death": 0, "camp2_death": 0}
            for pos in POSITIONS.values():
                start_snap[f"c1_hurt_{pos}"] = 0
                start_snap[f"c2_hurt_{pos}"] = 0

        def _kd(s: dict) -> float:
            return float(s.get("camp1_kill", 0) - s.get("camp2_kill", 0))

        def _dd(s: dict) -> float:
            return float(s.get("camp1_death", 0) - s.get("camp2_death", 0))

        def _hd(s: dict) -> float:
            return float(
                sum(s.get(f"c1_hurt_{p}", 0) for p in POSITIONS.values())
                - sum(s.get(f"c2_hurt_{p}", 0) for p in POSITIONS.values())
            )

        return {
            "kill_diff": _kd(end_snap) - _kd(start_snap),
            "death_diff": _dd(end_snap) - _dd(start_snap),
            "hurt_diff": _hd(end_snap) - _hd(start_snap),
        }

    @classmethod
    def _obj_tower_convert(cls, snap: dict[str, Any], history: list[dict[str, Any]], lookback_min: int = 3) -> float:
        """Tower-diff change in lookback after any objective gain (excl. legacy lord)."""
        minute = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
        if minute < lookback_min + 1:
            return 0.0
        past = cls._snap_at_or_before(history, minute - lookback_min)
        if past is None:
            return 0.0

        def _obj_counts(s: dict) -> tuple[float, float]:
            keys = ("tyrant", "dark_tyrant", "prophet", "shadow", "storm")
            c1 = sum(float(s.get(f"camp1_{k}", 0) or 0) for k in keys)
            c2 = sum(float(s.get(f"camp2_{k}", 0) or 0) for k in keys)
            return c1, c2

        c1_now, c2_now = _obj_counts(snap)
        c1_past, c2_past = _obj_counts(past)
        c1_gain = c1_now - c1_past
        c2_gain = c2_now - c2_past
        if c1_gain <= 0 and c2_gain <= 0:
            return 0.0
        tower_now = float(snap.get("camp1_tower", 0) - snap.get("camp2_tower", 0))
        tower_past = float(past.get("camp1_tower", 0) - past.get("camp2_tower", 0))
        tower_delta = tower_now - tower_past
        # Sign by who took more objectives in the window
        side = 1.0 if c1_gain > c2_gain else (-1.0 if c2_gain > c1_gain else 0.0)
        return float(tower_delta * side) if side != 0 else 0.0

    @staticmethod
    def _carry_share_diff(snap: dict[str, Any]) -> float:
        c1_gold = max(snap.get("camp1_gold", 0), 1)
        c2_gold = max(snap.get("camp2_gold", 0), 1)
        c1_carry = snap.get("c1_gold_mid", 0) + snap.get("c1_gold_adc", 0)
        c2_carry = snap.get("c2_gold_mid", 0) + snap.get("c2_gold_adc", 0)
        return c1_carry / c1_gold - c2_carry / c2_gold

def _apply_calibrator(artifact: dict[str, Any], prob_raw: float, *, which: str = "auto") -> float:
    if which == "early":
        calibrator = artifact.get("calibrator_early") or artifact.get("calibrator")
    elif which == "midlate":
        calibrator = artifact.get("calibrator_midlate") or artifact.get("calibrator")
    else:
        calibrator = artifact.get("calibrator")
    if calibrator is None:
        return float(prob_raw)
    try:
        if hasattr(calibrator, "predict"):
            out = calibrator.predict(np.array([prob_raw], dtype=float))
            return float(out[0])
        if hasattr(calibrator, "predict_proba"):
            out = calibrator.predict_proba(np.array([[prob_raw]], dtype=float))[:, 1]
            return float(out[0])
    except Exception:
        return float(prob_raw)
    return float(prob_raw)


def _apply_time_shrinkage(prob: float, minute: int) -> float:
    if minute <= 2:
        conf = 0.4
    elif minute <= 5:
        conf = 0.5 + (minute - 2) * 0.1
    elif minute <= 8:
        conf = 0.8 + (minute - 5) * 0.067
    else:
        conf = 1.0
    return 0.5 + (prob - 0.5) * conf


def _blend_weight(minute: float, blend_start: float = 6.0, blend_end: float = 10.0) -> float:
    m = float(minute)
    if m <= blend_start:
        return 0.0
    if m >= blend_end:
        return 1.0
    return (m - blend_start) / max(blend_end - blend_start, 1e-6)


def gold_prior_prob(gold_ratio: float, k: float = 10.0) -> float:
    """Weak prior from gold lead: sigmoid(k * gold_ratio)."""
    x = float(np.clip(gold_ratio, -1.0, 1.0))
    return float(1.0 / (1.0 + np.exp(-k * x)))


def apply_gold_consistency_guard(
    prob: float,
    feats: dict[str, float],
    *,
    enabled: bool = True,
    minute: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Blend model prob toward gold prior when they clash with clear economy.

    Mid/late (minute>=8): stronger pull — forbids win-prob floating against economy
    (V10 midlate isotonic failure mode).
    Late-early (minute 5–7): soft pull only on severe sign clash (|gold_diff|>=1200),
    so stomps are not reported as 0.8+ for the trailing side.
    Skip aggressive pull when major objectives support the model side (true comebacks).
    """
    meta = {"gold_guard_applied": 0.0, "p_gold": 0.5, "gold_guard_alpha": 0.0}
    if not enabled:
        return float(prob), meta
    minute = int(minute if minute is not None else feats.get("minute_bin", 1))
    gold_ratio = float(feats.get("gold_ratio", 0.0))
    p_gold = gold_prior_prob(gold_ratio)
    meta["p_gold"] = p_gold
    if minute < 5:
        return float(prob), meta

    gold_diff = float(feats.get("gold_diff", feats.get("gold_diff_per_min", 0.0) * max(minute, 1)))
    soft_early = minute < 8
    min_gold = 1200 if soft_early else 1200
    if abs(gold_ratio) < 0.02 and abs(gold_diff) < min_gold:
        return float(prob), meta
    # Soft early: only act on clear gold lead, ignore tiny ratio noise
    if soft_early and abs(gold_diff) < 1200:
        return float(prob), meta

    p = float(prob)
    # Major objectives can justify disagreeing with gold (comeback path)
    big_obj = (
        float(feats.get("lord_diff", 0.0))
        + 1.5 * float(feats.get("storm_diff", 0.0))
        + 0.5 * float(feats.get("tyrant_diff", 0.0))
        + 0.75 * float(feats.get("dark_tyrant_diff", 0.0))
    )
    if p >= 0.5 and gold_diff < 0 and big_obj >= 1.0:
        return p, meta
    if p < 0.5 and gold_diff > 0 and big_obj <= -1.0:
        return p, meta

    clash = abs(p - p_gold)
    sign_clash = ((p - 0.5) * gold_ratio < 0) or ((p - 0.5) * gold_diff < 0 and abs(gold_diff) >= 1200)
    if soft_early:
        # Require direction clash; tiny clash alone is not enough
        if not sign_clash or clash <= 0.18:
            return p, meta
        alpha = 0.18 + min(0.17, max(0.0, clash - 0.18) * 0.45)
        if abs(gold_diff) >= 2500:
            alpha += 0.06
        alpha = float(np.clip(alpha, 0.15, 0.40))
    else:
        if clash <= 0.20 and not sign_clash:
            return p, meta
        # alpha ~0.25→0.55 per plan
        alpha = 0.25 + min(0.25, (minute - 8) * 0.03) + min(0.15, max(0.0, clash - 0.20) * 0.5)
        if abs(gold_diff) >= 2500:
            alpha += 0.08
        alpha = float(np.clip(alpha, 0.25, 0.55))

    blended = (1.0 - alpha) * p + alpha * p_gold

    # Cap toward gold prior only on clear direction clashes without objective support
    if sign_clash and clash >= 0.12 and gold_diff <= -1200:
        slack = 0.06 if soft_early else (0.04 if abs(gold_diff) < 3500 else 0.0)
        blended = min(blended, p_gold + slack)
    elif sign_clash and clash >= 0.12 and gold_diff >= 1200:
        slack = 0.06 if soft_early else (0.04 if abs(gold_diff) < 3500 else 0.0)
        blended = max(blended, p_gold - slack)

    meta["gold_guard_applied"] = 1.0
    meta["gold_guard_alpha"] = alpha
    return float(np.clip(blended, 0.02, 0.98)), meta


def classify_battle_swing(snapshots: list[dict[str, Any]], win_camp: int) -> dict[str, Any]:
    """Classify comeback / swing from gold trajectory vs final winner."""
    result = {
        "comeback": False,
        "swing": False,
        "early_avg_gold_diff": 0.0,
        "mid_avg_gold_diff": 0.0,
        "win_camp": int(win_camp),
    }
    if not snapshots or win_camp not in (1, 2):
        return result
    early = [s for s in snapshots if max(int(s.get("minute", s.get("minute_bin", 0)) or 0), 1) <= 8]
    mid = [
        s
        for s in snapshots
        if 9 <= max(int(s.get("minute", s.get("minute_bin", 0)) or 0), 1) <= 12
    ]
    if len(early) >= 2:
        early_gd = float(np.mean([s.get("camp1_gold", 0) - s.get("camp2_gold", 0) for s in early]))
        result["early_avg_gold_diff"] = early_gd
        early_favors_c1 = early_gd > 0
        winner_is_c1 = win_camp == 1
        if early_favors_c1 != winner_is_c1 and abs(early_gd) >= 80:
            result["comeback"] = True
    if len(early) >= 2 and len(mid) >= 1:
        mid_gd = float(np.mean([s.get("camp1_gold", 0) - s.get("camp2_gold", 0) for s in mid]))
        result["mid_avg_gold_diff"] = mid_gd
        early_gd = result["early_avg_gold_diff"]
        if abs(early_gd) >= 80 and abs(mid_gd) >= 80 and (early_gd > 0) != (mid_gd > 0):
            result["swing"] = True
    return result


def predict_probability(artifact: dict[str, Any], snap: dict[str, Any], history: list[dict[str, Any]] | deque | None = None) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    builder = FeatureBuilder(artifact.get("team_winrate", {}))
    feats = builder.build(snap, history)
    minute = int(feats.get("minute_bin", 1))

    model_early = artifact.get("model_early")
    model_midlate = artifact.get("model_midlate") or artifact.get("model")
    is_blend = artifact.get("version") in ("V10", "V11") or model_early is not None

    if is_blend and model_early is not None and model_midlate is not None:
        early_cols = artifact.get("early_feature_columns") or artifact.get("feature_columns", [])
        mid_cols = artifact.get("feature_columns", [])
        x_e = pd.DataFrame([[feats.get(col, 0.0) for col in early_cols]], columns=early_cols)
        x_m = pd.DataFrame([[feats.get(col, 0.0) for col in mid_cols]], columns=mid_cols)
        p_e = float(model_early.predict_proba(x_e)[0, 1])
        p_m = float(model_midlate.predict_proba(x_m)[0, 1])
        p_e = _apply_calibrator(artifact, p_e, which="early")
        p_m = _apply_calibrator(artifact, p_m, which="midlate")
        clip = artifact.get("early_prob_clip") or [0.02, 0.98]
        p_e = float(np.clip(p_e, float(clip[0]), float(clip[1])))
        w = _blend_weight(
            minute,
            float(artifact.get("blend_start", 6.0)),
            float(artifact.get("blend_end", 10.0)),
        )
        prob_raw = (1 - w) * p_e + w * p_m
        prob = prob_raw
        feats["_prob_early"] = p_e
        feats["_prob_midlate"] = p_m
        feats["_blend_w"] = w
    else:
        columns = artifact.get("feature_columns", [])
        x = pd.DataFrame([[feats.get(col, 0.0) for col in columns]], columns=columns)
        prob_raw = float(artifact["model"].predict_proba(x)[0, 1])
        prob = _apply_calibrator(artifact, prob_raw)
        if artifact.get("use_time_shrinkage", False):
            prob = _apply_time_shrinkage(prob, minute)

    guard_on = bool(artifact.get("use_gold_guard", True))
    prob, guard_meta = apply_gold_consistency_guard(prob, feats, enabled=guard_on, minute=minute)
    feats["_gold_guard_applied"] = float(guard_meta["gold_guard_applied"])
    feats["_p_gold"] = float(guard_meta["p_gold"])
    feats["_gold_guard_alpha"] = float(guard_meta["gold_guard_alpha"])

    conf = prediction_confidence(minute, prob, artifact)
    feats["_prob_raw"] = float(prob_raw)
    feats["_low_confidence"] = float(conf["low_confidence"])
    feats["_confidence"] = conf["confidence"]
    return float(np.clip(prob, 0.02, 0.98)), feats, builder.explain(snap, feats)


def detect_turning_points(
    timeline: pd.DataFrame,
    events: list[dict[str, Any]] | None = None,
    *,
    delta_threshold: float = 0.08,
) -> list[dict[str, Any]]:
    """Detect win-prob swings and align nearby objective events."""
    if timeline is None or timeline.empty:
        return []
    df = timeline.copy()
    prob_col = "prob" if "prob" in df.columns else ("camp1_win_prob" if "camp1_win_prob" in df.columns else ("prob_camp1" if "prob_camp1" in df.columns else None))
    minute_col = "minute" if "minute" in df.columns else ("minute_bin" if "minute_bin" in df.columns else None)
    if not prob_col or not minute_col:
        return []
    df = df.sort_values(minute_col).reset_index(drop=True)
    df["_prob"] = pd.to_numeric(df[prob_col], errors="coerce")
    df["_minute"] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df.dropna(subset=["_prob", "_minute"])
    turns: list[dict[str, Any]] = []
    events = events or []
    for i in range(1, len(df)):
        prev_p = float(df.loc[i - 1, "_prob"])
        cur_p = float(df.loc[i, "_prob"])
        delta = cur_p - prev_p
        minute = float(df.loc[i, "_minute"])
        related = []
        for ev in events:
            ev_m = float(ev.get("minute", 0) or 0)
            if abs(ev_m - minute) <= 1.2:
                related.append(ev.get("label", ev.get("type", "")))
        if abs(delta) >= delta_threshold or related:
            turns.append(
                {
                    "minute": minute,
                    "delta": round(delta, 4),
                    "prob_before": round(prev_p, 4),
                    "prob_after": round(cur_p, 4),
                    "direction": "camp1" if delta > 0 else "camp2",
                    "label": f"{'上升' if delta > 0 else '下降'} {abs(delta):.0%}",
                    "events": related[:3],
                }
            )
    # de-dupe close minutes
    cleaned: list[dict[str, Any]] = []
    for t in turns:
        if cleaned and abs(cleaned[-1]["minute"] - t["minute"]) < 0.8:
            if abs(t["delta"]) > abs(cleaned[-1]["delta"]):
                cleaned[-1] = t
            continue
        cleaned.append(t)
    return cleaned


def generate_battle_report(
    battle: dict[str, Any],
    timeline: pd.DataFrame | None,
    events: list[dict[str, Any]] | None = None,
    turns: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a short Chinese match report for dashboard / interview demo."""
    events = events or []
    turns = turns or []
    c1 = battle.get("camp1") or battle.get("team1") or "Camp1"
    c2 = battle.get("camp2") or battle.get("team2") or "Camp2"
    winner = battle.get("winner") or ""
    win_camp = int(battle.get("win_camp", 0) or 0)
    if not winner and win_camp in (1, 2):
        winner = c1 if win_camp == 1 else c2
    lines = [f"【战报】{c1} vs {c2}"]

    if timeline is not None and not timeline.empty:
        prob_col = "prob" if "prob" in timeline.columns else ("camp1_win_prob" if "camp1_win_prob" in timeline.columns else "prob_camp1")
        minute_col = "minute" if "minute" in timeline.columns else "minute_bin"
        if prob_col in timeline.columns:
            tl = timeline.copy()
            tl[prob_col] = pd.to_numeric(tl[prob_col], errors="coerce")
            tl[minute_col] = pd.to_numeric(tl[minute_col], errors="coerce")
            tl = tl.dropna(subset=[prob_col, minute_col]).sort_values(minute_col)
            if not tl.empty:
                early = tl[tl[minute_col] <= 5]
                if not early.empty:
                    p0 = float(early.iloc[0][prob_col])
                    lean = c1 if p0 >= 0.55 else (c2 if p0 <= 0.45 else "双方均衡")
                    lines.append(f"开局：模型倾向 {lean}（约 {p0:.0%}）。")
                if "gold_diff" in tl.columns:
                    gd = pd.to_numeric(tl["gold_diff"], errors="coerce").dropna()
                    if not gd.empty:
                        lines.append(f"经济：最大领先 {gd.max():+.0f}，最大落后 {gd.min():+.0f}。")
                late = tl[tl[minute_col] >= 8]
                low_conf = 0
                if "_low_confidence" in tl.columns:
                    low_conf = int(pd.to_numeric(tl["_low_confidence"], errors="coerce").fillna(0).sum())
                elif "low_confidence" in tl.columns:
                    low_conf = int(tl["low_confidence"].astype(bool).sum())
                if low_conf:
                    lines.append(f"置信：有 {low_conf} 个切片标记为低置信，解说请谨慎引用绝对值。")

    if turns:
        top = sorted(turns, key=lambda x: abs(x.get("delta", 0)), reverse=True)[:3]
        bits = []
        for t in top:
            who = c1 if t.get("direction") == "camp1" else c2
            ev = "；".join(t.get("events") or [])
            bits.append(f"{t['minute']:.0f}分钟胜率向{who}摆动{abs(t.get('delta', 0)):.0%}" + (f"（{ev}）" if ev else ""))
        lines.append("关键拐点：" + "；".join(bits) + "。")
    elif events:
        heavy = sorted(events, key=lambda e: e.get("weight", 0), reverse=True)[:3]
        lines.append("关键事件：" + "；".join(f"{e.get('minute', '?')}分 {e.get('label', e.get('type'))}" for e in heavy) + "。")

    if winner:
        lines.append(f"终局：{winner} 获胜。")
    else:
        lines.append("终局：胜负未知。")
    return "\n".join(lines)


def detect_events(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        cur = snapshots[i]
        minute = cur.get("minute", cur.get("minute_bin", 0))
        total_prev_kill = prev.get("camp1_kill", 0) + prev.get("camp2_kill", 0)
        total_cur_kill = cur.get("camp1_kill", 0) + cur.get("camp2_kill", 0)
        if total_prev_kill == 0 and total_cur_kill > 0:
            who = cur["camp1_team"] if cur.get("camp1_kill", 0) > prev.get("camp1_kill", 0) else cur["camp2_team"]
            events.append({"minute": minute, "type": "first_blood", "label": f"一血 · {who}", "team": who, "weight": 2})

        checks = [
            ("tower", "推塔", "camp1_tower", "camp2_tower", 3),
            ("tyrant", "暴君", "camp1_tyrant", "camp2_tyrant", 2),
            ("dark_tyrant", "黑暗暴君", "camp1_dark_tyrant", "camp2_dark_tyrant", 4),
            ("lord", "主宰", "camp1_lord", "camp2_lord", 5),
            ("storm", "风暴龙王", "camp1_storm", "camp2_storm", 8),
        ]
        for ev_type, name, c1_key, c2_key, weight in checks:
            c1_delta = cur.get(c1_key, 0) - prev.get(c1_key, 0)
            c2_delta = cur.get(c2_key, 0) - prev.get(c2_key, 0)
            if c1_delta > 0:
                events.append({"minute": minute, "type": ev_type, "label": f"{name} · {cur['camp1_team']}", "team": cur["camp1_team"], "weight": weight})
            if c2_delta > 0:
                events.append({"minute": minute, "type": ev_type, "label": f"{name} · {cur['camp2_team']}", "team": cur["camp2_team"], "weight": weight})
    return events


def infer_stage_from_battle_id(battle_id: str, index: int = -1) -> str:
    """Resolve stage from schedule archive; never hard-code every battle as finals."""
    if not battle_id:
        return "本地数据"
    if SCHEDULE_ARCHIVE_PATH.exists():
        try:
            archive = pd.read_csv(SCHEDULE_ARCHIVE_PATH)
            for _, row in archive.iterrows():
                ids = [x for x in str(row.get("battle_ids") or "").split("|") if x]
                if battle_id not in ids:
                    continue
                stage = str(row.get("stage") or "").strip()
                round_name = str(row.get("round_name") or "").strip()
                if stage and stage not in ("未知赛段", "nan", "None"):
                    return stage
                if round_name and round_name not in ("未知轮次", "nan", "None"):
                    return round_name
                date = str(row.get("date") or "").strip()
                bo = row.get("bo_type", "")
                if date:
                    return f"{date} · BO{bo}" if bo != "" else date
        except (OSError, ValueError, KeyError):
            pass
    if index >= 0:
        return f"赛程局{index + 1}"
    return "常规赛"


def resolve_snapshot_root(*, from_raw: bool = False) -> Path:
    """Prefer curated clean set for train/backtest; fall back to raw ingest pool."""
    if from_raw:
        return RAW_DIR
    if CLEAN_DIR.exists() and any(CLEAN_DIR.iterdir()):
        return CLEAN_DIR
    return RAW_DIR


def load_battle_jsons(battle_dir: Path) -> list[dict[str, Any]]:
    rows = []
    seen_sec = set()
    for path in sorted(battle_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data = payload.get("data", {})
        if data.get("status", 0) not in (1, 2):
            continue
        sec = get_seconds(data.get("game_duration"))
        if sec in seen_sec:
            continue
        seen_sec.add(sec)
        rows.append(parse_snapshot(data, battle_dir.name))
    return sorted(rows, key=lambda x: x.get("time_sec", 0))


def fetch_official_winners(match_id: str = "2026052301") -> dict[str, tuple[int, str, str]]:
    result: dict[str, tuple[int, str, str]] = {}
    battle_rows = ScheduleCenter().fetch_battles(match_id)
    for battle in battle_rows:
        battle_id = str(battle.get("battle_id") or "")
        if not battle_id:
            continue
        data = api_get(BASE_URL_BATTLE, {"battle_id": battle_id}, timeout=10)
        bd = data.get("data", {}) if data else {}
        result[battle_id] = (
            int(bd.get("win_camp", 0) or 0),
            bd.get("camp1", {}).get("team_name", ""),
            bd.get("camp2", {}).get("team_name", ""),
        )
    return result


def load_replay_battles(match_id: str = "2026052301") -> list[dict[str, Any]]:
    official = fetch_official_winners(match_id)
    battles: list[dict[str, Any]] = []
    if not RAW_DIR.exists():
        return battles

    for idx, battle_dir in enumerate(sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])):
        snapshots = load_battle_jsons(battle_dir)
        if len(snapshots) < 1:
            continue
        battle_id = battle_dir.name
        win_camp, api_c1, api_c2 = official.get(battle_id, (0, "", ""))
        if win_camp == 0:
            for snap in reversed(snapshots):
                if int(snap.get("status", 0) or 0) == 2 and int(snap.get("win_camp", 0) or 0) > 0:
                    win_camp = int(snap.get("win_camp", 0))
                    api_c1 = snap.get("camp1_team", "")
                    api_c2 = snap.get("camp2_team", "")
                    break
        api_winner = api_c1 if win_camp == 1 else api_c2 if win_camp == 2 else ""
        snap_c1 = snapshots[0].get("camp1_team", "")
        snap_c2 = snapshots[0].get("camp2_team", "")
        snap_win_camp = 1 if api_winner == snap_c1 else 2 if api_winner == snap_c2 else win_camp
        battles.append(
            {
                "battle_id": battle_id,
                "match_id": match_id,
                "game_no": idx + 1,
                "stage": infer_stage_from_battle_id(battle_id, idx),
                "camp1": snap_c1,
                "camp2": snap_c2,
                "winner": api_winner or (snap_c1 if snap_win_camp == 1 else snap_c2),
                "win_camp": snap_win_camp,
                "duration_min": snapshots[-1].get("minute", 0),
                "snapshots": snapshots,
                "events": detect_events(snapshots),
                "has_timeline": len({int(s.get("minute", 0)) for s in snapshots}) > 3,
                "snapshot_count": len(snapshots),
                "raw_path": str(battle_dir),
            }
        )
    return battles


def build_prediction_timeline(artifact: dict[str, Any], battle: dict[str, Any]) -> pd.DataFrame:
    rows = []
    history: deque[dict[str, Any]] = deque(maxlen=20)
    seen_minutes = set()
    for snap in battle.get("snapshots", []):
        minute_bin = max(int(snap.get("minute", 0)), 1)
        history.append(snap)
        if minute_bin in seen_minutes:
            continue
        seen_minutes.add(minute_bin)
        prob, feats, explain = predict_probability(artifact, snap, list(history))
        rows.append(
            {
                "battle_id": battle.get("battle_id"),
                "game_no": battle.get("game_no"),
                "minute": snap.get("minute", 0),
                "prob_camp1": prob,
                "prob_camp2": 1 - prob,
                "camp1": battle.get("camp1"),
                "camp2": battle.get("camp2"),
                "gold_diff": snap.get("camp1_gold", 0) - snap.get("camp2_gold", 0),
                "kill_diff": snap.get("camp1_kill", 0) - snap.get("camp2_kill", 0),
                "tower_diff": snap.get("camp1_tower", 0) - snap.get("camp2_tower", 0),
                "objective_value_score": feats.get("objective_value_score", 0),
                "tempo_swing_score": feats.get("tempo_swing_score", 0),
                "top_factor": explain[0]["factor"] if explain else "",
                "top_factor_leader": explain[0]["leader"] if explain else "",
                "correct": (prob >= 0.5 and battle.get("win_camp") == 1) or (prob < 0.5 and battle.get("win_camp") == 2),
            }
        )
    return pd.DataFrame(rows)


def build_team_summary(battles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for battle in battles:
        final = battle["snapshots"][-1]
        for camp in (1, 2):
            team = battle.get(f"camp{camp}")
            opp = battle.get("camp2" if camp == 1 else "camp1")
            prefix = f"camp{camp}"
            other = "camp2" if camp == 1 else "camp1"
            rows.append(
                {
                    "team": team,
                    "opponent": opp,
                    "game_no": battle.get("game_no"),
                    "stage": battle.get("stage"),
                    "win": int(battle.get("winner") == team),
                    "duration_min": battle.get("duration_min", 0),
                    "gold": final.get(f"{prefix}_gold", 0),
                    "gold_diff": final.get(f"{prefix}_gold", 0) - final.get(f"{other}_gold", 0),
                    "kills": final.get(f"{prefix}_kill", 0),
                    "kill_diff": final.get(f"{prefix}_kill", 0) - final.get(f"{other}_kill", 0),
                    "towers": final.get(f"{prefix}_tower", 0),
                    "tower_diff": final.get(f"{prefix}_tower", 0) - final.get(f"{other}_tower", 0),
                    "tyrant": final.get(f"{prefix}_tyrant", 0),
                    "lord": final.get(f"{prefix}_lord", 0),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def data_health_report(battles: list[dict[str, Any]], schedule_matches: list[ScheduleMatch] | None = None) -> dict[str, Any]:
    snapshot_count = sum(b.get("snapshot_count", 0) for b in battles)
    latest_file = None
    if RAW_DIR.exists():
        files = list(RAW_DIR.glob("*/*.json"))
        latest_file = max(files, key=lambda p: p.stat().st_mtime) if files else None
    missing_timeline = [b["game_no"] for b in battles if not b.get("has_timeline")]
    teams = Counter()
    for b in battles:
        teams.update([b.get("camp1", ""), b.get("camp2", "")])
    return {
        "battle_count": len(battles),
        "snapshot_count": snapshot_count,
        "team_count": len([t for t in teams if t]),
        "schedule_count": len(schedule_matches or []),
        "latest_snapshot_file": str(latest_file) if latest_file else "无",
        "latest_snapshot_time": datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if latest_file else "无",
        "missing_timeline_games": missing_timeline,
        "raw_dir": str(RAW_DIR),
        "prediction_dir": str(PREDICTION_DIR),
    }


def load_monitor_status() -> dict[str, Any]:
    if not MONITOR_STATUS_FILE.exists():
        return {"state": "not_started", "message": "监控脚本尚未运行"}
    try:
        return json.loads(MONITOR_STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "status_error", "message": str(exc)}


def load_kpl_knowledge(path: Path = KPL_KNOWLEDGE_PATH) -> dict[str, Any]:
    """Load configurable KPL/Honor of Kings domain knowledge for UI and features."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def historical_coverage_report(archive: pd.DataFrame | None = None) -> dict[str, Any]:
    """Summarize how much schedule and battle data has been cached locally."""
    center = ScheduleCenter()
    if archive is None:
        archive = center.load_schedule_archive()
    raw_battle_ids = {p.name for p in RAW_DIR.iterdir() if p.is_dir()} if RAW_DIR.exists() else set()
    archive_battle_ids: list[str] = []
    if archive is not None and not archive.empty and "battle_ids" in archive.columns:
        for value in archive["battle_ids"].fillna("").astype(str):
            archive_battle_ids.extend([x for x in value.split("|") if x])
    archive_set = set(archive_battle_ids)
    covered = archive_set & raw_battle_ids
    return {
        "archive_matches": 0 if archive is None or archive.empty else int(len(archive)),
        "archive_battles": int(len(archive_set)),
        "raw_battles": int(len(raw_battle_ids)),
        "covered_battles": int(len(covered)),
        "coverage_rate": float(len(covered) / len(archive_set)) if archive_set else 0.0,
        "missing_battles": sorted(archive_set - raw_battle_ids)[:50],
    }


