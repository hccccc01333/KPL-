"""
KPL 历史赛事分析中心

一个面向 KPL 官方场景的数据产品，而不是功能堆叠型 Demo。

运行：
    streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from kpl_official_core import (
    PREDICTION_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    detect_events,
    detect_turning_points,
    generate_battle_report,
    load_battle_jsons,
    load_kpl_knowledge,
    load_model,
    predict_probability,
    prediction_confidence,
)


ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"

COLORS = {
    "primary": "#2E86AB",
    "danger": "#E85D75",
    "gold": "#F0B429",
    "green": "#22C55E",
    "muted": "#94A3B8",
    "bg": "#0B1020",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600;700;900&display=swap');
* { font-family: 'Noto Sans SC', sans-serif; }
.main { background: #0B1020; }
[data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid #1E293B; }
h1, h2, h3 { color: #F8FAFC !important; }
.topbar {
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(46,134,171,.16), rgba(15,23,42,.88));
}
.title { font-size: 2rem; font-weight: 900; color: #F8FAFC; }
.subtitle { color: #CBD5E1; margin-top: 4px; }
.kpi {
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(15,23,42,0.82);
}
.kpi .label { color: #94A3B8; font-size: .82rem; }
.kpi .value { color: #F8FAFC; font-size: 1.7rem; font-weight: 900; margin-top: 4px; }
.kpi .note { color: #64748B; font-size: .76rem; margin-top: 4px; }
.section-note { color: #94A3B8; font-size: .9rem; margin-bottom: 10px; }
</style>
"""


@st.cache_data(ttl=300)
def load_tables():
    tables = {}
    for name in ["matches", "battles", "teams", "events"]:
        path = ANALYSIS_DIR / f"{name}.csv"
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def read_status(path: Path) -> dict:
    if not path.exists():
        return {"state": "not_started"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "status_error"}


def team_color(team: str) -> str:
    if "AG" in str(team):
        return COLORS["danger"]
    if "狼" in str(team):
        return COLORS["primary"]
    if "TTG" in str(team):
        return "#A78BFA"
    if "KSG" in str(team):
        return COLORS["gold"]
    return "#38BDF8"


def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def base_layout(fig, height=380):
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        margin=dict(t=45, b=45, l=50, r=25),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return fig


def apply_filters(matches, battles, teams, events):
    dates = sorted([x for x in matches["date"].dropna().unique()]) if not matches.empty else []
    team_options = sorted(set(teams["team"].dropna().unique())) if not teams.empty else []

    with st.sidebar:
        st.markdown("## 筛选器")
        date_range = st.multiselect("比赛日期", dates, default=dates)
        selected_teams = st.multiselect("战队", team_options, default=[])
        only_timeline = st.toggle("仅看有时间线数据的赛局", value=False)

    match_f = matches.copy()
    battle_f = battles.copy()
    team_f = teams.copy()
    event_f = events.copy()

    if date_range:
        match_f = match_f[match_f["date"].isin(date_range)]
        battle_f = battle_f[battle_f["date"].isin(date_range)]
        team_f = team_f[team_f["date"].isin(date_range)]
        event_f = event_f[event_f["date"].isin(date_range)] if not event_f.empty else event_f
    if selected_teams:
        battle_f = battle_f[battle_f["team1"].isin(selected_teams) | battle_f["team2"].isin(selected_teams)]
        team_f = team_f[team_f["team"].isin(selected_teams)]
        match_ids = battle_f["match_id"].unique()
        match_f = match_f[match_f["match_id"].isin(match_ids)]
        event_f = event_f[event_f["match_id"].isin(match_ids)] if not event_f.empty else event_f
    if only_timeline and "has_timeline" in battle_f.columns:
        battle_f = battle_f[battle_f["has_timeline"] == True]
        team_f = team_f[team_f["battle_id"].isin(battle_f["battle_id"])]
        event_f = event_f[event_f["battle_id"].isin(battle_f["battle_id"])] if not event_f.empty else event_f
    return match_f, battle_f, team_f, event_f


def render_header():
    st.markdown(
        """
        <div class="topbar">
            <div class="title">KPL 历史赛事分析中心</div>
            <div class="subtitle">以赛程索引和历史 battle 数据为基础，提供比赛覆盖、战队表现、赛局下钻与数据运维视图。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(matches, battles, teams, events):
    st.subheader("赛事总览")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("比赛场次", str(matches["match_id"].nunique() if not matches.empty else 0), "match_id 粒度")
    with c2:
        kpi("赛局数量", str(len(battles)), "battle_id 粒度")
    with c3:
        kpi("战队数量", str(teams["team"].nunique() if not teams.empty else 0), "出场队伍")
    with c4:
        kpi("平均时长", f"{battles['duration_min'].mean():.1f}m" if not battles.empty else "0", "单局平均")
    with c5:
        kpi("关键事件", str(len(events)), "由快照差分识别")

    left, right = st.columns([1.35, 1])
    with left:
        daily = battles.groupby("date").agg(games=("battle_id", "count"), avg_duration=("duration_min", "mean")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=daily["date"], y=daily["games"], name="赛局数", marker_color=COLORS["primary"]), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["avg_duration"], name="平均时长", line=dict(color=COLORS["gold"], width=3), mode="lines+markers"), secondary_y=True)
        fig.update_yaxes(title="赛局数", secondary_y=False)
        fig.update_yaxes(title="平均时长", secondary_y=True)
        st.plotly_chart(base_layout(fig, 390), use_container_width=True)
    with right:
        win_rank = teams.groupby("team").agg(games=("battle_id", "count"), wins=("win", "sum"), avg_gold=("gold_diff", "mean")).reset_index()
        win_rank["win_rate"] = win_rank["wins"] / win_rank["games"]
        win_rank = win_rank.sort_values(["wins", "win_rate"], ascending=False).head(12)
        fig = go.Figure(go.Bar(x=win_rank["wins"], y=win_rank["team"], orientation="h", marker_color=[team_color(t) for t in win_rank["team"]], text=(win_rank["win_rate"] * 100).round(0).astype(int).astype(str) + "%"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="胜局数", title="战队胜场排行")
        st.plotly_chart(base_layout(fig, 390), use_container_width=True)

    st.markdown("#### 比赛清单")
    view = matches[["date", "start_time", "league_id", "match_id", "team1", "team2", "stage", "scheduled_battles", "has_raw"]].sort_values("start_time")
    st.dataframe(view, use_container_width=True, hide_index=True)


def page_match_drilldown(matches, battles, teams, events):
    st.subheader("比赛下钻")
    if battles.empty:
        st.info("没有可分析赛局。")
        return
    options = (
        matches.assign(label=lambda x: x["date"].astype(str) + " · " + x["match_id"].astype(str) + " · " + x["team1"].astype(str) + " vs " + x["team2"].astype(str))
        .set_index("label")
        .to_dict("index")
    )
    label = st.selectbox("选择比赛", list(options.keys()))
    match_id = str(options[label]["match_id"])
    b = battles[battles["match_id"].astype(str) == match_id].sort_values("game_no")
    t = teams[teams["match_id"].astype(str) == match_id]
    e = events[events["match_id"].astype(str) == match_id] if not events.empty else events

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi("赛局数", str(len(b)), "该 BO 已有数据")
    with col2:
        kpi("平均时长", f"{b['duration_min'].mean():.1f}m", "按 battle 统计")
    with col3:
        kpi("总击杀", str(int(b["total_kills"].sum())), "双方合计")
    with col4:
        kpi("完整时间线", f"{b['has_timeline'].mean():.0%}", "用于走势复盘")

    fig = make_subplots(rows=2, cols=2, subplot_titles=("终局经济差", "终局击杀差", "推塔差", "资源差"))
    metrics = [("gold_diff", "经济差"), ("kill_diff", "击杀差"), ("tower_diff", "推塔差"), ("objective_diff", "资源差")]
    for i, (col, _) in enumerate(metrics):
        r = i // 2 + 1
        c = i % 2 + 1
        fig.add_trace(go.Bar(x=b["game_no"], y=b[col], marker_color=[team_color(x) if v >= 0 else team_color(y) for x, y, v in zip(b["team1"], b["team2"], b[col])], name=col), row=r, col=c)
    st.plotly_chart(base_layout(fig, 560), use_container_width=True)

    st.markdown("#### 赛局明细")
    st.dataframe(b[["game_no", "battle_id", "team1", "team2", "winner", "duration_min", "snapshot_count", "has_timeline", "gold_diff", "kill_diff", "tower_diff", "objective_diff"]], use_container_width=True, hide_index=True)

    # Per-battle win-prob replay + turning points + auto report
    battle_ids = [str(x) for x in b["battle_id"].tolist() if str(x)]
    if battle_ids:
        st.markdown("#### 胜率复盘")
        pick = st.selectbox("选择单局", battle_ids, key="drill_battle")
        row = b[b["battle_id"].astype(str) == pick].iloc[0]
        timeline = build_replay_timeline(pick)
        battle_dir = RAW_DIR / pick
        snaps = load_battle_jsons(battle_dir) if battle_dir.exists() else []
        ev_list = detect_events(snaps) if snaps else []
        if not e.empty:
            # merge analysis events for this game if present
            ge = e[e["battle_id"].astype(str) == pick] if "battle_id" in e.columns else e
            if not ge.empty:
                for _, er in ge.iterrows():
                    ev_list.append(
                        {
                            "minute": float(er.get("minute", 0) or 0),
                            "type": er.get("type", ""),
                            "label": er.get("label", ""),
                            "team": er.get("team", ""),
                            "weight": float(er.get("weight", 1) or 1),
                        }
                    )
        turns = detect_turning_points(timeline, ev_list) if not timeline.empty else []
        winner = str(row.get("winner", "") or "")
        t1, t2 = str(row.get("team1", "Camp1")), str(row.get("team2", "Camp2"))
        if winner == t1:
            win_camp = 1
        elif winner == t2:
            win_camp = 2
        else:
            win_camp = int(row.get("win_camp", 0) or 0)
        battle_meta = {
            "camp1": t1,
            "camp2": t2,
            "winner": winner,
            "win_camp": win_camp,
        }
        report = generate_battle_report(battle_meta, timeline, ev_list, turns)

        left, right = st.columns([1.4, 1])
        with left:
            if timeline.empty or "prob" not in timeline.columns:
                st.info("该局暂无预测时间线（缺少 raw snapshots 或模型）。")
            else:
                c1n, c2n = battle_meta["camp1"], battle_meta["camp2"]
                fig_p = go.Figure()
                fig_p.add_trace(
                    go.Scatter(
                        x=timeline["minute"],
                        y=timeline["prob"],
                        mode="lines",
                        name=f"{c1n} 胜率",
                        line=dict(color=COLORS["primary"], width=3),
                    )
                )
                fig_p.add_trace(
                    go.Scatter(
                        x=timeline["minute"],
                        y=1.0 - timeline["prob"],
                        mode="lines",
                        name=f"{c2n} 胜率",
                        line=dict(color=COLORS["danger"], width=3),
                    )
                )
                fig_p.add_hline(y=0.5, line_dash="dash", line_color=COLORS["muted"])
                if turns:
                    fig_p.add_trace(
                        go.Scatter(
                            x=[t["minute"] for t in turns],
                            y=[t["prob_after"] for t in turns],
                            mode="markers+text",
                            name="拐点",
                            text=[t["label"] for t in turns],
                            textposition="top center",
                            marker=dict(size=12, color=COLORS["gold"], symbol="triangle-up"),
                        )
                    )
                if ev_list:
                    ev_m = [float(ev.get("minute", 0) or 0) for ev in ev_list]
                    ev_y = []
                    for m in ev_m:
                        nearest = timeline.iloc[(timeline["minute"] - m).abs().argsort()[:1]]
                        ev_y.append(float(nearest["prob"].iloc[0]) if len(nearest) else 0.5)
                    fig_p.add_trace(
                        go.Scatter(
                            x=ev_m,
                            y=ev_y,
                            mode="markers",
                            name="关键事件",
                            marker=dict(size=8, color=COLORS["green"], symbol="diamond"),
                            text=[ev.get("label", "") for ev in ev_list],
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )
                fig_p.update_layout(
                    title=f"{c1n} vs {c2n} · 胜率曲线",
                    xaxis_title="比赛分钟",
                    yaxis_title="胜率",
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(base_layout(fig_p, 420), use_container_width=True)
                if turns:
                    st.dataframe(pd.DataFrame(turns)[["minute", "delta", "direction", "label", "events"]], use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### 自动战报")
            st.text(report)
            if ev_list:
                st.markdown("#### 关键事件")
                st.dataframe(
                    pd.DataFrame(ev_list)[["minute", "type", "label", "team", "weight"]].sort_values("minute"),
                    use_container_width=True,
                    hide_index=True,
                )

    if not e.empty:
        st.markdown("#### 关键事件（全场）")
        st.dataframe(e[["game_no", "minute", "type", "label", "team", "weight"]].sort_values(["game_no", "minute"]), use_container_width=True, hide_index=True)


def build_replay_timeline(battle_id: str) -> pd.DataFrame:
    """Prefer saved prediction CSV; else re-run model on raw snapshots."""
    saved = load_prediction_timeline(battle_id)
    if not saved.empty:
        df = saved.copy()
        if "camp1_win_prob" in df.columns and "prob" not in df.columns:
            df["prob"] = pd.to_numeric(df["camp1_win_prob"], errors="coerce")
        if "minute" not in df.columns and "minute_bin" in df.columns:
            df["minute"] = pd.to_numeric(df["minute_bin"], errors="coerce")
        if "gold_diff" not in df.columns and {"camp1_gold", "camp2_gold"}.issubset(df.columns):
            df["gold_diff"] = pd.to_numeric(df["camp1_gold"], errors="coerce") - pd.to_numeric(df["camp2_gold"], errors="coerce")
        return df.dropna(subset=["prob", "minute"]) if "prob" in df.columns else pd.DataFrame()

    battle_dir = RAW_DIR / battle_id
    if not battle_dir.exists():
        return pd.DataFrame()
    try:
        artifact = load_model()
    except Exception:
        return pd.DataFrame()
    snaps = load_battle_jsons(battle_dir)
    if not snaps:
        return pd.DataFrame()
    from collections import deque

    history = deque(maxlen=40)
    seen = set()
    rows = []
    for snap in snaps:
        minute_bin = max(int(snap.get("minute_bin") or snap.get("minute", 1)), 1)
        history.append(snap)
        if minute_bin in seen:
            continue
        seen.add(minute_bin)
        prob, feats, _ = predict_probability(artifact, snap, list(history))
        rows.append(
            {
                "minute": float(snap.get("minute", minute_bin)),
                "minute_bin": minute_bin,
                "prob": prob,
                "camp1_win_prob": prob,
                "camp2_win_prob": 1.0 - prob,
                "gold_diff": float(snap.get("camp1_gold", 0) or 0) - float(snap.get("camp2_gold", 0) or 0),
                "low_confidence": bool(feats.get("_low_confidence", 0)),
            }
        )
    return pd.DataFrame(rows)


def page_team_profile(battles, teams):
    st.subheader("战队画像")
    if teams.empty:
        st.info("没有战队数据。")
        return
    team = st.selectbox("选择战队", sorted(teams["team"].unique()))
    df = teams[teams["team"] == team].copy().sort_values(["date", "game_no"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("出场局数", str(len(df)), "battle 粒度")
    with c2:
        kpi("胜率", f"{df['win'].mean():.0%}", f"{int(df['win'].sum())}/{len(df)}")
    with c3:
        kpi("平均经济差", f"{df['gold_diff'].mean():+.0f}", "终局")
    with c4:
        kpi("平均击杀差", f"{df['kill_diff'].mean():+.1f}", "团战")
    with c5:
        kpi("平均推塔差", f"{df['tower_diff'].mean():+.1f}", "推进")

    df = df.reset_index(drop=True)
    df["idx"] = np.arange(1, len(df) + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["idx"], y=df["gold_diff"], name="经济差", marker_color=[team_color(team) if v >= 0 else COLORS["danger"] for v in df["gold_diff"]]), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["idx"], y=df["kill_diff"], name="击杀差", line=dict(color=COLORS["gold"], width=3), mode="lines+markers"), secondary_y=True)
    fig.update_yaxes(title="经济差", secondary_y=False)
    fig.update_yaxes(title="击杀差", secondary_y=True)
    st.plotly_chart(base_layout(fig, 420), use_container_width=True)

    radar = {
        "胜率": df["win"].mean(),
        "经济": np.clip((df["gold_diff"].mean() + 8000) / 16000, 0, 1),
        "团战": np.clip((df["kill_diff"].mean() + 10) / 20, 0, 1),
        "推进": np.clip((df["tower_diff"].mean() + 6) / 12, 0, 1),
        "资源": np.clip(df["objectives"].mean() / 6, 0, 1),
    }
    cats = list(radar.keys())
    vals = list(radar.values())
    fig_r = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", line_color=team_color(team), name=team))
    fig_r.update_layout(height=420, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,.18)")), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
    st.plotly_chart(fig_r, use_container_width=True)
    st.dataframe(df[["date", "match_id", "battle_id", "opponent", "win", "duration_min", "gold_diff", "kill_diff", "tower_diff", "objectives"]], use_container_width=True, hide_index=True)


def load_prediction_timeline(battle_id: str) -> pd.DataFrame:
    path = PREDICTION_DIR / f"{battle_id}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def list_live_battles() -> list[str]:
    if not PREDICTION_DIR.exists():
        return []
    return sorted([p.stem for p in PREDICTION_DIR.glob("*.csv")], reverse=True)


@st.fragment(run_every=timedelta(seconds=5))
def render_live_control_panel():
    monitor = read_status(REALTIME_DIR / "monitor_status.json")
    state = str(monitor.get("state", "unknown"))
    battle_id = monitor.get("battle_id") or ""

    st.caption(f"自动刷新 · 5s · 监控状态 `{state}` · 更新于 {monitor.get('updated_at', '—')}")

    if state in {"not_started", "waiting_schedule", "waiting_battle", "waiting_start", "waiting_next_match"}:
        st.info(monitor.get("message", "等待比赛开始…"))
        return

    if not battle_id:
        battle_id = list_live_battles()[0] if list_live_battles() else ""

    if not battle_id:
        st.warning("暂无实时预测数据，请先启动 `official_match_monitor.py`。")
        return

    timeline = load_prediction_timeline(battle_id)
    if timeline.empty:
        st.warning(f"未找到 `{battle_id}` 的预测时间线。")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    camp1 = monitor.get("camp1_team") or timeline.iloc[-1].get("camp1_team", "Camp1")
    camp2 = monitor.get("camp2_team") or timeline.iloc[-1].get("camp2_team", "Camp2")
    p1 = float(monitor.get("camp1_win_prob", timeline.iloc[-1].get("camp1_win_prob", 0.5)))
    p2 = float(monitor.get("camp2_win_prob", timeline.iloc[-1].get("camp2_win_prob", 0.5)))
    minute = monitor.get("minute", timeline.iloc[-1].get("minute", 0))
    lag = monitor.get("lag_sec", timeline.iloc[-1].get("lag_sec", ""))
    snap_sec = monitor.get("snapshot_time_sec", timeline.iloc[-1].get("snapshot_time_sec", ""))
    conf = prediction_confidence(minute, p1)
    if "low_confidence" in monitor:
        conf["low_confidence"] = bool(monitor.get("low_confidence"))
        conf["confidence"] = monitor.get("confidence", conf["confidence"])

    with c1:
        kpi(camp1, f"{p1:.1%}", "Camp1 胜率")
    with c2:
        kpi(camp2, f"{p2:.1%}", "Camp2 胜率")
    with c3:
        kpi("局内时间", f"{float(minute):.1f} min", f"快照 {snap_sec}s" if snap_sec != "" else "")
    with c4:
        if monitor.get("clock_jump"):
            lag_text = "时钟跳动"
        elif lag not in ("", None) and not (isinstance(lag, float) and pd.isna(lag)):
            try:
                lag_text = f"{float(lag):+.0f}s"
            except (TypeError, ValueError):
                lag_text = "—"
        else:
            lag_text = "—"
        kpi("数据滞后", lag_text, "相对开局锚点")
    with c5:
        badge = "低置信" if conf.get("low_confidence") else str(conf.get("confidence", "high")).upper()
        kpi("置信度", badge, conf.get("reason", ""))

    if conf.get("low_confidence"):
        st.warning("当前为低置信区间（开局信息不足或胜率接近 50%），解说请谨慎引用绝对值。")

    x_col = "minute" if "minute" in timeline.columns else None
    events = []
    battle_dir = RAW_DIR / str(battle_id)
    if battle_dir.exists():
        try:
            snaps = load_battle_jsons(battle_dir)
            events = detect_events(snaps)
        except Exception:
            events = []

    if x_col:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timeline[x_col],
                y=timeline["camp1_win_prob"],
                mode="lines",
                name=str(camp1),
                line=dict(color=COLORS["primary"], width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timeline[x_col],
                y=timeline["camp2_win_prob"],
                mode="lines",
                name=str(camp2),
                line=dict(color=COLORS["danger"], width=3),
            )
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["muted"])
        if events:
            ev_m = [float(e.get("minute", 0) or 0) for e in events]
            ev_y = []
            for m in ev_m:
                nearest = timeline.iloc[(timeline[x_col] - m).abs().argsort()[:1]]
                ev_y.append(float(nearest["camp1_win_prob"].iloc[0]) if len(nearest) else 0.5)
            fig.add_trace(
                go.Scatter(
                    x=ev_m,
                    y=ev_y,
                    mode="markers+text",
                    name="关键事件",
                    text=[e.get("label", "") for e in events],
                    textposition="top center",
                    marker=dict(size=10, color=COLORS["gold"], symbol="diamond"),
                )
            )
        tl_for_turns = timeline.copy()
        if "prob" not in tl_for_turns.columns and "camp1_win_prob" in tl_for_turns.columns:
            tl_for_turns["prob"] = pd.to_numeric(tl_for_turns["camp1_win_prob"], errors="coerce")
        turns = detect_turning_points(tl_for_turns, events)
        if turns:
            fig.add_trace(
                go.Scatter(
                    x=[t["minute"] for t in turns],
                    y=[t["prob_after"] for t in turns],
                    mode="markers",
                    name="拐点",
                    marker=dict(size=11, color=COLORS["green"], symbol="triangle-up"),
                    text=[t["label"] for t in turns],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"{camp1} vs {camp2} · 实时胜率",
            xaxis_title="比赛分钟",
            yaxis_title="胜率",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(base_layout(fig, 420), use_container_width=True)
        if turns:
            battle_meta = {"camp1": camp1, "camp2": camp2, "winner": "", "win_camp": 0}
            st.markdown("#### 自动战报（进行中）")
            st.text(generate_battle_report(battle_meta, tl_for_turns, events, turns))

    if "gold_diff" in timeline.columns and x_col:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=timeline[x_col],
                y=timeline["gold_diff"],
                mode="lines",
                name="经济差",
                line=dict(color=COLORS["gold"], width=2),
            )
        )
        fig2.update_layout(title="经济差走势", xaxis_title="比赛分钟", yaxis_title="经济差")
        st.plotly_chart(base_layout(fig2, 280), use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### 最近预测记录")
        show_cols = [c for c in [
            "collected_at", "minute", "snapshot_time_sec", "lag_sec",
            "camp1_win_prob", "gold_diff", "kill_diff", "top_factor",
        ] if c in timeline.columns]
        st.dataframe(timeline[show_cols].tail(12).iloc[::-1], use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### 领先原因（启发式解释 / 非 SHAP）")
        st.caption("因子来自 FeatureBuilder.explain，用于解说叙事，不是模型归因权重。")
        top = monitor.get("top_factor") or (timeline.iloc[-1].get("top_factor") if not timeline.empty else "")
        if top:
            st.info(f"当前主因：**{top}**")
        if events:
            st.markdown("#### 关键事件")
            st.dataframe(
                pd.DataFrame(events)[["minute", "type", "label", "team", "weight"]].tail(12),
                use_container_width=True,
                hide_index=True,
            )


def page_live_control():
    st.subheader("实时中控")
    st.markdown(
        '<div class="section-note">解说/导播视图：展示监控脚本最新胜率、局内时间、置信度与关键事件。'
        "页面每 5 秒自动刷新。</div>",
        unsafe_allow_html=True,
    )
    render_live_control_panel()


def page_ops(matches, battles):
    st.subheader("数据运维")
    daemon = read_status(REALTIME_DIR / "daemon_status.json")
    backfill = read_status(REALTIME_DIR / "backfill_status.json")
    monitor = read_status(REALTIME_DIR / "monitor_status.json")
    train = read_status(REALTIME_DIR / "train_status.json")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("守护进程", str(daemon.get("state")), str(daemon.get("updated_at", "")))
    with c2:
        kpi("回补状态", str(backfill.get("state")), f"{backfill.get('success', 0)}/{backfill.get('target_count', 0)}")
    with c3:
        kpi("实时监控", str(monitor.get("state")), str(monitor.get("battle_id", "")))
    with c4:
        coverage = battles["battle_id"].nunique() / max(matches["scheduled_battles"].sum(), 1)
        kpi("数据覆盖", f"{coverage:.0%}", f"{battles['battle_id'].nunique()}/{int(matches['scheduled_battles'].sum())}")
    with c5:
        holdout_brier = train.get("holdout_real_brier")
        brier_txt = f"{float(holdout_brier):.3f}" if holdout_brier not in (None, "") else "—"
        kpi("模型训练", str(train.get("state", "—")), f"holdout Brier {brier_txt}")

    st.markdown("#### 训练状态（主指标 = 真实 holdout Brier/ECE）")
    train_view = {
        "state": train.get("state"),
        "model_name": train.get("model_name"),
        "trained_at": train.get("trained_at"),
        "holdout_real_brier": train.get("holdout_real_brier"),
        "holdout_real_ece": train.get("holdout_real_ece"),
        "holdout_early_brier": train.get("holdout_early_brier"),
        "holdout_mid_brier": train.get("holdout_mid_brier"),
        "holdout_late_brier": train.get("holdout_late_brier"),
        "mixed_auc_secondary": train.get("auc"),
        "calibrator": train.get("calibrator"),
        "use_time_shrinkage": train.get("use_time_shrinkage"),
        "message": train.get("message"),
    }
    st.json(train_view)

    st.markdown("#### 状态文件")
    st.json({"daemon": daemon, "backfill": backfill, "monitor": monitor, "train": train})

    knowledge = load_kpl_knowledge()
    if knowledge:
        st.markdown("#### 业务知识配置")
        pos = [
            {"位置": v.get("name"), "职责": v.get("role"), "关注特征": "、".join(v.get("feature_focus", []))}
            for v in knowledge.get("positions", {}).values()
        ]
        obj = [
            {"资源": v.get("name"), "战略价值": v.get("strategic_value"), "权重": v.get("default_weight")}
            for v in knowledge.get("objectives", {}).values()
        ]
        st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(obj), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="KPL 历史赛事分析中心", page_icon="🏆", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()
    tables = load_tables()
    matches, battles, teams, events = tables["matches"], tables["battles"], tables["teams"], tables["events"]

    if matches.empty or battles.empty:
        st.error("分析数据集为空，请先运行 `python scripts/build_history_dataset.py`。")
        return

    matches_f, battles_f, teams_f, events_f = apply_filters(matches, battles, teams, events)

    with st.sidebar:
        st.markdown("---")
        page = st.radio("导航", ["赛事总览", "实时中控", "比赛下钻", "战队画像", "数据运维"], index=0)

    if page == "赛事总览":
        page_overview(matches_f, battles_f, teams_f, events_f)
    elif page == "实时中控":
        page_live_control()
    elif page == "比赛下钻":
        page_match_drilldown(matches_f, battles_f, teams_f, events_f)
    elif page == "战队画像":
        page_team_profile(battles_f, teams_f)
    else:
        page_ops(matches, battles)


if __name__ == "__main__":
    main()

