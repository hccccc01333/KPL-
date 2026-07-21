"""
KPL 实时胜率预测系统 · Streamlit Dashboard
=============================================
功能页面：
  1. 首页概览 - 项目基础统计
  2. 赛前预测 - 选择双方战队，展示 V1/V2 预测胜率
  3. 实时胜率曲线 - 选择 battle_id，展示 V4 模型的胜率变化
  4. 模型对比 - V1 / V2 / V4 指标横向对比

运行方式：
    cd 项目根目录
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
MODEL_DIR = PROJECT_ROOT / "output" / "models"

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="KPL 实时胜率预测系统",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 数据加载（缓存）
# ============================================================

@st.cache_data
def load_battles() -> pd.DataFrame:
    path = PROCESSED_DIR / "battles.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_matches() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_bp() -> pd.DataFrame:
    path = PROCESSED_DIR / "bp.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_snapshots() -> pd.DataFrame:
    path = REALTIME_DIR / "simulated_snapshots.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(name: str):
    path = MODEL_DIR / name
    if not path.exists():
        return None
    return joblib.load(path)


# ============================================================
# 特征工程辅助函数
# ============================================================

def build_team_history(battles: pd.DataFrame) -> pd.DataFrame:
    """为每支战队计算历史统计特征（V1 模型使用）。"""
    records = []
    for camp in [1, 2]:
        prefix = f"camp{camp}_"
        subset = battles[["battle_id", f"{prefix}team_name", f"{prefix}is_win",
                          f"{prefix}kill_num", f"{prefix}death_num",
                          f"{prefix}gold", f"{prefix}push_tower_num"]].copy()
        subset.columns = ["battle_id", "team_name", "is_win",
                          "kill_num", "death_num", "gold", "push_tower_num"]
        records.append(subset)
    all_records = pd.concat(records, ignore_index=True)

    team_stats = all_records.groupby("team_name").agg(
        total_battles=("battle_id", "count"),
        wins=("is_win", "sum"),
        avg_kills=("kill_num", "mean"),
        avg_deaths=("death_num", "mean"),
        avg_gold=("gold", "mean"),
        avg_towers=("push_tower_num", "mean"),
    ).reset_index()
    team_stats["win_rate"] = team_stats["wins"] / team_stats["total_battles"]
    team_stats["kd_ratio"] = team_stats["avg_kills"] / team_stats["avg_deaths"].replace(0, 1)
    return team_stats


def build_v1_features(team1_stats: pd.Series, team2_stats: pd.Series) -> pd.DataFrame:
    """构造 V1 模型的输入特征（双方战队历史差值）。"""
    features = {
        "win_rate_diff": team1_stats["win_rate"] - team2_stats["win_rate"],
        "avg_kills_diff": team1_stats["avg_kills"] - team2_stats["avg_kills"],
        "avg_deaths_diff": team1_stats["avg_deaths"] - team2_stats["avg_deaths"],
        "avg_gold_diff": team1_stats["avg_gold"] - team2_stats["avg_gold"],
        "avg_towers_diff": team1_stats["avg_towers"] - team2_stats["avg_towers"],
        "kd_ratio_diff": team1_stats["kd_ratio"] - team2_stats["kd_ratio"],
        "camp1_win_rate": team1_stats["win_rate"],
        "camp2_win_rate": team2_stats["win_rate"],
        "camp1_total_battles": team1_stats["total_battles"],
        "camp2_total_battles": team2_stats["total_battles"],
    }
    return pd.DataFrame([features])


def build_v2_features(
    team1_stats: pd.Series,
    team2_stats: pd.Series,
    bp_data: pd.DataFrame,
    team1_name: str,
    team2_name: str,
    battles: pd.DataFrame,
) -> pd.DataFrame:
    """构造 V2 模型的输入特征（V1 特征 + BP 英雄特征）。"""
    v1_feat = build_v1_features(team1_stats, team2_stats)

    team1_battles = battles[battles["camp1_team_name"] == team1_name]["battle_id"].tolist() + \
                    battles[battles["camp2_team_name"] == team1_name]["battle_id"].tolist()
    team2_battles = battles[battles["camp1_team_name"] == team2_name]["battle_id"].tolist() + \
                    battles[battles["camp2_team_name"] == team2_name]["battle_id"].tolist()

    team1_picks = bp_data[(bp_data["battle_id"].isin(team1_battles)) & (bp_data["is_ban_or_pick"] == 1)]
    team2_picks = bp_data[(bp_data["battle_id"].isin(team2_battles)) & (bp_data["is_ban_or_pick"] == 1)]

    v1_feat["camp1_unique_heroes"] = team1_picks["hero_id"].nunique() if not team1_picks.empty else 0
    v1_feat["camp2_unique_heroes"] = team2_picks["hero_id"].nunique() if not team2_picks.empty else 0
    v1_feat["hero_pool_diff"] = v1_feat["camp1_unique_heroes"] - v1_feat["camp2_unique_heroes"]
    v1_feat["camp1_avg_picks"] = len(team1_picks) / max(len(team1_battles), 1)
    v1_feat["camp2_avg_picks"] = len(team2_picks) / max(len(team2_battles), 1)
    return v1_feat


# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.title("🎮 KPL 胜率预测")
page = st.sidebar.radio(
    "选择页面",
    ["📊 首页概览", "🔮 赛前预测", "📈 实时胜率曲线", "⚖️ 模型对比"],
)

# ============================================================
# 加载数据
# ============================================================
battles = load_battles()
matches = load_matches()
bp_data = load_bp()
snapshots = load_snapshots()

if battles.empty:
    st.error("❌ 未找到 data/processed/battles.csv，请先运行前序 notebook 生成数据。")
    st.stop()

# ============================================================
# 页面 1：首页概览
# ============================================================
if page == "📊 首页概览":
    st.title("📊 KPL 实时胜率预测系统")
    st.markdown("---")
    st.markdown("""
    **项目简介**：基于 KPL（王者荣耀职业联赛）历史数据，构建从赛前到实时的多阶段胜率预测模型。

    - **V1 基线模型**：使用战队历史战绩特征进行赛前预测
    - **V2 BP 模型**：在 V1 基础上加入 Ban/Pick 英雄池特征
    - **V4 实时模型**：利用比赛进行中的实时快照数据动态预测胜率
    """)

    st.markdown("### 数据概览")
    col1, col2, col3, col4 = st.columns(4)

    total_battles = len(battles)
    all_teams = set(battles["camp1_team_name"].tolist() + battles["camp2_team_name"].tolist())
    total_teams = len(all_teams)

    if not matches.empty and "start_time" in matches.columns:
        date_min = matches["start_time"].min()[:10]
        date_max = matches["start_time"].max()[:10]
        date_range = f"{date_min} ~ {date_max}"
    else:
        date_range = "未知"

    col1.metric("总比赛场次", f"{total_battles} 场")
    col2.metric("参赛战队数", f"{total_teams} 支")
    col3.metric("时间范围", date_range)
    col4.metric("实时快照数", f"{len(snapshots)} 条")

    st.markdown("### 模型文件状态")
    model_files = {
        "V1 基线模型": "v1_baseline.joblib",
        "V2 BP模型": "v2_bp_features.joblib",
        "V4 实时模型": "v4_realtime_model.joblib",
    }
    status_cols = st.columns(3)
    for i, (label, fname) in enumerate(model_files.items()):
        exists = (MODEL_DIR / fname).exists()
        status_cols[i].metric(label, "✅ 已就绪" if exists else "⏳ 待训练")

    st.markdown("### 战队一览")
    team_stats = build_team_history(battles)
    team_stats_display = team_stats[["team_name", "total_battles", "wins", "win_rate"]].copy()
    team_stats_display["win_rate"] = (team_stats_display["win_rate"] * 100).round(1).astype(str) + "%"
    team_stats_display.columns = ["战队", "总场次", "胜场", "胜率"]
    st.dataframe(team_stats_display.sort_values("胜场", ascending=False), use_container_width=True, hide_index=True)

# ============================================================
# 页面 2：赛前预测
# ============================================================
elif page == "🔮 赛前预测":
    st.title("🔮 赛前胜率预测")
    st.markdown("选择两支战队，使用 V1（历史战绩）和 V2（BP 特征）模型进行赛前预测。")
    st.markdown("---")

    team_stats = build_team_history(battles)
    team_list = sorted(team_stats["team_name"].tolist())

    col_left, col_right = st.columns(2)
    with col_left:
        team1 = st.selectbox("🔵 Camp1 战队（蓝方）", team_list, index=0)
    with col_right:
        team2 = st.selectbox("🔴 Camp2 战队（红方）", team_list, index=min(1, len(team_list) - 1))

    if team1 == team2:
        st.warning("请选择两支不同的战队。")
        st.stop()

    team1_stats = team_stats[team_stats["team_name"] == team1].iloc[0]
    team2_stats = team_stats[team_stats["team_name"] == team2].iloc[0]

    st.markdown("### 战队历史对比")
    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        st.markdown(f"**🔵 {team1}**")
        st.metric("胜率", f"{team1_stats['win_rate']:.1%}")
        st.metric("场均击杀", f"{team1_stats['avg_kills']:.1f}")
        st.metric("场均推塔", f"{team1_stats['avg_towers']:.1f}")
    with compare_col2:
        st.markdown(f"**🔴 {team2}**")
        st.metric("胜率", f"{team2_stats['win_rate']:.1%}")
        st.metric("场均击杀", f"{team2_stats['avg_kills']:.1f}")
        st.metric("场均推塔", f"{team2_stats['avg_towers']:.1f}")

    st.markdown("---")
    st.markdown("### 模型预测结果")

    v1_model = load_model("v1_baseline.joblib")
    v2_model = load_model("v2_bp_features.joblib")

    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        st.markdown("#### V1 基线模型")
        if v1_model is not None:
            v1_feat = build_v1_features(team1_stats, team2_stats)
            try:
                v1_prob = v1_model.predict_proba(v1_feat)[0]
                camp1_prob_v1 = v1_prob[1] if len(v1_prob) > 1 else v1_prob[0]
                st.metric(f"{team1} 胜率", f"{camp1_prob_v1:.1%}")
                st.metric(f"{team2} 胜率", f"{1 - camp1_prob_v1:.1%}")
                fig_v1 = go.Figure(go.Bar(
                    x=[team1, team2],
                    y=[camp1_prob_v1, 1 - camp1_prob_v1],
                    marker_color=["#1f77b4", "#d62728"],
                    text=[f"{camp1_prob_v1:.1%}", f"{1 - camp1_prob_v1:.1%}"],
                    textposition="outside",
                ))
                fig_v1.update_layout(
                    title="V1 预测胜率",
                    yaxis_range=[0, 1],
                    yaxis_title="胜率",
                    height=300,
                )
                st.plotly_chart(fig_v1, use_container_width=True)
            except Exception as e:
                st.error(f"V1 模型预测失败：{e}")
                st.info("可能原因：模型期望的特征列与当前构造的特征不匹配。请检查训练时的特征列表。")
        else:
            st.info("⏳ V1 模型文件未找到，请先运行 notebook 05 训练模型。")

    with pred_col2:
        st.markdown("#### V2 BP 模型")
        if v2_model is not None:
            v2_feat = build_v2_features(team1_stats, team2_stats, bp_data, team1, team2, battles)
            try:
                v2_prob = v2_model.predict_proba(v2_feat)[0]
                camp1_prob_v2 = v2_prob[1] if len(v2_prob) > 1 else v2_prob[0]
                st.metric(f"{team1} 胜率", f"{camp1_prob_v2:.1%}")
                st.metric(f"{team2} 胜率", f"{1 - camp1_prob_v2:.1%}")
                fig_v2 = go.Figure(go.Bar(
                    x=[team1, team2],
                    y=[camp1_prob_v2, 1 - camp1_prob_v2],
                    marker_color=["#1f77b4", "#d62728"],
                    text=[f"{camp1_prob_v2:.1%}", f"{1 - camp1_prob_v2:.1%}"],
                    textposition="outside",
                ))
                fig_v2.update_layout(
                    title="V2 预测胜率",
                    yaxis_range=[0, 1],
                    yaxis_title="胜率",
                    height=300,
                )
                st.plotly_chart(fig_v2, use_container_width=True)
            except Exception as e:
                st.error(f"V2 模型预测失败：{e}")
                st.info("可能原因：模型期望的特征列与当前构造的特征不匹配。请检查训练时的特征列表。")
        else:
            st.info("⏳ V2 模型文件未找到，请先运行 notebook 06 训练模型。")

# ============================================================
# 页面 3：实时胜率曲线
# ============================================================
elif page == "📈 实时胜率曲线":
    st.title("📈 实时胜率曲线")
    st.markdown("从模拟快照中选择一场比赛，查看 V4 模型在不同时间点的胜率预测变化。")
    st.markdown("---")

    if snapshots.empty:
        st.warning("⏳ 未找到 data/realtime/simulated_snapshots.csv，请先运行 notebook 07/08 生成快照数据。")
        st.stop()

    v4_model = load_model("v4_realtime_model.joblib")

    battle_ids = snapshots["battle_id"].unique().tolist()
    selected_battle = st.sidebar.selectbox("选择比赛 (battle_id)", battle_ids)

    battle_data = snapshots[snapshots["battle_id"] == selected_battle].copy()

    if "camp1_team_name" in battle_data.columns:
        team_info = battle_data.iloc[0]
        st.markdown(f"**对阵**：🔵 {team_info.get('camp1_team_name', 'Camp1')} vs 🔴 {team_info.get('camp2_team_name', 'Camp2')}")

    if v4_model is not None:
        time_col = None
        for candidate in ["minute", "time_minute", "game_time", "snapshot_minute"]:
            if candidate in battle_data.columns:
                time_col = candidate
                break

        if time_col is None and "collected_at" in battle_data.columns:
            battle_data = battle_data.sort_values("collected_at").reset_index(drop=True)
            battle_data["time_point"] = range(1, len(battle_data) + 1)
            time_col = "time_point"

        if time_col is None:
            battle_data["time_point"] = range(1, len(battle_data) + 1)
            time_col = "time_point"

        try:
            feature_cols = [c for c in battle_data.columns
                           if c not in ["battle_id", "camp1_team_name", "camp2_team_name",
                                        "win_camp", "label", "collected_at", time_col]]
            X_battle = battle_data[feature_cols].select_dtypes(include=[np.number])
            probs = v4_model.predict_proba(X_battle)
            camp1_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            curve_df = pd.DataFrame({
                "时间点": battle_data[time_col].values,
                "Camp1 胜率": camp1_probs,
                "Camp2 胜率": 1 - camp1_probs,
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_df["时间点"], y=curve_df["Camp1 胜率"],
                mode="lines+markers", name="🔵 Camp1 胜率",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=curve_df["时间点"], y=curve_df["Camp2 胜率"],
                mode="lines+markers", name="🔴 Camp2 胜率",
                line=dict(color="#d62728", width=3),
                marker=dict(size=8),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text="50% 均势线")
            fig.update_layout(
                title=f"比赛 {selected_battle} · 实时胜率变化",
                xaxis_title="时间点（分钟）",
                yaxis_title="预测胜率",
                yaxis_range=[0, 1],
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            if "win_camp" in battle_data.columns:
                actual_winner = battle_data["win_camp"].iloc[0]
                st.success(f"🏆 实际获胜方：Camp{int(actual_winner)}")
        except Exception as e:
            st.error(f"V4 模型预测失败：{e}")
            st.info("可能原因：快照数据的特征列与模型训练时不一致。")
    else:
        st.info("⏳ V4 模型文件未找到，请先运行 notebook 08 训练模型。")
        st.markdown("**快照数据预览**：")
        st.dataframe(battle_data.head(10), use_container_width=True)

# ============================================================
# 页面 4：模型对比
# ============================================================
elif page == "⚖️ 模型对比":
    st.title("⚖️ 模型性能对比")
    st.markdown("横向对比三个版本模型的核心指标，了解每次迭代带来的提升。")
    st.markdown("---")

    metrics_data = {
        "模型版本": ["V1 基线（历史战绩）", "V2 BP增强", "V4 实时快照"],
        "特征类型": ["战队历史胜率、KD、推塔等", "V1 + 英雄池宽度、BP 选择", "实时经济差、击杀差、推塔差等"],
        "适用场景": ["赛前（赛程公布后）", "赛前（BP 结束后）", "比赛进行中（每分钟更新）"],
        "Accuracy": ["—", "—", "—"],
        "AUC": ["—", "—", "—"],
    }

    metrics_path = PROJECT_ROOT / "output" / "metrics"
    for i, version in enumerate(["v1", "v2", "v4"]):
        report_path = metrics_path / f"{version}_metrics.csv"
        if report_path.exists():
            try:
                report = pd.read_csv(report_path)
                if "accuracy" in report.columns:
                    metrics_data["Accuracy"][i] = f"{report['accuracy'].iloc[0]:.3f}"
                if "auc" in report.columns:
                    metrics_data["AUC"][i] = f"{report['auc'].iloc[0]:.3f}"
            except Exception:
                pass

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 模型迭代思路")
    st.markdown("""
    | 迭代 | 核心改进 | 业务价值 |
    |------|---------|---------|
    | V1 → V2 | 加入 BP 英雄信息 | 能在 Ban/Pick 结束后给出更准确的预测 |
    | V2 → V4 | 引入实时对局数据 | 比赛过程中动态更新胜率，观赛体验升级 |
    """)

    st.markdown("### 适用场景对比")
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=["赛程公布", "BP结束", "比赛开始", "比赛中期", "比赛结束"],
        y=[1, 1, 1, 1, 1],
        mode="markers+text",
        marker=dict(size=20, color=["#2ecc71", "#2ecc71", "#3498db", "#3498db", "#95a5a6"]),
        text=["V1可用", "V2可用", "V4启动", "V4持续预测", "出结果"],
        textposition="top center",
    ))
    fig_timeline.update_layout(
        title="模型在比赛生命周期中的适用时间",
        showlegend=False,
        yaxis_visible=False,
        height=200,
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ============================================================
# 页脚
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**KPL 实时胜率预测系统**")
st.sidebar.markdown("版本：V5 Dashboard")
st.sidebar.caption("数据分析项目 · 2026")
