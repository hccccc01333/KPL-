"""
Pre-match and post-BP feature engineering.

Extracted from notebooks 05 and 06 after stabilization.

Rules:
- Use only features available before the match starts or after BP is complete.
- All historical win-rate features must use shift(1).
- Keep feature column order stable and save it with the model artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


V1_FEATURES = ['camp1_recent_wr', 'camp2_recent_wr', 'wr_diff', 'h2h_camp1_wr']

V2_FEATURES = V1_FEATURES + [
    'camp1_avg_hero_wr', 'camp2_avg_hero_wr', 'hero_wr_diff',
    'camp1_avg_hero_games', 'camp2_avg_hero_games',
    'camp1_avg_team_hero_wr', 'camp2_avg_team_hero_wr', 'team_hero_wr_diff',
    'camp1_new_hero_count', 'camp2_new_hero_count', 'new_hero_count_diff',
]


def build_team_history_features(battles: pd.DataFrame) -> pd.DataFrame:
    """Build V1 team-form and head-to-head features.

    Returns battles with added columns:
      camp1_recent_wr, camp2_recent_wr, wr_diff, h2h_camp1_wr
    """
    df = battles.copy()

    # long_df: one row per team per battle
    camp1 = df[['battle_id', 'battles_time_id', 'camp1_team_id']].copy()
    camp1.columns = ['battle_id', 'battles_time_id', 'team_id']
    camp1['win'] = (df['win_camp'] == 1).astype(int)

    camp2 = df[['battle_id', 'battles_time_id', 'camp2_team_id']].copy()
    camp2.columns = ['battle_id', 'battles_time_id', 'team_id']
    camp2['win'] = (df['win_camp'] == 2).astype(int)

    long_df = pd.concat([camp1, camp2], ignore_index=True)
    long_df = long_df.sort_values(['team_id', 'battles_time_id']).reset_index(drop=True)

    # Recent 5-game win rate (shift to prevent leakage)
    long_df['recent_wr'] = long_df.groupby('team_id')['win'].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Merge back
    df = df.merge(
        long_df[['battle_id', 'team_id', 'recent_wr']],
        left_on=['battle_id', 'camp1_team_id'],
        right_on=['battle_id', 'team_id'], how='left'
    ).rename(columns={'recent_wr': 'camp1_recent_wr'}).drop(columns='team_id')

    df = df.merge(
        long_df[['battle_id', 'team_id', 'recent_wr']],
        left_on=['battle_id', 'camp2_team_id'],
        right_on=['battle_id', 'team_id'], how='left'
    ).rename(columns={'recent_wr': 'camp2_recent_wr'}).drop(columns='team_id')

    df['wr_diff'] = df['camp1_recent_wr'] - df['camp2_recent_wr']

    # Head-to-head win rate (camp1 perspective)
    df['pair'] = df.apply(lambda x: tuple(sorted([x['camp1_team_id'], x['camp2_team_id']])), axis=1)
    df['win_camp_team_id'] = np.where(df['win_camp'] == 1, df['camp1_team_id'], df['camp2_team_id'])
    df['pair_win'] = (df['pair'].apply(lambda p: p[0]) == df['win_camp_team_id']).astype(int)
    df['history_win_rate'] = df.groupby('pair')['pair_win'].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df['h2h_camp1_wr'] = np.where(
        df['camp1_team_id'] == df['pair'].apply(lambda p: p[0]),
        df['history_win_rate'],
        1 - df['history_win_rate']
    )

    return df


def build_bp_hero_features(battles: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Build V2 BP hero strength and team-hero proficiency features.

    Returns battles with added hero/team-hero feature columns.
    """
    df = battles.copy()

    # Add win to players
    time_map = df[['battle_id', 'battles_time_id', 'win_camp']]
    p = players.merge(time_map, on='battle_id', how='left')
    p['win'] = (p['camp'] == p['win_camp']).astype(int)

    # Hero historical win rate
    p = p.sort_values(['hero_id', 'battles_time_id']).reset_index(drop=True)
    p['hero_games_before'] = p.groupby('hero_id').cumcount()
    p['hero_wr_before'] = p.groupby('hero_id')['win'].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Aggregate to battle + camp
    hero_camp = p.groupby(['battle_id', 'camp']).agg(
        avg_hero_wr=('hero_wr_before', 'mean'),
        min_hero_wr=('hero_wr_before', 'min'),
        max_hero_wr=('hero_wr_before', 'max'),
        avg_hero_games=('hero_games_before', 'mean'),
    ).reset_index()

    # Team-hero proficiency
    p = p.sort_values(['team_id', 'hero_id', 'battles_time_id']).reset_index(drop=True)
    p['team_hero_games_before'] = p.groupby(['team_id', 'hero_id']).cumcount()
    p['team_hero_wr_before'] = p.groupby(['team_id', 'hero_id'])['win'].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    team_hero_camp = p.groupby(['battle_id', 'camp']).agg(
        avg_team_hero_wr=('team_hero_wr_before', 'mean'),
        avg_team_hero_games=('team_hero_games_before', 'mean'),
        new_hero_count=('team_hero_games_before', lambda s: (s == 0).sum()),
    ).reset_index()

    # Merge hero features
    for camp_num, prefix in [(1, 'camp1'), (2, 'camp2')]:
        sub = hero_camp[hero_camp['camp'] == camp_num].drop(columns='camp')
        sub = sub.rename(columns={c: f'{prefix}_{c}' for c in sub.columns if c != 'battle_id'})
        df = df.merge(sub, on='battle_id', how='left')

    df['hero_wr_diff'] = df['camp1_avg_hero_wr'] - df['camp2_avg_hero_wr']

    # Merge team-hero features
    for camp_num, prefix in [(1, 'camp1'), (2, 'camp2')]:
        sub = team_hero_camp[team_hero_camp['camp'] == camp_num].drop(columns='camp')
        sub = sub.rename(columns={c: f'{prefix}_{c}' for c in sub.columns if c != 'battle_id'})
        df = df.merge(sub, on='battle_id', how='left')

    df['team_hero_wr_diff'] = df['camp1_avg_team_hero_wr'] - df['camp2_avg_team_hero_wr']
    df['new_hero_count_diff'] = df['camp1_new_hero_count'] - df['camp2_new_hero_count']

    return df
