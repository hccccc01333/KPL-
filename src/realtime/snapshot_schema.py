"""
Realtime snapshot schema.

One row represents one snapshot of one battle at one point in time.
For simulated snapshots: one row per battle per minute_bin (3/5/8/10).
For real-time snapshots: one row per battle per collection timestamp.
"""

SNAPSHOT_COLUMNS = [
    "battle_id",
    "snapshot_time_sec",
    "minute_bin",
    "camp1_gold",
    "camp2_gold",
    "gold_diff",
    "camp1_kill_num",
    "camp2_kill_num",
    "kill_diff",
    "camp1_push_tower_num",
    "camp2_push_tower_num",
    "tower_diff",
    "camp1_tyrant",
    "camp2_tyrant",
    "tyrant_diff",
    "win_camp",
    "is_simulated",
]

MINUTE_BINS = [3, 5, 8, 10]

# Scaling exponents for simulation:
# gold grows ~linearly, kills slightly back-loaded, towers late-game
SCALING_ALPHA = {
    "gold": 1.0,
    "kill": 1.3,
    "tower": 1.5,
    "tyrant": 1.2,
}
