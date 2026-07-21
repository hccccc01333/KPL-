# Realtime Data

This folder stores live in-game snapshots collected during matches.

## Files

- `raw_snapshots/`: raw JSON responses collected from the live battle API
- `realtime_snapshots.csv`: parsed snapshot table, one row per `battle_id + collected_at`

## Important

Do not overwrite snapshots from the same battle. Realtime modeling depends on preserving the full time series.
