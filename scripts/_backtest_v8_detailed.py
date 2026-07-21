"""Deprecated: use scripts/backtest_realtime_v9.py instead."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

print("[DEPRECATED] 请改用: python scripts/backtest_realtime_v9.py")
print("旧 V8 自建特征已废弃，避免与 FeatureBuilder 漂移。")

from backtest_realtime_v9 import main

if __name__ == "__main__":
    main()
