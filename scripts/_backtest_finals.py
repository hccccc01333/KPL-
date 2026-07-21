"""Deprecated: use scripts/backtest_realtime_v9.py instead."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

print("[DEPRECATED] 请改用: python scripts/backtest_realtime_v9.py")
print("该脚本已统一走 kpl_official_core，并输出真实 holdout 分阶段 Brier/ECE。")

from backtest_realtime_v9 import main

if __name__ == "__main__":
    main()
