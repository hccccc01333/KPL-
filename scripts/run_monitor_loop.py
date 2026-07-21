"""Cross-platform monitor supervisor (works in Git Bash / PowerShell / cmd)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "official_match_monitor.py"
RESTART_SEC = 30


def main() -> None:
    extra = sys.argv[1:]
    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] official_match_monitor.py starting...")
        proc = subprocess.run([sys.executable, str(SCRIPT), *extra])
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exited (code={proc.returncode}), "
            f"restart in {RESTART_SEC}s. Ctrl+C to stop."
        )
        try:
            time.sleep(RESTART_SEC)
        except KeyboardInterrupt:
            print("\n[STOP] supervisor exited.")
            break


if __name__ == "__main__":
    main()
