"""
Realtime feature engineering for in-game win probability models.

Code should be extracted here after notebooks 07 and 08 are stable.
"""

from __future__ import annotations

import pandas as pd


def build_snapshot_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features from realtime snapshots.

    TODO:
    - Extract feature logic from notebook 08.
    - Ensure each feature is available at the snapshot timestamp.
    - Never use final post-game fields except the training label.
    """
    raise NotImplementedError("Extract implementation from notebook 08 after it is finalized.")
