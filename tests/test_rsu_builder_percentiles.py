from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qfrsu_dashboard.rsu_builder import build_score_timeseries


def test_score_timeseries_dashboard_profile_includes_percentiles():
    df_master = pd.DataFrame(
        {
            "menage_ano": [1, 2, 3, 1, 2, 3],
            "date_calcul": pd.to_datetime(
                ["2025-01-06", "2025-01-06", "2025-01-06", "2025-01-13", "2025-01-13", "2025-01-13"]
            ),
            "score_final": [8.0, 10.0, 12.0, 7.5, 9.5, 11.5],
            "type_demande": ["inscription"] * 6,
        }
    )

    out = build_score_timeseries(
        df_master,
        include_demo_breakdowns=False,
        include_percentiles=True,
        timeseries_freq="W-MON",
        batch_size=0,
    )["daily_stats"]

    assert not out.empty
    for col in ["score_p10", "score_p25", "score_p75", "score_p90"]:
        assert col in out.columns
        assert out[col].notna().any()
