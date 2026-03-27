from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qfrsu_dashboard.rsu_builder import build_reentry_analysis


def test_reentry_summary_splits_zero_reentries_populations():
    df_master = pd.DataFrame(
        {
            "menage_ano": [1, 1, 2, 2, 3, 3, 4, 4],
            "programme": ["AMOT"] * 8,
            "date_calcul": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-01",
                    "2025-02-01",
                ]
            ),
            # 1: always eligible, 2: never eligible, 3: lost no return, 4: one re-entry
            "eligible": [True, True, False, False, True, False, False, True],
        }
    )

    detail, summary = build_reentry_analysis(df_master)
    assert not detail.empty
    assert not summary.empty

    sub = summary[summary["programme"] == "AMOT"].set_index("n_reentries")
    assert int(sub.loc["toujours éligible", "n_menages"]) == 1
    assert int(sub.loc["jamais éligible", "n_menages"]) == 1
    assert int(sub.loc["perdu sans retour", "n_menages"]) == 1
    assert int(sub.loc["1", "n_menages"]) == 1
