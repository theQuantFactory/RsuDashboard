from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qfrsu_dashboard.rsu_loaders import load_scores


def test_load_scores_chunked_dedup_by_score_id(tmp_path):
    csv_path = tmp_path / "scores.csv"
    src = pd.DataFrame(
        {
            "menage_ano": [1, 2, 3, 4],
            "score_id_ano": [10, 20, 10, 40],
            "type_demande": ["inscription", "inscription", "correction", "inscription"],
            "score_corrige": [None, None, 6.5, None],
            "score_calcule": [5.0, 7.0, 6.0, 8.0],
            "date_calcul": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )
    src.to_csv(csv_path, index=False)

    out = load_scores(csv_path, chunk_size=2, max_score=15.0)

    assert len(out) == 3
    assert set(out["score_id_ano"].dropna().astype(int).tolist()) == {10, 20, 40}
    assert "score_final" in out.columns
    assert "was_corrected" in out.columns


def test_load_scores_applies_max_score_filter(tmp_path):
    csv_path = tmp_path / "scores.csv"
    src = pd.DataFrame(
        {
            "menage_ano": [1, 2],
            "score_id_ano": [100, 200],
            "type_demande": ["inscription", "inscription"],
            "score_corrige": [None, None],
            "score_calcule": [9.0, 16.0],
            "date_calcul": ["2025-01-01", "2025-01-02"],
        }
    )
    src.to_csv(csv_path, index=False)

    out = load_scores(csv_path, chunk_size=1, max_score=15.0)
    assert len(out) == 1
    assert float(out["score_final"].iloc[0]) == 9.0
