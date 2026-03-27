from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline
import qfpytoolbox
from qfrsu_dashboard import rsu


def test_pipeline_uses_local_rsu_module() -> None:
    assert pipeline.run_rsu_pipeline is rsu.run_rsu_pipeline
    assert pipeline.run_rsu_pipeline.__module__ == "qfrsu_dashboard.rsu"


def test_dashboard_owns_rsu_orchestration_boundary() -> None:
    # Keep app orchestration out of qfpytoolbox package public API.
    assert not hasattr(qfpytoolbox, "run_rsu_pipeline")
