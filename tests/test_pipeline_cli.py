from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


def test_pipeline_forwards_snapshot_profile(monkeypatch, capsys, tmp_path):
    captured: dict[str, object] = {}

    def _fake_run_rsu_pipeline(
        *,
        input_dir,
        output_dir,
        snapshot_profile,
        timeseries_freq,
        timeseries_batch_size,
        sources=None,
    ):
        captured["input_dir"] = input_dir
        captured["output_dir"] = output_dir
        captured["snapshot_profile"] = snapshot_profile
        captured["timeseries_freq"] = timeseries_freq
        captured["timeseries_batch_size"] = timeseries_batch_size
        captured["sources"] = sources
        return {"master_events": pd.DataFrame({"x": [1]})}

    monkeypatch.setattr(pipeline, "run_rsu_pipeline", _fake_run_rsu_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline.py",
            "--input-dir",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--snapshot-profile",
            "dashboard",
        ],
    )

    pipeline.main()
    out = capsys.readouterr().out

    assert "Built 1 frames" in out
    assert captured["input_dir"] == Path(tmp_path / "in")
    assert captured["output_dir"] == Path(tmp_path / "out")
    assert captured["snapshot_profile"] == "dashboard"
    assert captured["timeseries_freq"] == "W-MON"
    assert captured["timeseries_batch_size"] == 0


def test_pipeline_forwards_daily_timeseries_flags(monkeypatch, capsys, tmp_path):
    captured: dict[str, object] = {}

    def _fake_run_rsu_pipeline(
        *,
        input_dir,
        output_dir,
        snapshot_profile,
        timeseries_freq,
        timeseries_batch_size,
        sources=None,
    ):
        captured["input_dir"] = input_dir
        captured["output_dir"] = output_dir
        captured["snapshot_profile"] = snapshot_profile
        captured["timeseries_freq"] = timeseries_freq
        captured["timeseries_batch_size"] = timeseries_batch_size
        captured["sources"] = sources
        return {"master_events": pd.DataFrame({"x": [1]})}

    monkeypatch.setattr(pipeline, "run_rsu_pipeline", _fake_run_rsu_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline.py",
            "--input-dir",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--snapshot-profile",
            "dashboard",
            "--timeseries-freq",
            "daily",
            "--timeseries-batch-size",
            "30",
        ],
    )

    pipeline.main()
    out = capsys.readouterr().out

    assert "Built 1 frames" in out
    assert captured["input_dir"] == Path(tmp_path / "in")
    assert captured["output_dir"] == Path(tmp_path / "out")
    assert captured["snapshot_profile"] == "dashboard"
    assert captured["timeseries_freq"] == "D"
    assert captured["timeseries_batch_size"] == 30
