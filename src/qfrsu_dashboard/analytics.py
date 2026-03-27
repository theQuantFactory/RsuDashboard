"""Analytics pipeline helpers for parquet snapshots and dashboard cache.

This module provides a small, source-agnostic workflow:
1) load tabular sources into DataFrames,
2) save snapshots as parquet,
3) compute derived frames,
4) prebake dashboard cache payloads.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from qfpytoolbox.io.dataframes import read_dataframe

FrameMap = dict[str, pd.DataFrame]
CalcFn = Callable[[FrameMap], pd.DataFrame]
AggFn = Callable[[FrameMap], Any]
log = logging.getLogger(__name__)

__all__ = [
    "load_frames",
    "save_frames_as_parquet",
    "load_parquet_frames",
    "run_calculations",
    "build_dashboard_cache",
    "load_dashboard_cache",
]


def load_frames(sources: dict[str, Any], *, read_options: dict[str, dict[str, Any]] | None = None) -> FrameMap:
    """Load named data sources as DataFrames.

    Each source can be any input accepted by :func:`qfpytoolbox.io.read_dataframe`
    (path, IO stream, media object, or DataFrame).
    """
    options = read_options or {}
    frames: FrameMap = {}
    for name, source in sources.items():
        kwargs = options.get(name, {})
        frames[name] = read_dataframe(source, **kwargs)
    return frames


def save_frames_as_parquet(
    frames: FrameMap,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
    skip_empty: bool = False,
) -> dict[str, Path]:
    """Persist named frames to ``<output_dir>/<name>.parquet``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for name, df in frames.items():
        if skip_empty and df.empty:
            continue
        path = out_dir / f"{name}.parquet"
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        n_rows, n_cols = len(df), len(df.columns)
        t0 = time.perf_counter()
        log.info("Writing parquet frame '%s' (%d rows x %d cols) -> %s", name, n_rows, n_cols, path)
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
        elapsed = time.perf_counter() - t0
        log.info("Finished frame '%s' in %.2fs", name, elapsed)
        out[name] = path
    return out


def load_parquet_frames(input_dir: str | Path, *, names: list[str] | None = None) -> FrameMap:
    """Load parquet snapshots from a directory into named DataFrames."""
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise ValueError(f"Snapshot directory not found: {in_dir}")

    frames: FrameMap = {}
    if names is None:
        paths = sorted(in_dir.glob("*.parquet"))
        for path in paths:
            try:
                frames[path.stem] = pd.read_parquet(path)
            except Exception:
                # Keep loading other valid snapshots in production runs.
                continue
        return frames

    for name in names:
        path = in_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")
        frames[name] = pd.read_parquet(path)
    return frames


def run_calculations(base_frames: FrameMap, calculations: dict[str, CalcFn]) -> FrameMap:
    """Run named calculation functions and return base + derived frames."""
    frames: FrameMap = {k: v.copy() for k, v in base_frames.items()}
    for name, fn in calculations.items():
        result = fn(frames)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Calculation '{name}' must return a pandas.DataFrame")
        frames[name] = result
    return frames


def build_dashboard_cache(
    *,
    frames: FrameMap | None = None,
    parquet_dir: str | Path | None = None,
    aggregations: dict[str, AggFn | str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a cache dictionary for fast dashboard loading.

    ``aggregations`` supports two styles:
    - callable(frames) -> any result
    - SQL query string (requires duckdb and ``parquet_dir``)
    """
    if frames is None and parquet_dir is None:
        raise ValueError("Provide either frames or parquet_dir")

    if frames is None:
        frames = load_parquet_frames(parquet_dir)  # type: ignore[arg-type]

    cache: dict[str, Any] = {
        "_baked_at": dt.datetime.now(),
        "_frame_names": sorted(frames.keys()),
    }

    aggs = aggregations or {}
    if aggs:
        for key, agg in aggs.items():
            if callable(agg):
                cache[key] = agg(frames)
            elif isinstance(agg, str):
                if parquet_dir is None:
                    raise ValueError("SQL aggregations require parquet_dir")
                try:
                    import duckdb  # noqa: PLC0415
                except ImportError as exc:  # pragma: no cover - environment dependent
                    raise ImportError("duckdb is required for SQL cache aggregations") from exc
                con = duckdb.connect()
                try:
                    con.execute(f"SET VARIABLE parquet_dir='{Path(parquet_dir).as_posix()}'")
                    cache[key] = con.execute(agg).df()
                finally:
                    con.close()
            else:
                raise TypeError(f"Unsupported aggregation type for key '{key}'")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return cache


def load_dashboard_cache(cache_path: str | Path) -> dict[str, Any]:
    """Load a previously baked dashboard cache (.pkl)."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError("Invalid cache format: expected a dictionary payload")
    return obj
