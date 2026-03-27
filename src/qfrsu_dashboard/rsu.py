"""RSU-oriented production pipeline built on top of qfpytoolbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qfrsu_dashboard.analytics import save_frames_as_parquet
from qfrsu_dashboard.rsu_builder import (
    build_beneficiaire_enriched_events,
    build_churn_timeline,
    build_delta_frame,
    build_eligibility_churn,
    build_master_events,
    build_menage_timeline,
    build_menage_trajectory,
    build_monthly_beneficiaire_flows,
    build_monthly_eligibility_flows,
    build_near_threshold_timeseries,
    build_pivot_wide,
    build_programme_frame,
    build_reentry_analysis,
    build_score_timeseries,
    build_volatility_summary,
)
from qfrsu_dashboard.rsu_loaders import (
    load_all_programmes,
    load_beneficiaire,
    load_menage,
    load_scores,
    load_scores_multi,
)

__all__ = [
    "discover_rsu_sources",
    "run_csv_etl",
    "run_rsu_pipeline",
    "load_menage",
    "load_scores",
    "load_scores_multi",
    "load_all_programmes",
    "load_beneficiaire",
    "build_master_events",
    "build_menage_timeline",
    "build_delta_frame",
    "build_programme_frame",
    "build_beneficiaire_enriched_events",
    "build_pivot_wide",
    "build_volatility_summary",
    "build_eligibility_churn",
    "build_menage_trajectory",
    "build_score_timeseries",
    "build_monthly_eligibility_flows",
    "build_monthly_beneficiaire_flows",
    "build_churn_timeline",
    "build_reentry_analysis",
    "build_near_threshold_timeseries",
]


def discover_rsu_sources(input_dir: str | Path) -> dict[str, str]:
    """Discover RSU input files in mixed formats under a directory."""
    in_dir = Path(input_dir)
    out: dict[str, str] = {}
    aliases: dict[str, tuple[str, ...]] = {
        "menage": ("menage",),
        "scores": ("scores", "score"),
        "programmes": ("programmes", "programme"),
        "beneficiaire": ("beneficiaire", "beneficiaires"),
    }
    for key, stems in aliases.items():
        found = False
        for stem in stems:
            for ext in ("csv", "arrow", "xlsx"):
                p = in_dir / f"{stem}.{ext}"
                if p.exists():
                    out[key] = str(p)
                    found = True
                    break
            if found:
                break
    return out


def _raw_row_count(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    try:
        with p.open("rb") as f:
            count = sum(1 for _ in f)
        return max(count - 1, 0)
    except Exception:
        return 0


def _build_qc_summary(
    scores_path: str | Path,
    menage_path: str | Path,
    df_scores_raw: pd.DataFrame,
    df_menage_raw: pd.DataFrame,
    df_master: pd.DataFrame,
    *,
    max_score: float = 15.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _pct(dropped: int, before: int) -> float:
        return round((dropped / before * 100.0), 4) if before else 0.0

    raw_score = _raw_row_count(scores_path)
    raw_menage = _raw_row_count(menage_path)
    n_scores_after = len(df_scores_raw)
    n_menage_after = len(df_menage_raw)

    # Avoid a second heavy pass on giant score files while building QC summary.
    # We keep exact total dropped count and attribute it to duplicate-cleanup stage.
    n_scores_no_max = n_scores_after
    n_aberrant = int((df_scores_raw["score_final"] > max_score).sum()) if "score_final" in df_scores_raw.columns else 0
    n_score_dupes = max(raw_score - n_scores_after, 0)
    n_menage_dupes = max(raw_menage - n_menage_after, 0)

    rows.append(
        {
            "metric": "scores_raw_rows",
            "label_fr": "Lignes brutes score chargées",
            "before": raw_score,
            "after": n_scores_after,
            "dropped": max(raw_score - n_scores_after, 0),
            "pct_dropped": _pct(max(raw_score - n_scores_after, 0), raw_score),
            "note": "Volume brut de score.csv comparé au dataset final après nettoyage.",
        }
    )
    rows.append(
        {
            "metric": "scores_duplicate_rows",
            "label_fr": "Doublons exacts (scores)",
            "before": raw_score,
            "after": raw_score - n_score_dupes,
            "dropped": n_score_dupes,
            "pct_dropped": _pct(n_score_dupes, raw_score),
            "note": "Lignes retirées au nettoyage (doublons et autres exclusions avant dataset final).",
        }
    )
    rows.append(
        {
            "metric": "scores_aberrant",
            "label_fr": "Scores aberrants (> max)",
            "before": n_scores_no_max,
            "after": n_scores_after,
            "dropped": n_aberrant,
            "pct_dropped": _pct(n_aberrant, n_scores_no_max),
            "note": f"Scores avec score_final > {max_score} exclus (sur données déjà nettoyées).",
        }
    )
    rows.append(
        {
            "metric": "menage_raw_rows",
            "label_fr": "Lignes brutes ménage chargées",
            "before": raw_menage,
            "after": n_menage_after,
            "dropped": n_menage_dupes,
            "pct_dropped": _pct(n_menage_dupes, raw_menage),
            "note": "Volume brut de menage.csv comparé au dataset ménage nettoyé.",
        }
    )
    rows.append(
        {
            "metric": "menage_duplicate_ids",
            "label_fr": "Doublons ID ménage",
            "before": raw_menage,
            "after": n_menage_after,
            "dropped": n_menage_dupes,
            "pct_dropped": _pct(n_menage_dupes, raw_menage),
            "note": "Conservation de la première occurrence par menage_ano.",
        }
    )
    if "menage_ano" in df_menage_raw.columns and "menage_ano" in df_master.columns:
        n_no_score = int((~df_menage_raw["menage_ano"].isin(df_master["menage_ano"].dropna().unique())).sum())
        rows.append(
            {
                "metric": "menage_no_score",
                "label_fr": "Ménages sans score associé",
                "before": n_menage_after,
                "after": n_menage_after - n_no_score,
                "dropped": n_no_score,
                "pct_dropped": _pct(n_no_score, n_menage_after),
                "note": "Ménages présents dans le référentiel sans événement score.",
            }
        )
    if "programme" in df_master.columns and "menage_ano" in df_master.columns:
        n_nc = df_master[df_master["programme"].astype(str).str.upper() == "NON CLASSIFIE"]["menage_ano"].nunique()
        n_tot = df_master["menage_ano"].nunique()
        rows.append(
            {
                "metric": "menage_non_classifie",
                "label_fr": "Ménages non classifiés",
                "before": n_tot,
                "after": n_tot - n_nc,
                "dropped": n_nc,
                "pct_dropped": _pct(n_nc, n_tot),
                "note": "Ménages non appariés aux listes programme AMOT/ASD/AMOA.",
            }
        )
    return pd.DataFrame(rows)


def run_csv_etl(
    menage_path: str | Path,
    scores_path: str | Path,
    programme_paths: dict[str, str | Path],
    beneficiaire_path: str | Path | None = None,
    save_snapshots: bool = False,
    snapshot_dir: str | Path | None = None,
    chunk_size: int = 200000,
    verbose: bool = True,
    max_score: float = 15.0,
    snapshot_profile: str = "full",
    timeseries_freq: str = "W-MON",
    timeseries_batch_size: int = 0,
    return_frames: bool = True,
) -> dict[str, pd.DataFrame]:
    del verbose
    out_dir = snapshot_dir or "snapshots/csv"

    frames: dict[str, pd.DataFrame] = {}

    def _emit(name: str, df: pd.DataFrame) -> None:
        if save_snapshots:
            save_frames_as_parquet({name: df}, out_dir, overwrite=True, skip_empty=False)
        if return_frames:
            frames[name] = df
    df_menage = load_menage(menage_path)
    if str(scores_path).lower().endswith(".parquet"):
        df_scores = pd.read_parquet(scores_path)
    else:
        df_scores = load_scores(scores_path, chunk_size=chunk_size, max_score=max_score)
    df_programmes = load_all_programmes(programme_paths)
    df_beneficiaire = pd.DataFrame() if not beneficiaire_path else load_beneficiaire(beneficiaire_path)
    _emit("raw_menage", df_menage)
    _emit("raw_programmes", df_programmes)
    _emit("raw_beneficiaire", df_beneficiaire)

    df_master = build_master_events(df_menage, df_scores, df_programmes)
    _emit("master_events", df_master)
    df_delta = build_delta_frame(df_master)
    _emit("delta_frame", df_delta)
    df_timeline = build_menage_timeline(df_master)
    _emit("menage_timeline", df_timeline)
    df_pivot = build_pivot_wide(df_master) if snapshot_profile == "full" else pd.DataFrame()
    df_trajectory = build_menage_trajectory(df_master)
    _emit("menage_trajectory", df_trajectory)
    df_ts = build_score_timeseries(
        df_master,
        include_demo_breakdowns=(snapshot_profile == "full"),
        # Keep percentiles in dashboard profile for Tendances Temporelles decile charts.
        include_percentiles=True,
        timeseries_freq=timeseries_freq,
        batch_size=timeseries_batch_size,
    )["daily_stats"]
    _emit("score_timeseries", df_ts)
    df_near = build_near_threshold_timeseries(
        df_master,
        timeseries_freq=timeseries_freq,
        batch_size=timeseries_batch_size,
    )
    _emit("near_threshold_timeseries", df_near)
    df_monthly = build_monthly_eligibility_flows(df_master)
    _emit("monthly_eligibility_flows", df_monthly)
    df_churn = build_churn_timeline(df_master)
    _emit("churn_timeline", df_churn)
    df_reentry_detail, df_reentry_summary = build_reentry_analysis(df_master)
    _emit("reentry_summary", df_reentry_summary)
    df_ben_enriched = (
        build_beneficiaire_enriched_events(df_master, df_beneficiaire)
        if (not df_beneficiaire.empty and snapshot_profile == "full")
        else None
    )
    df_ben_monthly = (
        build_monthly_beneficiaire_flows(df_beneficiaire)
        if (not df_beneficiaire.empty and snapshot_profile == "full")
        else None
    )
    df_qc = _build_qc_summary(
        scores_path,
        menage_path,
        df_scores,
        df_menage,
        df_master,
        max_score=max_score,
    )
    _emit("qc_summary", df_qc)

    programme_frames = (
        {f"programme_{p}": build_programme_frame(df_master, p) for p in df_programmes["programme"].unique()}
        if not df_programmes.empty
        else {}
    )
    for pname, pdf in programme_frames.items():
        _emit(pname, pdf)

    if snapshot_profile == "full":
        _emit("pivot_wide", df_pivot)
    if snapshot_profile == "full":
        _emit("raw_scores", df_scores)
        _emit("reentry_detail", df_reentry_detail)
    if df_ben_enriched is not None:
        _emit("beneficiaire_enriched_events", df_ben_enriched)
    if df_ben_monthly is not None:
        _emit("monthly_beneficiaire_flows", df_ben_monthly)

    if not return_frames:
        return {}
    return frames


def run_rsu_pipeline(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    sources: dict[str, Any] | None = None,
    snapshot_profile: str = "full",
    timeseries_freq: str = "W-MON",
    timeseries_batch_size: int = 0,
    chunk_size: int = 200000,
    return_frames: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run RSU computations from discovered raw files and save snapshots."""
    src = sources if sources is not None else discover_rsu_sources(input_dir)
    if "scores" not in src:
        raise ValueError("Missing scores input source")
    programme_paths: dict[str, str] = {}
    if "programmes" in src:
        p = Path(src["programmes"])
        df_prog = pd.read_csv(p)
        if {"programme", "menage_ano"}.issubset(df_prog.columns):
            for prog in df_prog["programme"].dropna().astype(str).str.upper().unique():
                programme_paths[prog] = src["programmes"]
    else:
        for prog in ("AMOT", "ASD", "AMOA"):
            p = Path(input_dir) / f"{prog.lower()}.csv"
            if p.exists():
                programme_paths[prog] = str(p)

    frames = run_csv_etl(
        menage_path=src.get("menage", ""),
        scores_path=src["scores"],
        programme_paths=programme_paths,
        beneficiaire_path=src.get("beneficiaire"),
        save_snapshots=True,
        snapshot_dir=output_dir,
        snapshot_profile=snapshot_profile,
        timeseries_freq=timeseries_freq,
        timeseries_batch_size=timeseries_batch_size,
        chunk_size=chunk_size,
        return_frames=return_frames,
    )
    return frames
