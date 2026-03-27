"""Robust RSU loaders for mixed flat-file inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from qfrsu_dashboard.rsu_encoding import detect_encoding, repair_dataframe

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

MENAGE_DTYPES = {
    "menage_ano": "Int64",
    "milieu": "string",
    "region_id": "Int64",
    "region": "string",
    "province_id": "Int64",
    "province": "string",
    "commune_id": "Int64",
    "commune": "string",
    "type_menage": "string",
    "taille_menage": "Int64",
    "etat_matrimonial_cm": "string",
    "genre_cm": "string",
    "sexe_id": "Int64",
    "date_naissance_cm": "string",
}
SCORE_DTYPES = {
    "menage_ano": "Int64",
    "score_id_ano": "Int64",
    "type_demande": "string",
    "score_corrige": "float64",
    "score_calcule": "float64",
    "date_calcul": "string",
}
PROGRAMME_DTYPES = {"menage_ano": "Int64"}
BENEFICIAIRE_DTYPES = {
    "menage_ano": "Int64",
    "partner_id": "string",
    "motif": "string",
    "date_insert": "string",
    "actif": "boolean",
}

__all__ = [
    "load_menage",
    "load_scores",
    "load_scores_multi",
    "load_programme",
    "load_all_programmes",
    "load_beneficiaire",
]


def _read_csv_safe(
    path: PathLike,
    *,
    dtypes: dict[str, str] | None = None,
    parse_dates: list[str] | None = None,
    chunk_size: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    enc = detect_encoding(p)
    kwargs = dict(encoding=enc, encoding_errors="replace", on_bad_lines="warn", engine="python", usecols=usecols)
    try:
        if chunk_size:
            chunks = [c for c in pd.read_csv(p, chunksize=chunk_size, **kwargs)]
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(p, **kwargs)
    except Exception:
        kwargs["encoding"] = "latin-1"
        if chunk_size:
            chunks = [c for c in pd.read_csv(p, chunksize=chunk_size, **kwargs)]
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(p, **kwargs)

    df = repair_dataframe(df, verbose=False)
    if dtypes:
        for col, dtype in dtypes.items():
            if col not in df.columns:
                continue
            try:
                if dtype == "Int64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception:
                log.warning("could not cast %s to %s", col, dtype)
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_menage(path: PathLike, chunk_size: int | None = None) -> pd.DataFrame:
    df = _read_csv_safe(path, dtypes=MENAGE_DTYPES, parse_dates=["date_naissance_cm"], chunk_size=chunk_size)
    force_repair_cols = [
        "milieu",
        "region",
        "province",
        "commune",
        "type_menage",
        "etat_matrimonial_cm",
        "genre_cm",
    ]
    existing_force_cols = [c for c in force_repair_cols if c in df.columns]
    if existing_force_cols:
        df = repair_dataframe(df, columns=existing_force_cols, verbose=False)
    if "milieu" in df.columns:
        df["milieu"] = df["milieu"].astype("string").str.strip().str.capitalize()
    if "genre_cm" in df.columns:
        df["genre_cm"] = (
            df["genre_cm"]
            .astype("string")
            .str.strip()
            .str.capitalize()
            .replace(
                {
                    "Masculin": "Homme",
                    "Male": "Homme",
                    "M": "Homme",
                    "1": "Homme",
                    "Feminin": "Femme",
                    "Féminin": "Femme",
                    "Female": "Femme",
                    "F": "Femme",
                    "2": "Femme",
                }
            )
        )
    if "menage_ano" in df.columns:
        df = df.drop_duplicates(subset=["menage_ano"], keep="first")
    return df


def _postprocess_scores(df: pd.DataFrame, max_score: float) -> pd.DataFrame:
    if "type_demande" in df.columns:
        df["type_demande"] = df["type_demande"].astype("string").str.strip()
    if {"score_corrige", "score_calcule"}.issubset(df.columns):
        df["score_final"] = df["score_corrige"].fillna(df["score_calcule"])
        df["was_corrected"] = df["score_corrige"].notna() & (df["score_corrige"] != df["score_calcule"])
    if "date_calcul" in df.columns:
        df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    if "score_final" in df.columns and max_score is not None:
        df = df[df["score_final"] <= max_score].copy()
    return df


def load_scores(path: PathLike, chunk_size: int = 200000, max_score: float = 15.0) -> pd.DataFrame:
    """Load score events with chunked dedup for large CSV files.

    For large files, a global ``drop_duplicates`` on the full frame is too costly.
    We deduplicate incrementally by ``score_id_ano`` when available.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    if chunk_size <= 0:
        df = _read_csv_safe(
            p,
            dtypes=SCORE_DTYPES,
            parse_dates=["date_calcul"],
            chunk_size=None,
            usecols=list(SCORE_DTYPES.keys()),
        )
        df = _postprocess_scores(df, max_score=max_score)
        if "score_id_ano" in df.columns:
            df = df.drop_duplicates(subset=["score_id_ano"], keep="first")
        return df

    enc = detect_encoding(p)
    kwargs = dict(
        encoding=enc,
        encoding_errors="replace",
        on_bad_lines="warn",
        engine="python",
        chunksize=chunk_size,
        usecols=list(SCORE_DTYPES.keys()),
    )
    try:
        reader = pd.read_csv(p, **kwargs)
    except Exception:
        kwargs["encoding"] = "latin-1"
        reader = pd.read_csv(p, **kwargs)

    chunks: list[pd.DataFrame] = []
    seen_ids: set[int] = set()
    rows_in = 0
    rows_kept = 0

    for idx, chunk in enumerate(reader, start=1):
        chunk = repair_dataframe(chunk, verbose=False)
        rows_in += len(chunk)

        for col, dtype in SCORE_DTYPES.items():
            if col not in chunk.columns:
                continue
            if dtype == "Int64":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")
            elif dtype == "float64":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            else:
                chunk[col] = chunk[col].astype(dtype)

        chunk = _postprocess_scores(chunk, max_score=max_score)

        if "score_id_ano" in chunk.columns:
            keys = pd.to_numeric(chunk["score_id_ano"], errors="coerce")
            is_new = ~keys.isin(seen_ids)
            chunk = chunk[is_new].copy()
            seen_ids.update(int(v) for v in keys[is_new].dropna().astype("int64"))
        else:
            subset = [c for c in ["menage_ano", "date_calcul", "type_demande", "score_corrige", "score_calcule"] if c in chunk]
            if subset:
                chunk = chunk.drop_duplicates(subset=subset, keep="first")

        rows_kept += len(chunk)
        chunks.append(chunk)
        if idx % 10 == 0:
            log.info("Scores loading progress: chunks=%d rows_in=%d rows_kept=%d", idx, rows_in, rows_kept)

    if not chunks:
        return pd.DataFrame(columns=list(SCORE_DTYPES.keys()) + ["score_final", "was_corrected"])
    out = pd.concat(chunks, ignore_index=True)
    log.info("Scores loaded: rows_in=%d rows_kept=%d", rows_in, len(out))
    return out


def load_scores_multi(
    paths: list[PathLike], output_dir: PathLike, chunk_size: int = 200000, max_score: float = 15.0
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = [load_scores(p, chunk_size=chunk_size, max_score=max_score) for p in paths]
    df = pd.concat(all_frames, ignore_index=True)
    if "score_id_ano" in df.columns:
        df = df.sort_values("date_calcul" if "date_calcul" in df.columns else df.index.name).drop_duplicates(
            subset=["score_id_ano"], keep="last"
        )
    out = out_dir / "scores_merged.parquet"
    df.to_parquet(out, index=False)
    return out


def load_programme(path: PathLike, programme_name: str) -> pd.DataFrame:
    df = _read_csv_safe(path, dtypes=PROGRAMME_DTYPES)
    df["programme"] = str(programme_name).upper()
    return df[["menage_ano", "programme"]]


def load_all_programmes(paths: dict[str, PathLike]) -> pd.DataFrame:
    frames = [load_programme(p, name) for name, p in paths.items()]
    if not frames:
        return pd.DataFrame(columns=["menage_ano", "programme"])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_beneficiaire(path: PathLike) -> pd.DataFrame:
    df = _read_csv_safe(path, dtypes=BENEFICIAIRE_DTYPES, parse_dates=["date_insert"])
    if "partner_id" in df.columns:
        df["partner_id"] = df["partner_id"].astype("string").str.upper()
    return df
