"""Encoding detection and mojibake repair helpers for RSU flat files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

MOJIBAKE_MAP: dict[str, str] = {
    "Rabat-SalÃ©-KÃ©nitra": "Rabat-Salé-Kénitra",
    "Tanger-TÃ©touan-Al HoceÃ¯ma": "Tanger-Tétouan-Al Hoceïma",
    "FÃ¨s-MeknÃ¨s": "Fès-Meknès",
    "BÃ©ni Mellal-KhÃ©nifra": "Béni Mellal-Khénifra",
    "DrÃ¢a-Tafilalet": "Drâa-Tafilalet",
    "LaÃ¢youne-Sakia El Hamra": "Laâyoune-Sakia El Hamra",
    "GuelmimOued Noun": "Guelmim-Oued Noun",
    "Rabat-Sal\x82-K\x82nitra": "Rabat-Salé-Kénitra",
    "F\x82s-Mekn\x8as": "Fès-Meknès",
    "Mari\x82(e)": "Marié(e)",
    "MariÃ©(e)": "Marié(e)",
    "CÃ©libataire": "Célibataire",
    "DivorÃ©(e)": "Divorcé(e)",
    "SÃ©parÃ©(e)": "Séparé(e)",
    "FÃ©minin": "Féminin",
    "Feminin": "Féminin",
    "DÃ©cÃ©dÃ©(e)": "Décédé(e)",
}


def detect_encoding(path: str | Path, n_bytes: int = 50000) -> str:
    """Detect input encoding with chardet if available."""
    p = Path(path)
    try:
        import chardet  # noqa: PLC0415

        raw = p.read_bytes()[:n_bytes]
        d = chardet.detect(raw)
        return d.get("encoding") or "latin-1"
    except ImportError:
        return "latin-1"


def _recover_mojibake(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            # Secondary recovery for cp1252-like control bytes mis-decoded as latin-1.
            return text.encode("latin-1").decode("cp1252")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text


def _apply_map(text: str, mapping: dict[str, str]) -> str:
    fixed = text
    for bad, good in mapping.items():
        fixed = fixed.replace(bad, good)
    return fixed


def repair_series(s: pd.Series, *, extra_map: Optional[dict[str, str]] = None) -> pd.Series:
    mapping = {**MOJIBAKE_MAP, **(extra_map or {})}
    fixed = s.astype(str).map(_recover_mojibake).map(lambda x: _apply_map(x, mapping))
    return fixed.where(s.notna(), other=None)


def repair_dataframe(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    verbose: bool = False,
    extra_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Repair mojibake corruption in object columns."""
    out = df.copy()
    target_cols = columns
    if target_cols is None:
        target_cols = []
        for c in out.columns:
            s = out[c]
            if not (s.dtype == "object" or pd.api.types.is_string_dtype(s)):
                continue
            s_txt = s.astype(str)
            has_utf8_mojibake = s_txt.str.contains("Ã|â€|Â", na=False, regex=True).any()
            has_cp1252_controls = s_txt.str.contains("\x80|\x82|\x83|\x9c", na=False, regex=True).any()
            if has_utf8_mojibake or has_cp1252_controls:
                target_cols.append(c)

    for col in target_cols:
        s = out[col]
        if not (s.dtype == "object" or pd.api.types.is_string_dtype(s)):
            continue
        before = s.astype(str).str.contains("Ã|â€|Â", na=False, regex=True).sum()
        out[col] = repair_series(s, extra_map=extra_map)
        if verbose:
            after = out[col].astype(str).str.contains("Ã|â€|Â", na=False, regex=True).sum()
            log.info("repair %-30s %d -> %d", col, int(before), int(after))
    return out
