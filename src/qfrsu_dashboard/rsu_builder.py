"""RSU analytical DataFrame builders."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROGRAMME_THRESHOLDS = {"AMOT": 9.3264284, "ASD": 9.743001}

__all__ = [
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


def build_master_events(df_menage: pd.DataFrame, df_scores: pd.DataFrame, df_programmes: pd.DataFrame) -> pd.DataFrame:
    df = df_scores.merge(df_menage, on="menage_ano", how="left", suffixes=("", "_menage"))
    if not df_programmes.empty:
        df = df.merge(df_programmes, on="menage_ano", how="left")
        df["programme"] = df["programme"].fillna("NON CLASSIFIE")
    else:
        df["programme"] = "NON CLASSIFIE"
    df["programme"] = df["programme"].astype("string").str.upper()
    df["threshold"] = df["programme"].map(PROGRAMME_THRESHOLDS)
    df["dist_threshold"] = df["score_final"] - df["threshold"]
    df["eligible"] = df["dist_threshold"] < 0
    for b in [0.10, 0.25, 0.50]:
        df[f"near_{b:.2f}"] = df["dist_threshold"].abs() <= b
    sort_cols = [c for c in ["menage_ano", "programme", "date_calcul"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def build_menage_timeline(df_master: pd.DataFrame, score_col: str = "score_final") -> pd.DataFrame:
    grp_cols = [c for c in ["menage_ano", "programme"] if c in df_master.columns]
    g = df_master.sort_values("date_calcul").groupby(grp_cols, observed=True)
    agg = g.agg(
        n_events=(score_col, "count"),
        score_latest=(score_col, "last"),
        score_first=(score_col, "first"),
        score_best=(score_col, "min"),
        score_worst=(score_col, "max"),
        date_first=("date_calcul", "min"),
        date_last=("date_calcul", "max"),
    ).reset_index()
    if "eligible" in df_master.columns:
        elig = g["eligible"].agg(ever_eligible="any", currently_eligible="last", first_eligible="first").reset_index()
        agg = agg.merge(elig, on=grp_cols, how="left")
    agg["days_active"] = (agg["date_last"] - agg["date_first"]).dt.days
    return agg


def build_delta_frame(df_master: pd.DataFrame, score_col: str = "score_final", group_cols: Any = None) -> pd.DataFrame:
    _ = group_cols
    sort_keys = ["menage_ano", "date_calcul"] + [c for c in ["programme", "score_id_ano"] if c in df_master.columns]
    df = df_master.sort_values(sort_keys).drop_duplicates(["menage_ano", "date_calcul"], keep="first").copy()
    df = df.sort_values(["menage_ano", "date_calcul"]).reset_index(drop=True)
    df["score_avant"] = df.groupby("menage_ano", observed=True)[score_col].shift(1)
    df["score_apres"] = df[score_col]
    df["date_avant"] = df.groupby("menage_ano", observed=True)["date_calcul"].shift(1)
    df["date_apres"] = df["date_calcul"]
    if "eligible" in df.columns:
        df["side_avant"] = (
            df.groupby("menage_ano", observed=True)["eligible"].shift(1).map({True: "eligible", False: "excluded"})
        )
        df["side_apres"] = df["eligible"].map({True: "eligible", False: "excluded"})
    df = df.dropna(subset=["score_avant"]).copy()
    df["delta_ISE"] = df["score_apres"] - df["score_avant"]
    df["abs_delta"] = df["delta_ISE"].abs()
    df["days_between"] = (df["date_apres"] - df["date_avant"]).dt.days
    df["delta_distribue"] = np.where(df["days_between"] > 90, df["delta_ISE"] / df["days_between"], np.nan)
    if {"side_avant", "side_apres"}.issubset(df.columns):
        conds = [
            (df["side_avant"] == "excluded") & (df["side_apres"] == "eligible"),
            (df["side_avant"] == "eligible") & (df["side_apres"] == "excluded"),
        ]
        df["status_change"] = np.select(conds, ["gained", "lost"], default="stable")
    return df.reset_index(drop=True)


def build_programme_frame(df_master: pd.DataFrame, programme: str) -> pd.DataFrame:
    if "programme" not in df_master.columns:
        raise ValueError("df_master has no programme column")
    return df_master[df_master["programme"] == programme.upper()].copy()


def build_beneficiaire_enriched_events(df_master: pd.DataFrame, df_beneficiaire: pd.DataFrame) -> pd.DataFrame:
    if df_beneficiaire.empty:
        return df_master.copy()
    b = df_beneficiaire.copy()
    m = df_master.copy()
    if "date_insert" in b.columns:
        b["date_insert"] = pd.to_datetime(b["date_insert"], errors="coerce")
    if "date_calcul" in m.columns:
        m["date_calcul"] = pd.to_datetime(m["date_calcul"], errors="coerce")
    out = m.merge(b, on="menage_ano", how="left", suffixes=("", "_beneficiaire"))
    if {"date_insert", "date_calcul"}.issubset(out.columns):
        out = out[(out["date_insert"].isna()) | (out["date_insert"] <= out["date_calcul"])].copy()
    rename_map = {
        "partner_id": "beneficiaire_partner_id",
        "motif": "beneficiaire_motif",
        "actif": "beneficiaire_actif",
        "date_insert": "beneficiaire_date_insert",
    }
    return out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})


def build_pivot_wide(
    df_master: pd.DataFrame, score_col: str = "score_final", pivot_col: str = "type_demande", agg_func: str = "last"
) -> pd.DataFrame:
    group_cols = [c for c in ["menage_ano", "programme"] if c in df_master.columns]
    if pivot_col not in df_master.columns:
        return pd.DataFrame()
    df_sorted = df_master.sort_values(group_cols + ["date_calcul"])
    p = (
        df_sorted.groupby(group_cols + [pivot_col], observed=True)[score_col]
        .agg(agg_func)
        .unstack(pivot_col)
        .reset_index()
    )
    p.columns.name = None
    event_cols = [c for c in p.columns if c not in group_cols]
    return p.rename(columns={c: f"score_{c}" for c in event_cols})


def build_volatility_summary(df_delta: pd.DataFrame, group_cols: list[str], min_events: int = 3) -> pd.DataFrame:
    avail = [c for c in group_cols if c in df_delta.columns]
    if not avail:
        raise ValueError(f"None of {group_cols} found")
    g = df_delta.groupby(avail, observed=True)["delta_ISE"]
    tbl = g.agg(
        n="count",
        mean_delta="mean",
        sigma_delta="std",
        p25=lambda x: x.quantile(0.25),
        median="median",
        p75=lambda x: x.quantile(0.75),
        p90_abs=lambda x: x.abs().quantile(0.9),
    ).reset_index()
    return tbl[tbl["n"] >= min_events].reset_index(drop=True)


def build_eligibility_churn(df_delta: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    req = {"status_change", "side_avant", "side_apres", "menage_ano"}
    missing = req - set(df_delta.columns)
    if missing:
        raise ValueError(f"df_delta missing required columns: {missing}")
    avail = [c for c in group_cols if c in df_delta.columns]
    if not avail:
        raise ValueError(f"None of {group_cols} found")

    def _stats(grp: pd.DataFrame) -> pd.Series:
        n_entrees = int((grp["status_change"] == "gained").sum())
        n_sorties = int((grp["status_change"] == "lost").sum())
        n_elig_av = int((grp["side_avant"] == "eligible").sum())
        n_elig_ap = int((grp["side_apres"] == "eligible").sum())
        stock = (n_elig_av + n_elig_ap) / 2.0
        churn = (n_entrees + n_sorties) / stock if stock > 0 else np.nan
        return pd.Series(
            {
                "n_transitions": len(grp),
                "n_entrees": n_entrees,
                "n_sorties": n_sorties,
                "stock_moyen": stock,
                "churn_eligibilite": churn,
            }
        )

    return df_delta.groupby(avail, observed=True).apply(_stats).reset_index()


def build_menage_trajectory(df_master: pd.DataFrame, score_col: str = "score_final") -> pd.DataFrame:
    df = df_master.sort_values(["menage_ano", "date_calcul"])
    base = (
        df.groupby("menage_ano", observed=True)
        .agg(
            n_events=(score_col, "count"),
            score_actuel=(score_col, "last"),
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            date_premier=("date_calcul", "min"),
            date_dernier=("date_calcul", "max"),
        )
        .reset_index()
    )
    if "eligible" in df.columns:
        elig = (
            df.groupby("menage_ano", observed=True)["eligible"]
            .agg(eligible_inscription="first", eligible_actuel="last")
            .reset_index()
        )
        base = base.merge(elig, on="menage_ano", how="left")
    base["jours_dans_systeme"] = (base["date_dernier"] - base["date_premier"]).dt.days
    return base


def build_score_timeseries(
    df_master: pd.DataFrame,
    score_col: str = "score_final",
    include_demo_breakdowns: bool = True,
    include_percentiles: bool = True,
    timeseries_freq: str = "W-MON",
    batch_size: int = 0,
) -> dict[str, pd.DataFrame]:
    required = ["menage_ano", "date_calcul", score_col]
    if not set(required).issubset(df_master.columns):
        return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}

    demo_cols = [c for c in ["milieu", "region", "genre_cm"] if c in df_master.columns] if include_demo_breakdowns else []
    keep = ["menage_ano", "date_calcul", score_col, "type_demande"] + demo_cols
    if "type_demande" not in df_master.columns:
        df_master = df_master.copy()
        df_master["type_demande"] = ""
    ev = df_master[keep].copy()
    if not pd.api.types.is_datetime64_any_dtype(ev["date_calcul"]):
        ev["date_calcul"] = pd.to_datetime(ev["date_calcul"], errors="coerce", cache=False)
    ev[score_col] = pd.to_numeric(ev[score_col], errors="coerce")
    ev["type_norm"] = ev["type_demande"].astype(str).str.strip().str.lower()
    ev.loc[ev["type_norm"] == "radiation", score_col] = np.nan
    ev = (
        ev.drop(columns=["type_demande"])
        .dropna(subset=["date_calcul"])
        .sort_values(["date_calcul", "menage_ano"])
        .reset_index(drop=True)
    )
    if ev.empty:
        return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}

    t0 = ev["date_calcul"].min()
    tmax = ev["date_calcul"].max()
    on_t0 = ev[ev["date_calcul"] == t0].copy()
    first_per_hh = ev.drop_duplicates(subset=["menage_ano"], keep="first")
    pre = first_per_hh[(first_per_hh["type_norm"] != "inscription") & (first_per_hh["date_calcul"] > t0)].copy()
    pre["date_calcul"] = t0
    seed = pd.concat([on_t0, pre], ignore_index=True).drop_duplicates(subset=["menage_ano"], keep="first")
    if not seed.empty:
        pre_menages = set(pre["menage_ano"].tolist())
        first_dates = ev.groupby("menage_ano", observed=True)["date_calcul"].transform("min")
        ev = ev[~(ev["menage_ano"].isin(pre_menages) & (ev["date_calcul"] == first_dates))].copy()
        ev = pd.concat([seed, ev], ignore_index=True).sort_values(["date_calcul", "menage_ano"]).reset_index(drop=True)
    ev = ev.drop(columns=["type_norm"])

    freq_norm = (timeseries_freq or "W-MON").strip().upper()
    if freq_norm in {"D", "DAILY"}:
        period_points = pd.date_range(t0.normalize(), tmax.normalize(), freq="D")
    else:
        period_points = pd.date_range(t0.to_period("W").start_time, tmax.to_period("W").start_time, freq="W-MON")
    if len(period_points) == 0:
        return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}
    batch_n = int(batch_size) if int(batch_size) > 0 else len(period_points)

    # Fast path for dashboard profile (overall series only).
    # Keeps running-state behavior while avoiding Python dict/list churn.
    if not include_demo_breakdowns:
        hh_codes, _ = pd.factorize(ev["menage_ano"], sort=False)
        if len(hh_codes) == 0:
            return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}

        dates = ev["date_calcul"].to_numpy(dtype="datetime64[ns]")
        scores_ev = ev[score_col].to_numpy(dtype=float)
        order = np.argsort(dates, kind="stable")
        dates = dates[order]
        hh_codes = hh_codes[order]
        scores_ev = scores_ev[order]

        n_hh = int(hh_codes.max()) + 1
        cur_scores = np.full(n_hh, np.nan, dtype=float)
        ptr = 0
        n_ev = len(dates)
        rows_fast: list[dict[str, Any]] = []

        for i in range(0, len(period_points), batch_n):
            for week_end in period_points[i : i + batch_n]:
                week_np = np.datetime64(week_end.to_datetime64(), "ns")
                while ptr < n_ev and dates[ptr] <= week_np:
                    cur_scores[hh_codes[ptr]] = scores_ev[ptr]
                    ptr += 1

                active = cur_scores[~np.isnan(cur_scores)]
                if active.size == 0:
                    continue

                p10: float | None = None
                p25: float | None = None
                p75: float | None = None
                p90: float | None = None
                if include_percentiles:
                    p10 = round(float(np.percentile(active, 10)), 6)
                    p25 = round(float(np.percentile(active, 25)), 6)
                    p75 = round(float(np.percentile(active, 75)), 6)
                    p90 = round(float(np.percentile(active, 90)), 6)

                rows_fast.append(
                    {
                        "date_calcul": week_end,
                        "milieu": None,
                        "region": None,
                        "genre_cm": None,
                        "n_menages": int(active.size),
                        "score_mean": round(float(np.mean(active)), 6),
                        "score_median": round(float(np.median(active)), 6),
                        "score_std": round(float(np.std(active)), 6),
                        "score_min": round(float(np.min(active)), 6),
                        "score_max": round(float(np.max(active)), 6),
                        "score_p10": p10,
                        "score_p25": p25,
                        "score_p75": p75,
                        "score_p90": p90,
                    }
                )

        daily_stats = pd.DataFrame(rows_fast)
        if daily_stats.empty:
            return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}
        daily_stats["date_calcul"] = pd.to_datetime(daily_stats["date_calcul"], errors="coerce")
        daily_stats = daily_stats.sort_values(["date_calcul", "milieu", "region", "genre_cm"]).reset_index(drop=True)
        return {"daily_stats": daily_stats, "daily_menages": pd.DataFrame()}

    state: dict[Any, tuple[Any, ...]] = {}
    ev_records = list(ev.itertuples(index=False, name=None))
    col_idx = {c: i for i, c in enumerate(ev.columns)}
    date_idx = col_idx["date_calcul"]
    score_idx = col_idx[score_col]
    demo_idx = {d: col_idx[d] for d in demo_cols}

    ev_ptr = 0
    n_ev = len(ev_records)
    results: list[dict[str, Any]] = []

    def _stats(arr: np.ndarray) -> tuple[int, float, float, float, float, float, float | None, float | None, float | None, float | None]:
        p10: float | None = None
        p25: float | None = None
        p75: float | None = None
        p90: float | None = None
        if include_percentiles:
            p10 = round(float(np.percentile(arr, 10)), 6)
            p25 = round(float(np.percentile(arr, 25)), 6)
            p75 = round(float(np.percentile(arr, 75)), 6)
            p90 = round(float(np.percentile(arr, 90)), 6)
        return (
            len(arr),
            round(float(np.mean(arr)), 6),
            round(float(np.median(arr)), 6),
            round(float(np.std(arr)), 6),
            round(float(np.min(arr)), 6),
            round(float(np.max(arr)), 6),
            p10,
            p25,
            p75,
            p90,
        )

    for i in range(0, len(period_points), batch_n):
        for week_end in period_points[i : i + batch_n]:
            while ev_ptr < n_ev:
                row = ev_records[ev_ptr]
                if row[date_idx] <= week_end:
                    hh = row[0]
                    sv = row[score_idx]
                    if pd.isna(sv):
                        state.pop(hh, None)
                    else:
                        state[hh] = row
                    ev_ptr += 1
                else:
                    break

            if not state:
                continue

            rows = list(state.values())
            scores = np.array([r[score_idx] for r in rows], dtype=float)
            valid = ~np.isnan(scores)
            scores = scores[valid]
            if len(scores) == 0:
                continue

            n, mean, median, std, mn, mx, p10, p25, p75, p90 = _stats(scores)
            results.append(
                {
                    "date_calcul": week_end,
                    "milieu": None,
                    "region": None,
                    "genre_cm": None,
                    "n_menages": n,
                    "score_mean": mean,
                    "score_median": median,
                    "score_std": std,
                    "score_min": mn,
                    "score_max": mx,
                    "score_p10": p10,
                    "score_p25": p25,
                    "score_p75": p75,
                    "score_p90": p90,
                }
            )

            valid_rows = [r for r in rows if not pd.isna(r[score_idx])]
            if valid_rows and demo_idx:
                score_vals = np.array([float(r[score_idx]) for r in valid_rows], dtype=float)
                for dim, didx in demo_idx.items():
                    raw_pairs = [
                        (v, s)
                        for v, s in zip((r[didx] for r in valid_rows), score_vals, strict=False)
                        if (v is not None and not pd.isna(v) and (not isinstance(v, str) or v.strip() != ""))
                    ]
                    if not raw_pairs:
                        continue
                    dim_df = pd.DataFrame(raw_pairs, columns=["dim", "score"])
                    agg = (
                        dim_df.groupby("dim", observed=True)["score"]
                        .agg(
                            n_menages="count",
                            score_mean="mean",
                            score_median="median",
                            score_std="std",
                            score_min="min",
                            score_max="max",
                            score_p10=lambda x: x.quantile(0.10),
                            score_p25=lambda x: x.quantile(0.25),
                            score_p75=lambda x: x.quantile(0.75),
                            score_p90=lambda x: x.quantile(0.90),
                        )
                        .reset_index()
                    )
                    agg = agg[agg["n_menages"] >= 2]
                    for row_dim in agg.itertuples(index=False):
                        results.append(
                            {
                                "date_calcul": week_end,
                                "milieu": row_dim.dim if dim == "milieu" else None,
                                "region": row_dim.dim if dim == "region" else None,
                                "genre_cm": row_dim.dim if dim == "genre_cm" else None,
                                "n_menages": int(row_dim.n_menages),
                                "score_mean": round(float(row_dim.score_mean), 6),
                                "score_median": round(float(row_dim.score_median), 6),
                                "score_std": round(float(row_dim.score_std), 6) if not pd.isna(row_dim.score_std) else np.nan,
                                "score_min": round(float(row_dim.score_min), 6),
                                "score_max": round(float(row_dim.score_max), 6),
                                "score_p10": round(float(row_dim.score_p10), 6),
                                "score_p25": round(float(row_dim.score_p25), 6),
                                "score_p75": round(float(row_dim.score_p75), 6),
                                "score_p90": round(float(row_dim.score_p90), 6),
                            }
                        )

    daily_stats = pd.DataFrame(results)
    if daily_stats.empty:
        return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}
    daily_stats["date_calcul"] = pd.to_datetime(daily_stats["date_calcul"], errors="coerce")
    daily_stats = daily_stats.sort_values(["date_calcul", "milieu", "region", "genre_cm"]).reset_index(drop=True)
    return {"daily_stats": daily_stats, "daily_menages": pd.DataFrame()}


def build_monthly_eligibility_flows(df_master: pd.DataFrame) -> pd.DataFrame:
    if not {"menage_ano", "date_calcul", "eligible"}.issubset(df_master.columns):
        return pd.DataFrame()
    df = df_master[["menage_ano", "date_calcul", "eligible"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df["year_month"] = df["date_calcul"].dt.to_period("M")
    m = (
        df.sort_values("date_calcul")
        .groupby(["year_month", "menage_ano"], observed=True)["eligible"]
        .last()
        .reset_index()
    )
    m["eligible"] = m["eligible"].astype("boolean")
    m["prev_eligible"] = m.groupby("menage_ano", observed=True)["eligible"].shift(1).astype("boolean")
    m["eligible"] = m["eligible"].fillna(False)
    m["prev_eligible"] = m["prev_eligible"].fillna(False)
    m["became_eligible"] = m["eligible"] & (~m["prev_eligible"])
    m["became_ineligible"] = (~m["eligible"]) & m["prev_eligible"]
    counts = m.groupby(["year_month", "eligible"], observed=True).size().unstack(fill_value=0)
    if True not in counts.columns:
        counts[True] = 0
    if False not in counts.columns:
        counts[False] = 0
    trans = m.groupby("year_month", observed=True)[["became_eligible", "became_ineligible"]].sum()
    out = pd.DataFrame(
        {
            "date": counts.index.to_timestamp(),
            "unique_menages_eligible": counts[True].astype(int).values,
            "unique_menages_not_eligible": counts[False].astype(int).values,
            "menages_became_eligible": trans["became_eligible"].reindex(counts.index, fill_value=0).astype(int).values,
            "menages_became_ineligible": trans["became_ineligible"]
            .reindex(counts.index, fill_value=0)
            .astype(int)
            .values,
        }
    )
    out["n_transitions"] = out["menages_became_eligible"] + out["menages_became_ineligible"]
    return out.sort_values("date").reset_index(drop=True)


def build_monthly_beneficiaire_flows(df_beneficiaire: pd.DataFrame) -> pd.DataFrame:
    req = {"date_insert", "actif", "partner_id", "menage_ano"}
    if df_beneficiaire.empty or not req.issubset(df_beneficiaire.columns):
        return pd.DataFrame()
    df = df_beneficiaire[list(req)].copy()
    df["date_insert"] = pd.to_datetime(df["date_insert"], errors="coerce")
    df["year_month"] = df["date_insert"].dt.to_period("M")
    df["actif"] = df["actif"].astype("boolean")
    m = (
        df.sort_values("date_insert")
        .groupby(["year_month", "partner_id", "menage_ano"], observed=True)["actif"]
        .last()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for prog, sub in m.groupby("partner_id", observed=True):
        cur_elig: set[Any] = set()
        cur_not: set[Any] = set()
        for month in sorted(sub["year_month"].unique()):
            sm = sub[sub["year_month"] == month]
            became_e = 0
            became_i = 0
            for hid, actif in zip(sm["menage_ano"], sm["actif"]):
                was = hid in cur_elig
                now = bool(actif)
                if (not was) and now:
                    cur_not.discard(hid)
                    cur_elig.add(hid)
                    became_e += 1
                elif was and (not now):
                    cur_elig.discard(hid)
                    cur_not.add(hid)
                    became_i += 1
                elif (not was) and (not now):
                    cur_not.add(hid)
            rows.append(
                {
                    "date": month.to_timestamp(),
                    "programme": str(prog).upper(),
                    "unique_menages_eligible": len(cur_elig),
                    "unique_menages_not_eligible": len(cur_not),
                    "menages_became_eligible": became_e,
                    "menages_became_ineligible": became_i,
                }
            )
    return pd.DataFrame(rows).sort_values(["programme", "date"]).reset_index(drop=True)


def build_churn_timeline(df_master: pd.DataFrame) -> pd.DataFrame:
    if not {"eligible", "programme", "menage_ano", "date_calcul"}.issubset(df_master.columns):
        return pd.DataFrame()
    df = df_master[["menage_ano", "programme", "date_calcul", "eligible"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df["year_month"] = df["date_calcul"].dt.to_period("M")
    monthly = (
        df.sort_values(["menage_ano", "programme", "date_calcul"])
        .groupby(["programme", "year_month", "menage_ano"], observed=True)["eligible"]
        .last()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for prog, sub in monthly.groupby("programme", observed=True):
        prev: dict[Any, bool] = {}
        for month in sorted(sub["year_month"].unique()):
            sm = sub[sub["year_month"] == month]
            now = dict(zip(sm["menage_ano"], sm["eligible"]))
            pool_start = sum(1 for v in prev.values() if v)
            entries = 0
            exits = 0
            for hid, is_elig in now.items():
                prev_val = prev.get(hid)
                if prev_val is None:
                    pass
                elif (not prev_val) and bool(is_elig):
                    entries += 1
                elif bool(prev_val) and (not bool(is_elig)):
                    exits += 1
                prev[hid] = bool(is_elig)
            pool_end = sum(1 for v in prev.values() if v)
            denom = pool_start if pool_start > 0 else np.nan
            rows.append(
                {
                    "date": month.to_timestamp(),
                    "programme": prog,
                    "pool_start": pool_start,
                    "pool_end": pool_end,
                    "entries": entries,
                    "exits": exits,
                    "churn_rate": exits / denom,
                    "acquisition_rate": entries / denom,
                    "net_rate": (entries - exits) / denom,
                }
            )
    return pd.DataFrame(rows).sort_values(["programme", "date"]).reset_index(drop=True)


def build_reentry_analysis(df_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"eligible", "programme", "menage_ano", "date_calcul"}.issubset(df_master.columns):
        return pd.DataFrame(), pd.DataFrame()
    keys = ["menage_ano", "programme"]
    df = df_master[["menage_ano", "date_calcul", "eligible", "programme"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df = df.sort_values(keys + ["date_calcul"])
    df["eligible_bool"] = df["eligible"].fillna(False).astype(bool)
    # Match historical loop logic exactly:
    # count a re-entry when current is True and previous observed status is False.
    # The first row in each (menage_ano, programme) group is never counted.
    df["prev_eligible"] = df.groupby(keys, observed=True)["eligible_bool"].shift(1)
    df["reentry_event"] = df["eligible_bool"] & df["prev_eligible"].eq(False)

    grouped = df.groupby(keys, observed=True)
    detail = grouped.agg(
        n_reentries=("reentry_event", "sum"),
        all_eligible=("eligible_bool", "all"),
        ever_eligible=("eligible_bool", "any"),
    ).reset_index()
    detail["ever_lost"] = ~detail["all_eligible"]
    detail = detail.drop(columns=["all_eligible"])
    detail["n_reentries"] = detail["n_reentries"].astype(int)
    if detail.empty:
        return detail, pd.DataFrame()

    rows_summary: list[dict[str, Any]] = []
    for prog, grp in detail.groupby("programme", observed=True):
        total = grp["menage_ano"].nunique()
        n_always = grp[(grp["n_reentries"] == 0) & (~grp["ever_lost"])]["menage_ano"].nunique()
        n_never = grp[(grp["n_reentries"] == 0) & (~grp["ever_eligible"])]["menage_ano"].nunique()
        n_lost_no_return = grp[
            (grp["n_reentries"] == 0) & (grp["ever_lost"]) & (grp["ever_eligible"])
        ]["menage_ano"].nunique()

        rows_summary.append(
            {
                "programme": prog,
                "n_reentries": "toujours éligible",
                "n_menages": int(n_always),
                "pct_menages": round(n_always / total * 100, 2) if total else 0.0,
            }
        )
        rows_summary.append(
            {
                "programme": prog,
                "n_reentries": "jamais éligible",
                "n_menages": int(n_never),
                "pct_menages": round(n_never / total * 100, 2) if total else 0.0,
            }
        )
        rows_summary.append(
            {
                "programme": prog,
                "n_reentries": "perdu sans retour",
                "n_menages": int(n_lost_no_return),
                "pct_menages": round(n_lost_no_return / total * 100, 2) if total else 0.0,
            }
        )

        for n_r, sub in grp[grp["n_reentries"] > 0].groupby("n_reentries", observed=True):
            n_m = sub["menage_ano"].nunique()
            rows_summary.append(
                {
                    "programme": prog,
                    "n_reentries": str(int(n_r)),
                    "n_menages": int(n_m),
                    "pct_menages": round(n_m / total * 100, 2) if total else 0.0,
                }
            )

    summary = pd.DataFrame(rows_summary)
    order_map = {
        "toujours éligible": -3,
        "jamais éligible": -2,
        "perdu sans retour": -1,
    }
    summary["_order"] = summary["n_reentries"].map(order_map).fillna(
        summary["n_reentries"].astype(str).map(lambda x: int(x) if x.isdigit() else 999)
    )
    summary = summary.sort_values(["programme", "_order"]).drop(columns=["_order"]).reset_index(drop=True)
    summary["n_reentries"] = summary["n_reentries"].astype(str)
    return detail, summary


def build_near_threshold_timeseries(
    df_master: pd.DataFrame,
    score_col: str = "score_final",
    bands: tuple = (0.10, 0.25, 0.50),
    timeseries_freq: str = "W-MON",
    batch_size: int = 0,
) -> pd.DataFrame:
    required = ["menage_ano", "date_calcul", score_col, "programme", "dist_threshold"]
    if not set(required).issubset(df_master.columns):
        return pd.DataFrame()

    keep = ["menage_ano", "date_calcul", score_col, "type_demande", "programme", "dist_threshold"]
    if "type_demande" not in df_master.columns:
        df_master = df_master.copy()
        df_master["type_demande"] = ""
    ev = df_master[keep].copy()
    prog_norm = ev["programme"].astype(str).str.strip().str.upper()
    ev = ev[~prog_norm.isin({"NON CLASSIFIE", "NON CLASSIFIÉ"})]
    if not pd.api.types.is_datetime64_any_dtype(ev["date_calcul"]):
        ev["date_calcul"] = pd.to_datetime(ev["date_calcul"], errors="coerce", cache=False)
    ev[score_col] = pd.to_numeric(ev[score_col], errors="coerce")
    ev["dist_threshold"] = pd.to_numeric(ev["dist_threshold"], errors="coerce")
    ev["type_norm"] = ev["type_demande"].astype(str).str.strip().str.lower()
    ev.loc[ev["type_norm"] == "radiation", score_col] = np.nan
    ev.loc[ev["type_norm"] == "radiation", "dist_threshold"] = np.nan
    ev = (
        ev.drop(columns=["type_demande", "type_norm"])
        .dropna(subset=["date_calcul"])
        .sort_values(["date_calcul", "menage_ano", "programme"])
        .reset_index(drop=True)
    )
    if ev.empty:
        return pd.DataFrame()

    t0 = ev["date_calcul"].min()
    tmax = ev["date_calcul"].max()
    freq_norm = (timeseries_freq or "W-MON").strip().upper()
    if freq_norm in {"D", "DAILY"}:
        period_points = pd.date_range(t0.normalize(), tmax.normalize(), freq="D")
    else:
        period_points = pd.date_range(t0.to_period("W").start_time, tmax.to_period("W").start_time, freq="W-MON")
    if len(period_points) == 0:
        return pd.DataFrame()
    batch_n = int(batch_size) if int(batch_size) > 0 else len(period_points)

    # Fast path for dashboard profile shape (default bands only).
    if tuple(bands) == (0.10, 0.25, 0.50):
        pair_index = pd.MultiIndex.from_frame(ev[["menage_ano", "programme"]])
        pair_codes, pair_uniques = pd.factorize(pair_index, sort=False)
        if len(pair_codes) == 0:
            return pd.DataFrame()
        prog_labels = np.array([p[1] for p in pair_uniques], dtype=object)

        dates = ev["date_calcul"].to_numpy(dtype="datetime64[ns]")
        score_vals = ev[score_col].to_numpy(dtype=float)
        dist_vals = ev["dist_threshold"].to_numpy(dtype=float)
        order = np.argsort(dates, kind="stable")
        dates = dates[order]
        pair_codes = pair_codes[order]
        score_vals = score_vals[order]
        dist_vals = dist_vals[order]

        n_pairs = int(pair_codes.max()) + 1
        cur_scores = np.full(n_pairs, np.nan, dtype=float)
        cur_dists = np.full(n_pairs, np.nan, dtype=float)
        ptr = 0
        n_ev = len(dates)
        out_rows: list[dict[str, Any]] = []
        unique_progs = np.unique(prog_labels)

        for i in range(0, len(period_points), batch_n):
            for week_end in period_points[i : i + batch_n]:
                week_np = np.datetime64(week_end.to_datetime64(), "ns")
                while ptr < n_ev and dates[ptr] <= week_np:
                    idx = pair_codes[ptr]
                    sv = score_vals[ptr]
                    if np.isnan(sv):
                        cur_scores[idx] = np.nan
                        cur_dists[idx] = np.nan
                    else:
                        cur_scores[idx] = sv
                        cur_dists[idx] = dist_vals[ptr]
                    ptr += 1

                valid = ~np.isnan(cur_scores) & ~np.isnan(cur_dists)
                if not np.any(valid):
                    continue

                for prog in unique_progs:
                    prog_mask = valid & (prog_labels == prog)
                    if not np.any(prog_mask):
                        continue
                    scores = cur_scores[prog_mask]
                    dists = cur_dists[prog_mask]
                    row_out: dict[str, Any] = {"week": week_end.date(), "programme": prog, "n_total": int(scores.size)}

                    for b in bands:
                        suffix = f"{int(b * 100):03d}"
                        mask_net = np.abs(dists) <= b
                        mask_neg = (dists >= -b) & (dists < 0)
                        mask_pos = (dists > 0) & (dists <= b)

                        near_s_net = scores[mask_net]
                        row_out[f"n_near_{suffix}"] = int(mask_net.sum())
                        row_out[f"mean_score_{suffix}"] = (
                            round(float(np.mean(near_s_net)), 6) if near_s_net.size > 0 else None
                        )
                        row_out[f"median_score_{suffix}"] = (
                            round(float(np.median(near_s_net)), 6) if near_s_net.size > 0 else None
                        )

                        near_s_neg = scores[mask_neg]
                        row_out[f"n_near_{suffix}_neg"] = int(mask_neg.sum())
                        row_out[f"mean_score_{suffix}_neg"] = (
                            round(float(np.mean(near_s_neg)), 6) if near_s_neg.size > 0 else None
                        )
                        row_out[f"median_score_{suffix}_neg"] = (
                            round(float(np.median(near_s_neg)), 6) if near_s_neg.size > 0 else None
                        )

                        near_s_pos = scores[mask_pos]
                        row_out[f"n_near_{suffix}_pos"] = int(mask_pos.sum())
                        row_out[f"mean_score_{suffix}_pos"] = (
                            round(float(np.mean(near_s_pos)), 6) if near_s_pos.size > 0 else None
                        )
                        row_out[f"median_score_{suffix}_pos"] = (
                            round(float(np.median(near_s_pos)), 6) if near_s_pos.size > 0 else None
                        )

                    out_rows.append(row_out)

        if not out_rows:
            return pd.DataFrame()
        out = pd.DataFrame(out_rows)
        out["week"] = pd.to_datetime(out["week"], errors="coerce")
        return out.sort_values(["week", "programme"]).reset_index(drop=True)

    col_idx = {c: i for i, c in enumerate(ev.columns)}
    date_idx = col_idx["date_calcul"]
    score_idx = col_idx[score_col]
    dist_idx = col_idx["dist_threshold"]
    prog_idx = col_idx["programme"]
    hh_idx = 0

    ev_records = list(ev.itertuples(index=False, name=None))
    n_ev = len(ev_records)
    state: dict[tuple[Any, Any], tuple[float, float]] = {}
    ev_ptr = 0
    results: list[dict[str, Any]] = []

    for i in range(0, len(period_points), batch_n):
        for week_end in period_points[i : i + batch_n]:
            while ev_ptr < n_ev:
                row = ev_records[ev_ptr]
                if row[date_idx] <= week_end:
                    key = (row[hh_idx], row[prog_idx])
                    sv = row[score_idx]
                    if pd.isna(sv):
                        state.pop(key, None)
                    else:
                        dv = row[dist_idx]
                        state[key] = (float(sv), float(dv) if not pd.isna(dv) else np.nan)
                    ev_ptr += 1
                else:
                    break

            if not state:
                continue

            prog_groups: dict[Any, list[tuple[float, float]]] = {}
            for (_, prog), pair in state.items():
                prog_groups.setdefault(prog, []).append(pair)

            for prog, pairs in prog_groups.items():
                scores = np.array([p[0] for p in pairs], dtype=float)
                dists = np.array([p[1] for p in pairs], dtype=float)
                valid = ~np.isnan(scores) & ~np.isnan(dists)
                scores = scores[valid]
                dists = dists[valid]
                if len(scores) == 0:
                    continue

                row_out: dict[str, Any] = {"week": week_end.date(), "programme": prog, "n_total": len(scores)}
                for b in bands:
                    suffix = f"{int(b * 100):03d}"
                    mask_net = np.abs(dists) <= b
                    mask_neg = (dists >= -b) & (dists < 0)
                    mask_pos = (dists > 0) & (dists <= b)

                    near_s_net = scores[mask_net]
                    row_out[f"n_near_{suffix}"] = int(mask_net.sum())
                    row_out[f"mean_score_{suffix}"] = (
                        round(float(np.mean(near_s_net)), 6) if len(near_s_net) > 0 else None
                    )
                    row_out[f"median_score_{suffix}"] = (
                        round(float(np.median(near_s_net)), 6) if len(near_s_net) > 0 else None
                    )

                    near_s_neg = scores[mask_neg]
                    row_out[f"n_near_{suffix}_neg"] = int(mask_neg.sum())
                    row_out[f"mean_score_{suffix}_neg"] = (
                        round(float(np.mean(near_s_neg)), 6) if len(near_s_neg) > 0 else None
                    )
                    row_out[f"median_score_{suffix}_neg"] = (
                        round(float(np.median(near_s_neg)), 6) if len(near_s_neg) > 0 else None
                    )

                    near_s_pos = scores[mask_pos]
                    row_out[f"n_near_{suffix}_pos"] = int(mask_pos.sum())
                    row_out[f"mean_score_{suffix}_pos"] = (
                        round(float(np.mean(near_s_pos)), 6) if len(near_s_pos) > 0 else None
                    )
                    row_out[f"median_score_{suffix}_pos"] = (
                        round(float(np.median(near_s_pos)), 6) if len(near_s_pos) > 0 else None
                    )
                results.append(row_out)

    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    out["week"] = pd.to_datetime(out["week"], errors="coerce")
    return out.sort_values(["week", "programme"]).reset_index(drop=True)
