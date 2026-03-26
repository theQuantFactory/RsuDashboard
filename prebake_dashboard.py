"""
prebake_dashboard.py
────────────────────
Run this script ONCE after every pipeline run to pre-compute all dashboard
aggregations and save them to a single small cache file.

Usage:
    python prebake_dashboard.py
    python prebake_dashboard.py --snap-dir path/to/snapshots/csv
    python prebake_dashboard.py --out path/to/dashboard_cache.pkl

The dashboard then loads this file in milliseconds instead of aggregating
3M+ row frames on every startup.
"""

import argparse
import datetime
import os
import pickle
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed. Run: pip install duckdb")
    sys.exit(1)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pre-bake RSU dashboard cache")
parser.add_argument("--snap-dir", default="snapshots/csv",
                    help="Directory containing .parquet snapshot files")
parser.add_argument("--out", default="snapshots/dashboard_cache.pkl",
                    help="Output path for the dashboard cache file")
args = parser.parse_args()

APP_DIR = Path(__file__).resolve().parent


def _resolve_cli_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    # Keep cwd-relative paths when they already make sense.
    if p.exists() or p.parent.exists():
        return p
    # Otherwise resolve relative to apps/rsu_dashboard.
    return APP_DIR / p


SNAP_DIR = _resolve_cli_path(args.snap_dir)
OUT_PATH = _resolve_cli_path(args.out)

if not SNAP_DIR.exists():
    print(f"ERROR: Snapshot directory not found: {SNAP_DIR}")
    sys.exit(1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parquet(name: str):
    p = SNAP_DIR / f"{name}.parquet"
    return str(p) if p.exists() else None

def qry(con, sql: str):
    try:
        return con.execute(sql).df()
    except Exception as e:
        print(f"  WARNING: query failed — {e}")
        import pandas as pd
        return pd.DataFrame()

def section(title: str):
    print(f"\n  {'─'*50}")
    print(f"  {title}")
    print(f"  {'─'*50}")

# ── Connect DuckDB ────────────────────────────────────────────────────────────
con = duckdb.connect()
import pandas as pd

print("\n╔══════════════════════════════════════════════╗")
print("║   RSU Dashboard — Pre-bake Cache Builder     ║")
print("╚══════════════════════════════════════════════╝")
print(f"\n  Source : {SNAP_DIR.resolve()}")
print(f"  Output : {OUT_PATH.resolve()}")

available = {p.stem for p in SNAP_DIR.glob("*.parquet")}
print(f"\n  Found {len(available)} snapshot(s): {', '.join(sorted(available))}")

a = {}  # the cache dictionary

# ── Snapshot file metadata ────────────────────────────────────────────────────
section("File metadata")

file_labels = {
    "raw_menage":        "Données démographiques des ménages",
    "raw_scores":        "Événements de score (après nettoyage)",
    "master_events":     "Événements enrichis (score + ménage + programme)",
    "delta_frame":       "Variations de score entre événements consécutifs (ΔISE)",
    "menage_timeline":   "Résumé par ménage et par programme",
    "menage_trajectory": "Trajectoire complète de chaque ménage",
    "programme_AMOT":    "Événements du programme AMOT",
    "programme_ASD":     "Événements du programme ASD",
    "programme_AMOA":    "Événements du programme AMOA",
    "raw_programmes":    "Listes d'affiliation aux programmes",
}

rows = []
for p in sorted(SNAP_DIR.glob("*.parquet")):
    n_rows = qry(con, f"SELECT COUNT(*) as n FROM read_parquet('{p}')").iloc[0, 0]
    rows.append({
        "Fichier":      p.stem,
        "Description":  file_labels.get(p.stem, p.stem),
        "Lignes":       int(n_rows),
        "Taille (MB)":  round(p.stat().st_size / 1e6, 1),
        "Mis à jour":   datetime.datetime.fromtimestamp(
                            p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
    })
    print(f"  ✓ {p.stem:<30} {int(n_rows):>10,} rows   {round(p.stat().st_size/1e6,1):>6.1f} MB")

a["files"] = pd.DataFrame(rows)
a["last_update"] = max(
    datetime.datetime.fromtimestamp(p.stat().st_mtime)
    for p in SNAP_DIR.glob("*.parquet")
)

# ── Data Quality metrics ──────────────────────────────────────────────────────
section("Data quality metrics")

rsp  = parquet("raw_scores")
rmp  = parquet("raw_menage")
mp   = parquet("master_events")
rpp  = parquet("raw_programmes")

# Raw adhesion counts per programme (from the raw_programmes parquet, not the derived frames)
if rpp:
    for _prog in ["AMOT", "ASD", "AMOA"]:
        _n = qry(con, f"""
            SELECT COUNT(*) AS n FROM read_parquet('{rpp}')
            WHERE UPPER(programme) = '{_prog}'
        """)
        a[f"prog_raw_rows_{_prog}"] = int(_n.iloc[0, 0]) if not _n.empty else 0

if rsp:
    # Total raw score rows
    a["qc_raw_score_rows"] = int(qry(con, f"""
        SELECT COUNT(*) FROM read_parquet('{rsp}')
    """).iloc[0, 0])

    # Duplicate score_id_ano
    dup_s = qry(con, f"""
        SELECT COUNT(*) - COUNT(DISTINCT score_id_ano) AS n_dupes
        FROM read_parquet('{rsp}')
        WHERE score_id_ano IS NOT NULL
    """)
    a["qc_score_dupes"] = int(dup_s["n_dupes"].iloc[0]) if not dup_s.empty else 0

    # Score period coverage
    period = qry(con, f"""
        SELECT MIN(date_calcul) AS date_min, MAX(date_calcul) AS date_max
        FROM read_parquet('{rsp}')
        WHERE date_calcul IS NOT NULL
    """)
    if not period.empty:
        a["qc_score_date_min"] = pd.to_datetime(period["date_min"].iloc[0])
        a["qc_score_date_max"] = pd.to_datetime(period["date_max"].iloc[0])

    # Aberrant scores: rows in raw_scores that didn't make it to master_events
    if mp:
        ab = qry(con, f"""
            SELECT
                (SELECT COUNT(*) FROM read_parquet('{rsp}')) -
                (SELECT COUNT(*) FROM read_parquet('{mp}')) AS n_aberrant
        """)
        a["qc_aberrant_scores"] = max(int(ab.iloc[0, 0]) if not ab.empty else 0, 0)
    else:
        a["qc_aberrant_scores"] = 0

    print(f"  qc_raw_score_rows  = {a['qc_raw_score_rows']:,}")
    print(f"  qc_score_dupes     = {a['qc_score_dupes']:,}")
    print(f"  qc_aberrant_scores = {a['qc_aberrant_scores']:,}")
    if "qc_score_date_min" in a:
        print(f"  score period       = {a['qc_score_date_min'].date()} → {a['qc_score_date_max'].date()}")
else:
    a["qc_raw_score_rows"] = 0
    a["qc_score_dupes"]    = 0
    a["qc_aberrant_scores"] = 0

if rmp:
    # Total raw menage rows
    a["qc_raw_menage_rows"] = int(qry(con, f"""
        SELECT COUNT(*) FROM read_parquet('{rmp}')
    """).iloc[0, 0])

    # Duplicate menage_ano
    dup_m = qry(con, f"""
        SELECT COUNT(*) - COUNT(DISTINCT menage_ano) AS n_dupes
        FROM read_parquet('{rmp}')
        WHERE menage_ano IS NOT NULL
    """)
    a["qc_menage_dupes"] = int(dup_m["n_dupes"].iloc[0]) if not dup_m.empty else 0

    # Menages in demographics but with no score events at all
    if mp:
        no_score = qry(con, f"""
            SELECT COUNT(*) AS n
            FROM read_parquet('{rmp}') m
            LEFT JOIN (
                SELECT DISTINCT menage_ano FROM read_parquet('{mp}')
            ) s USING (menage_ano)
            WHERE s.menage_ano IS NULL
        """)
        a["qc_menages_no_score"] = int(no_score["n"].iloc[0]) if not no_score.empty else 0
    else:
        a["qc_menages_no_score"] = 0

    # Missing region
    null_region = qry(con, f"""
        SELECT COUNT(*) AS n FROM read_parquet('{rmp}')
        WHERE region IS NULL OR TRIM(CAST(region AS VARCHAR)) = ''
    """)
    a["qc_null_region"] = int(null_region["n"].iloc[0]) if not null_region.empty else 0

    # Missing milieu
    null_milieu = qry(con, f"""
        SELECT COUNT(*) AS n FROM read_parquet('{rmp}')
        WHERE milieu IS NULL OR TRIM(CAST(milieu AS VARCHAR)) = ''
    """)
    a["qc_null_milieu"] = int(null_milieu["n"].iloc[0]) if not null_milieu.empty else 0

    print(f"  qc_raw_menage_rows = {a['qc_raw_menage_rows']:,}")
    print(f"  qc_menage_dupes    = {a['qc_menage_dupes']:,}")
    print(f"  qc_menages_no_score= {a['qc_menages_no_score']:,}")
    print(f"  qc_null_region     = {a['qc_null_region']:,}")
    print(f"  qc_null_milieu     = {a['qc_null_milieu']:,}")
else:
    a["qc_raw_menage_rows"]  = 0
    a["qc_menage_dupes"]     = 0
    a["qc_menages_no_score"] = 0
    a["qc_null_region"]      = 0
    a["qc_null_milieu"]      = 0

# ── QC summary (written by the ETL pipeline) ─────────────────────────────────
section("Data quality summary")

if parquet("qc_summary"):
    a["qc_summary"] = qry(con, f"""
        SELECT * FROM read_parquet('{parquet("qc_summary")}')
    """)
    print(f"  qc_summary = {len(a['qc_summary'])} metrics")
    for _, r in a["qc_summary"].iterrows():
        dropped = r.get("dropped", 0)
        pct     = r.get("pct_dropped")
        pct_str = f"  ({pct}%)" if pct is not None else ""
        print(f"    {r['metric']:<35} before={r['before']}  after={r['after']}  dropped={dropped}{pct_str}")
else:
    a["qc_summary"] = pd.DataFrame()
    print("  WARNING: qc_summary.parquet not found — re-run the ETL pipeline to generate it.")

# ── Volume KPIs ───────────────────────────────────────────────────────────────
section("Volume KPIs")

if rmp:
    a["n_menages"] = int(qry(con, f"""
        SELECT COUNT(*) FROM read_parquet('{rmp}')
    """).iloc[0, 0])
    print(f"  n_menages       = {a['n_menages']:,}")
else:
    a["n_menages"] = 0

if rsp:
    a["n_score_events"] = int(qry(con, f"""
        SELECT COUNT(*) FROM read_parquet('{rsp}')
    """).iloc[0, 0])
    print(f"  n_score_events  = {a['n_score_events']:,}")
else:
    a["n_score_events"] = 0

if parquet("delta_frame"):
    a["n_transitions"] = int(qry(con, f"""
        SELECT COUNT(*) FROM read_parquet('{parquet("delta_frame")}')
    """).iloc[0, 0])
    print(f"  n_transitions   = {a['n_transitions']:,}")
else:
    a["n_transitions"] = 0

if mp:
    a["n_menages_uniq"] = int(qry(con, f"""
        SELECT COUNT(DISTINCT menage_ano) FROM read_parquet('{mp}')
    """).iloc[0, 0])
    print(f"  n_menages_uniq  = {a['n_menages_uniq']:,}")
else:
    a["n_menages_uniq"] = 0

# ── Trajectory KPIs ───────────────────────────────────────────────────────────
section("Trajectory KPIs")

if parquet("menage_trajectory"):
    traj_path = parquet("menage_trajectory")
    cols = qry(con, f"DESCRIBE SELECT * FROM read_parquet('{traj_path}')")[
        "column_name"].tolist()

    # Radiés définitivement: radiated AND last event is still a radiation
    # (excludes households who were radiated but later re-inscribed)
    if "est_radie" in cols and "statut_actuel" in cols:
        rad_final = qry(con, f"""
            SELECT COUNT(*) AS n
            FROM read_parquet('{traj_path}')
            WHERE est_radie = TRUE
              AND statut_actuel = 'radiation'
        """)
        a["n_radiated"] = int(rad_final["n"].iloc[0]) if not rad_final.empty else 0
    elif "est_radie" in cols:
        # Fallback if statut_actuel column is absent
        a["n_radiated"] = int(qry(con, f"""
            SELECT COALESCE(SUM(CAST(est_radie AS INT)), 0)
            FROM read_parquet('{traj_path}')
        """).iloc[0, 0])
    else:
        a["n_radiated"] = 0

    print(f"  n_radiated (définitif) = {a['n_radiated']:,}")

    if "statut_actuel" in cols:
        status_labels = {
            "stable_eligible": "Stable — éligible",
            "stable_excluded": "Stable — exclu",
            "gained":          "A gagné l'éligibilité",
            "lost":            "A perdu l'éligibilité",
            "fluctuating":     "Fluctuant",
        }
        ts = qry(con, f"""
            SELECT statut_actuel AS code, COUNT(*) AS "Ménages"
            FROM read_parquet('{traj_path}')
            GROUP BY statut_actuel
            ORDER BY "Ménages" DESC
        """)
        ts["Statut"] = ts["code"].map(status_labels).fillna(ts["code"])
        a["traj_status"] = ts
        print(f"  traj_status     = {len(ts)} trajectory categories")
else:
    a["n_radiated"] = 0

# ── Programme breakdown (excluding Non classifié) ─────────────────────────────
section("Programme breakdown")

PROG_FULL = {
    "AMOT":          "AMO Tadamon (AMOT)",
    "ASD":           "ASD — Aide Sociale Directe",
    "AMOA":          "AMOA — Aide au Logement",
    "Non classifié": "Non classifié — Sans programme",
}

if mp:
    # Exclude Non classifié from the programme bar chart data
    pm = qry(con, f"""
        SELECT programme,
               COUNT(DISTINCT menage_ano) AS "Ménages uniques"
        FROM read_parquet('{mp}')
        WHERE programme != 'Non classifié'
        GROUP BY programme
        ORDER BY "Ménages uniques" DESC
    """)
    pm["Nom complet"] = pm["programme"].map(PROG_FULL).fillna(pm["programme"])
    a["prog_menages"] = pm

    # Count Non classifié separately for the status card
    nc = qry(con, f"""
        SELECT COUNT(DISTINCT menage_ano) AS n
        FROM read_parquet('{mp}')
        WHERE programme = 'Non classifié'
    """)
    a["n_non_classifie"] = int(nc["n"].iloc[0]) if not nc.empty else 0

    print(f"  prog_menages (excl. Non classifié) = {len(pm)} programmes")
    for _, r in pm.iterrows():
        print(f"    {r['programme']:<20} {int(r['Ménages uniques']):>8,} ménages")
    print(f"  n_non_classifie = {a['n_non_classifie']:,}")

# ── Demographics ──────────────────────────────────────────────────────────────
section("Demographics")

if rmp:
    a["milieu"] = qry(con, f"""
        SELECT milieu AS "Milieu", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}') GROUP BY milieu ORDER BY "Ménages" DESC
    """)
    a["genre"] = qry(con, f"""
        SELECT genre_cm AS "Genre", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}') GROUP BY genre_cm ORDER BY "Ménages" DESC
    """)
    a["matrimonial"] = qry(con, f"""
        SELECT etat_matrimonial_cm AS "État", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}')
        GROUP BY etat_matrimonial_cm ORDER BY "Ménages" DESC
    """)
    a["region_dem"] = qry(con, f"""
        SELECT region AS "Région", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}') GROUP BY region ORDER BY "Ménages" ASC
    """)
    a["type_menage"] = qry(con, f"""
        SELECT type_menage AS "Type", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}') GROUP BY type_menage ORDER BY "Ménages" DESC
    """)
    a["taille"] = qry(con, f"""
        SELECT taille_menage AS "Personnes", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}')
        WHERE taille_menage <= 12
        GROUP BY taille_menage ORDER BY taille_menage ASC
    """)
    a["top_provinces"] = qry(con, f"""
        SELECT province AS "Province", COUNT(*) AS "Ménages"
        FROM read_parquet('{rmp}')
        GROUP BY province ORDER BY "Ménages" DESC LIMIT 20
    """)
    a["top_provinces"] = a["top_provinces"].sort_values("Ménages")

    print(f"  milieu        = {len(a['milieu'])} categories")
    print(f"  genre         = {len(a['genre'])} categories")
    print(f"  regions       = {len(a['region_dem'])} regions")
    print(f"  top_provinces = {len(a['top_provinces'])} provinces")

# ── Score statistics ──────────────────────────────────────────────────────────
section("Score statistics")

if mp:
    a["score_stats"] = qry(con, f"""
        SELECT
            programme                       AS "Programme",
            COUNT(*)                        AS "Événements",
            ROUND(AVG(score_final), 4)      AS "Moyenne",
            ROUND(MEDIAN(score_final), 4)   AS "Médiane",
            ROUND(STDDEV(score_final), 4)   AS "Écart-type",
            ROUND(MIN(score_final), 4)      AS "Min",
            ROUND(MAX(score_final), 4)      AS "Max"
        FROM read_parquet('{mp}')
        GROUP BY programme ORDER BY programme
    """)
    print(f"  score_stats     = {len(a['score_stats'])} rows")

    a["score_hist"] = qry(con, f"""
        SELECT programme, ROUND(score_final * 20) / 20 AS bin, COUNT(*) AS n
        FROM read_parquet('{mp}')
        WHERE score_final IS NOT NULL
        GROUP BY programme, bin ORDER BY programme, bin
    """)
    print(f"  score_hist      = {len(a['score_hist'])} histogram bins")

    a["n_near_10"] = int(qry(con, f"""
        SELECT COALESCE(SUM(CAST("near_0.10" AS INT)), 0) FROM read_parquet('{mp}')
    """).iloc[0, 0])
    a["n_near_25"] = int(qry(con, f"""
        SELECT COALESCE(SUM(CAST("near_0.25" AS INT)), 0) FROM read_parquet('{mp}')
    """).iloc[0, 0])
    print(f"  n_near_10       = {a['n_near_10']:,}")
    print(f"  n_near_25       = {a['n_near_25']:,}")

    elig_result = qry(con, f"""
        SELECT SUM(CAST(eligible AS INT)) AS n_eligible, COUNT(*) AS n_total
        FROM read_parquet('{mp}')
    """)
    n_el = int(elig_result["n_eligible"].iloc[0])
    n_to = int(elig_result["n_total"].iloc[0])
    a["n_eligible"]     = n_el
    a["n_not_eligible"] = n_to - n_el
    a["pct_eligible"]   = round(n_el / n_to * 100, 1) if n_to else 0
    print(f"  n_eligible      = {a['n_eligible']:,}  ({a['pct_eligible']}%)")

    a["elig_by_prog"] = qry(con, f"""
        SELECT programme,
               SUM(CAST(eligible AS INT)) AS eligible,
               COUNT(*) AS total,
               ROUND(100.0 * SUM(CAST(eligible AS INT)) / COUNT(*), 1) AS "% éligible"
        FROM read_parquet('{mp}')
        GROUP BY programme ORDER BY programme
    """)
    a["elig_by_region"] = qry(con, f"""
        SELECT region AS "Région",
               ROUND(100.0 * AVG(CAST(eligible AS INT)), 1) AS "% éligible"
        FROM read_parquet('{mp}')
        WHERE region IS NOT NULL
        GROUP BY region ORDER BY "% éligible" ASC
    """)
    a["elig_by_milieu"] = qry(con, f"""
        SELECT programme AS "Programme", milieu AS "Milieu",
               ROUND(100.0 * AVG(CAST(eligible AS INT)), 1) AS "% éligible"
        FROM read_parquet('{mp}')
        WHERE milieu IS NOT NULL
        GROUP BY programme, milieu ORDER BY programme, milieu
    """)
    a["type_demande"] = qry(con, f"""
        SELECT type_demande AS "Type", COUNT(*) AS "Événements"
        FROM read_parquet('{mp}')
        GROUP BY type_demande ORDER BY "Événements" DESC
    """)
    print(f"  type_demande    = {len(a['type_demande'])} event types")

# ── ΔISE statistics ───────────────────────────────────────────────────────────
section("ΔISE statistics")

if parquet("delta_frame"):
    dp = parquet("delta_frame")

    delta_global = qry(con, f"""
        SELECT
            ROUND(AVG(delta_ISE), 4)                       AS mean,
            ROUND(STDDEV(delta_ISE), 4)                    AS std,
            ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.9), 4)  AS p90,
            ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.99), 4) AS p99
        FROM read_parquet('{dp}')
        WHERE delta_ISE IS NOT NULL
    """)
    a["delta_mean"] = float(delta_global["mean"].iloc[0])
    a["delta_std"]  = float(delta_global["std"].iloc[0])
    a["delta_p90"]  = float(delta_global["p90"].iloc[0])
    a["delta_p99"]  = float(delta_global["p99"].iloc[0])
    print(f"  delta_mean = {a['delta_mean']}  std = {a['delta_std']}  p90 = {a['delta_p90']}")

    a["vol_prog"] = qry(con, f"""
        SELECT programme, COUNT(*) AS n,
               ROUND(AVG(delta_ISE), 4)                       AS mean_ΔISE,
               ROUND(STDDEV(delta_ISE), 4)                    AS "sigma_ΔISE",
               ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.9), 4)  AS p90
        FROM read_parquet('{dp}')
        WHERE delta_ISE IS NOT NULL
        GROUP BY programme ORDER BY programme
    """)
    a["vol_reg"] = qry(con, f"""
        SELECT programme, region,
               ROUND(AVG(delta_ISE), 4)    AS mean_ΔISE,
               ROUND(STDDEV(delta_ISE), 4) AS "sigma_ΔISE"
        FROM read_parquet('{dp}')
        WHERE delta_ISE IS NOT NULL AND region IS NOT NULL
        GROUP BY programme, region ORDER BY programme, region
    """)
    if not a["vol_reg"].empty:
        a["vol_heatmap"] = a["vol_reg"].pivot(
            index="region", columns="programme", values="mean_ΔISE")

    a["delta_box"] = qry(con, f"""
        SELECT programme,
               ROUND(QUANTILE_CONT(delta_ISE, 0.10), 4) AS p10,
               ROUND(QUANTILE_CONT(delta_ISE, 0.25), 4) AS q1,
               ROUND(MEDIAN(delta_ISE), 4)               AS median,
               ROUND(QUANTILE_CONT(delta_ISE, 0.75), 4) AS q3,
               ROUND(QUANTILE_CONT(delta_ISE, 0.90), 4) AS p90,
               ROUND(AVG(delta_ISE), 4)                  AS mean
        FROM read_parquet('{dp}')
        WHERE delta_ISE IS NOT NULL
        GROUP BY programme
    """)
    print(f"  vol_prog    = {len(a['vol_prog'])} rows")
    print(f"  delta_box   = {len(a['delta_box'])} rows")

    # ── Delta distribué aggregations ──────────────────────────────────────────
    # delta_distribue = delta_ISE / days_between, only where days_between > 90.
    # The filter is already baked into the column (NaN for pairs <= 90 days).
    # Short gaps (≤ 90 days) are excluded because the window is too narrow to
    # attribute the change to a coherent period; long gaps are normalised by
    # duration to make them comparable across pairs with different intervals.
    a["dd_global"] = qry(con, f"""
        SELECT
            COUNT(*)                                                AS n_total,
            SUM(CASE WHEN delta_distribue IS NOT NULL THEN 1 END)  AS n_valid,
            SUM(CASE WHEN days_between <= 90 THEN 1 ELSE 0 END)    AS n_short_gap_excluded,
            ROUND(AVG(delta_distribue),   6)  AS mean,
            ROUND(MEDIAN(delta_distribue),6)  AS median,
            ROUND(STDDEV(delta_distribue),6)  AS std,
            ROUND(QUANTILE_CONT(delta_distribue, 0.10), 6) AS p10,
            ROUND(QUANTILE_CONT(delta_distribue, 0.25), 6) AS q1,
            ROUND(QUANTILE_CONT(delta_distribue, 0.75), 6) AS q3,
            ROUND(QUANTILE_CONT(delta_distribue, 0.90), 6) AS p90,
            ROUND(MIN(delta_distribue), 6) AS min_val,
            ROUND(MAX(delta_distribue), 6) AS max_val
        FROM read_parquet('{dp}')
        WHERE delta_ISE IS NOT NULL
    """)

    a["dd_by_programme"] = qry(con, f"""
        SELECT programme,
            COUNT(*)                                                AS n_total,
            SUM(CASE WHEN delta_distribue IS NOT NULL THEN 1 END)  AS n_valid,
            ROUND(AVG(delta_distribue),   6)  AS mean,
            ROUND(MEDIAN(delta_distribue),6)  AS median,
            ROUND(STDDEV(delta_distribue),6)  AS std,
            ROUND(QUANTILE_CONT(delta_distribue, 0.25), 6) AS q1,
            ROUND(QUANTILE_CONT(delta_distribue, 0.75), 6) AS q3,
            ROUND(QUANTILE_CONT(delta_distribue, 0.10), 6) AS p10,
            ROUND(QUANTILE_CONT(delta_distribue, 0.90), 6) AS p90
        FROM read_parquet('{dp}')
        WHERE delta_ISE IS NOT NULL
        GROUP BY programme ORDER BY programme
    """)

    a["dd_by_milieu"] = qry(con, f"""
        SELECT milieu,
            COUNT(*)                                                AS n_valid,
            ROUND(AVG(delta_distribue),   6)  AS mean,
            ROUND(STDDEV(delta_distribue),6)  AS std,
            ROUND(MEDIAN(delta_distribue),6)  AS median
        FROM read_parquet('{dp}')
        WHERE delta_distribue IS NOT NULL AND milieu IS NOT NULL
        GROUP BY milieu ORDER BY milieu
    """)

    a["dd_by_genre"] = qry(con, f"""
        SELECT genre_cm,
            COUNT(*)                                                AS n_valid,
            ROUND(AVG(delta_distribue),   6)  AS mean,
            ROUND(STDDEV(delta_distribue),6)  AS std,
            ROUND(MEDIAN(delta_distribue),6)  AS median
        FROM read_parquet('{dp}')
        WHERE delta_distribue IS NOT NULL AND genre_cm IS NOT NULL
        GROUP BY genre_cm ORDER BY genre_cm
    """)

    # Bucket labels reflect which side is excluded:
    # pairs <= 90 days have delta_distribue = NaN (excluded / short gap)
    # pairs >  90 days have delta_distribue computed  (included / long gap)
    a["dd_days_dist"] = qry(con, f"""
        SELECT
            CASE
                WHEN days_between <= 30  THEN '1–30 j (exclu)'
                WHEN days_between <= 60  THEN '31–60 j (exclu)'
                WHEN days_between <= 90  THEN '61–90 j (exclu)'
                WHEN days_between <= 180 THEN '91–180 j'
                ELSE '> 180 j'
            END AS bucket,
            COUNT(*) AS n
        FROM read_parquet('{dp}')
        WHERE days_between IS NOT NULL
        GROUP BY bucket ORDER BY MIN(days_between)
    """)

    a["dd_scatter_sample"] = qry(con, f"""
        SELECT days_between, delta_ISE, delta_distribue, programme
        FROM read_parquet('{dp}')
        WHERE delta_distribue IS NOT NULL
          AND delta_ISE IS NOT NULL
        USING SAMPLE 5000
    """)

    a["dd_timeseries"] = qry(con, f"""
        SELECT DATE_TRUNC('month', date_apres) AS month,
               programme,
               ROUND(AVG(delta_distribue),   6) AS mean_dd,
               ROUND(MEDIAN(delta_distribue),6) AS median_dd,
               ROUND(STDDEV(delta_distribue),6) AS std_dd,
               COUNT(*) AS n
        FROM read_parquet('{dp}')
        WHERE delta_distribue IS NOT NULL
        GROUP BY month, programme
        ORDER BY month, programme
    """)
    print(f"  dd_global       = {len(a['dd_global'])} row")
    print(f"  dd_by_programme = {len(a['dd_by_programme'])} rows")
    print(f"  dd_timeseries   = {len(a['dd_timeseries'])} rows")

    delta_cols = qry(con, f"DESCRIBE SELECT * FROM read_parquet('{dp}')")[
        "column_name"].tolist()

    if "status_change" in delta_cols:
        slbl = {"gained": "A gagné l'éligibilité",
                "lost":   "A perdu l'éligibilité",
                "stable": "Stable"}
        sc = qry(con, f"""
            SELECT status_change AS code, COUNT(*) AS "Transitions"
            FROM read_parquet('{dp}')
            GROUP BY status_change ORDER BY "Transitions" DESC
        """)
        sc["Statut"] = sc["code"].map(slbl).fillna(sc["code"])
        a["status_counts"] = sc

        if "region" in delta_cols:
            sr = qry(con, f"""
                SELECT region, status_change, COUNT(*) AS "Transitions"
                FROM read_parquet('{dp}')
                WHERE status_change != 'stable' AND region IS NOT NULL
                GROUP BY region, status_change ORDER BY region, status_change
            """)
            sr["Statut"] = sr["status_change"].map(slbl).fillna(sr["status_change"])
            a["status_by_region"] = sr
        print(f"  status_counts = {len(a['status_counts'])} categories")

    if "status_change" in delta_cols and "side_avant" in delta_cols:
        churn_prog = qry(con, f"""
            SELECT programme,
                SUM(CASE WHEN status_change = 'gained' THEN 1 ELSE 0 END)
                    AS "Devenus éligibles (entrées)",
                SUM(CASE WHEN status_change = 'lost' THEN 1 ELSE 0 END)
                    AS "Perdus l'éligibilité (sorties)",
                (SUM(CASE WHEN side_avant = 'eligible' THEN 1 ELSE 0 END) +
                 SUM(CASE WHEN side_apres = 'eligible' THEN 1 ELSE 0 END)) / 2.0
                    AS "Stock moyen d'éligibles"
            FROM read_parquet('{dp}')
            GROUP BY programme
        """)
        churn_prog["Taux de rotation (churn)"] = (
            (churn_prog["Devenus éligibles (entrées)"] +
             churn_prog["Perdus l'éligibilité (sorties)"]) /
            churn_prog["Stock moyen d'éligibles"]
        ).round(4)
        a["churn_prog"] = churn_prog.dropna(subset=["Taux de rotation (churn)"])

        if "region" in delta_cols:
            churn_reg = qry(con, f"""
                SELECT programme, region,
                    SUM(CASE WHEN status_change = 'gained' THEN 1 ELSE 0 END) AS entries,
                    SUM(CASE WHEN status_change = 'lost'   THEN 1 ELSE 0 END) AS exits,
                    (SUM(CASE WHEN side_avant = 'eligible' THEN 1 ELSE 0 END) +
                     SUM(CASE WHEN side_apres = 'eligible' THEN 1 ELSE 0 END)) / 2.0 AS stock
                FROM read_parquet('{dp}')
                WHERE programme IN ('AMOT', 'ASD') AND region IS NOT NULL
                GROUP BY programme, region
            """)
            churn_reg["Taux de rotation (churn)"] = (
                (churn_reg["entries"] + churn_reg["exits"]) / churn_reg["stock"]
            ).round(4)
            a["churn_reg"] = churn_reg.dropna(subset=["Taux de rotation (churn)"])
        print(f"  churn_prog  = {len(a['churn_prog'])} rows")

        # ── Weekly volatility aggregations ────────────────────────────────────
        # Both queries deduplicate to ONE ROW PER MENAGE PER WEEK before
        # aggregating. A household can have multiple score events in the same
        # calendar week (e.g. a correction 2 days after an inscription).
        # Diagnostic showed 7.34% of transitions were within-week duplicates,
        # with median days_between = 2 days — short-gap administrative updates,
        # not meaningful score changes.
        #
        # Deduplication strategy: keep the transition with the LONGEST
        # days_between within the week (most meaningful / stable change).
        # Ties broken by largest ABS(delta_ISE).
        #
        # COUNT(*) after dedup = n_menages (unique households that week),
        # not raw transition count.

        # ── σ(ΔISE) per week ──────────────────────────────────────────────────
        a["delta_vol_weekly"] = qry(con, f"""
            WITH deduped AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY menage_ano, DATE_TRUNC('week', date_apres)
                        ORDER BY days_between DESC, ABS(delta_ISE) DESC
                    ) AS rn
                FROM read_parquet('{dp}')
                WHERE delta_ISE IS NOT NULL
                  AND date_apres IS NOT NULL
            )
            SELECT
                DATE_TRUNC('week', date_apres)            AS week,
                COUNT(*)                                  AS n_menages,
                ROUND(AVG(delta_ISE),    6)               AS mean_delta,
                ROUND(MEDIAN(delta_ISE), 6)               AS median_delta,
                ROUND(STDDEV(delta_ISE), 6)               AS sigma_delta,
                ROUND(QUANTILE_CONT(delta_ISE, 0.10), 6) AS p10_delta,
                ROUND(QUANTILE_CONT(delta_ISE, 0.25), 6) AS p25_delta,
                ROUND(QUANTILE_CONT(delta_ISE, 0.75), 6) AS p75_delta,
                ROUND(QUANTILE_CONT(delta_ISE, 0.90), 6) AS p90_delta
            FROM deduped
            WHERE rn = 1
            GROUP BY week
            ORDER BY week
        """)

        # ── mean(|ΔISE|) per week ─────────────────────────────────────────────
        a["delta_abs_weekly"] = qry(con, f"""
            WITH deduped AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY menage_ano, DATE_TRUNC('week', date_apres)
                        ORDER BY days_between DESC, ABS(delta_ISE) DESC
                    ) AS rn
                FROM read_parquet('{dp}')
                WHERE delta_ISE IS NOT NULL
                  AND date_apres IS NOT NULL
            )
            SELECT
                DATE_TRUNC('week', date_apres)                                   AS week,
                COUNT(*)                                                         AS n_menages,
                ROUND(AVG(ABS(delta_ISE)),    6)                                 AS mean_abs_delta,
                ROUND(MEDIAN(ABS(delta_ISE)), 6)                                 AS median_abs_delta,
                ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.10), 6)                   AS p10_abs_delta,
                ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.25), 6)                   AS p25_abs_delta,
                ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.75), 6)                   AS p75_abs_delta,
                ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.90), 6)                   AS p90_abs_delta,
                ROUND(QUANTILE_CONT(ABS(delta_ISE), 0.99), 6)                   AS p99_abs_delta,
                ROUND(AVG(CASE WHEN delta_ISE < 0 THEN ABS(delta_ISE) END), 6)  AS mean_improvement,
                ROUND(AVG(CASE WHEN delta_ISE > 0 THEN delta_ISE END),      6)  AS mean_degradation,
                SUM(CASE WHEN delta_ISE < 0 THEN 1 ELSE 0 END)                  AS n_improving,
                SUM(CASE WHEN delta_ISE > 0 THEN 1 ELSE 0 END)                  AS n_degrading
            FROM deduped
            WHERE rn = 1
            GROUP BY week
            ORDER BY week
        """)

        print(f"  delta_vol_weekly  = {len(a.get('delta_vol_weekly',  pd.DataFrame()))} weeks")
        print(f"  delta_abs_weekly  = {len(a.get('delta_abs_weekly',  pd.DataFrame()))} weeks")


# ── Score timeseries ──────────────────────────────────────────────────────────
section("Score timeseries aggregates")

if parquet("score_timeseries"):
    ts_path = parquet("score_timeseries")
    ts_cols = qry(con, f"DESCRIBE SELECT * FROM read_parquet('{ts_path}')")["column_name"].tolist()

    overall_cols = [
        c for c in [
            "date_calcul",
            "n_menages",
            "score_mean",
            "score_median",
            "score_std",
            "score_min",
            "score_max",
            "score_p10",
            "score_p25",
            "score_p75",
            "score_p90",
        ]
        if c in ts_cols
    ]
    # Keep dashboard payload light: aggregate overall trend to weekly granularity.
    if overall_cols:
        df_ts_overall_raw = qry(con, f"""
            SELECT {", ".join(overall_cols)}
            FROM read_parquet('{ts_path}')
            ORDER BY date_calcul
        """)
        if not df_ts_overall_raw.empty and "date_calcul" in df_ts_overall_raw.columns:
            df_ts_overall_raw["date_calcul"] = pd.to_datetime(df_ts_overall_raw["date_calcul"], errors="coerce")
            df_ts_overall_raw["date_calcul"] = (
                df_ts_overall_raw["date_calcul"].dt.to_period("W").dt.start_time
            )
            agg_map = {c: "mean" for c in df_ts_overall_raw.columns if c != "date_calcul"}
            if "score_min" in agg_map:
                agg_map["score_min"] = "min"
            if "score_max" in agg_map:
                agg_map["score_max"] = "max"
            if "n_menages" in agg_map:
                agg_map["n_menages"] = "max"
            a["score_ts_overall"] = (
                df_ts_overall_raw.groupby("date_calcul", as_index=False).agg(agg_map).sort_values("date_calcul")
            )
        else:
            a["score_ts_overall"] = df_ts_overall_raw
    else:
        a["score_ts_overall"] = pd.DataFrame()

    # Build segmented time-series from master_events when score_timeseries is overall-only.
    if mp:
        a["score_ts_by_region"] = qry(con, f"""
            SELECT DATE_TRUNC('week', date_calcul) AS date_calcul, region, COUNT(DISTINCT menage_ano) AS n_menages,
                   AVG(score_final) AS score_mean, MEDIAN(score_final) AS score_median,
                   STDDEV(score_final) AS score_std
            FROM read_parquet('{mp}')
            WHERE region IS NOT NULL AND date_calcul IS NOT NULL AND score_final IS NOT NULL
            GROUP BY DATE_TRUNC('week', date_calcul), region
            ORDER BY date_calcul
        """)
        a["score_ts_by_milieu"] = qry(con, f"""
            SELECT DATE_TRUNC('week', date_calcul) AS date_calcul, milieu, COUNT(DISTINCT menage_ano) AS n_menages,
                   AVG(score_final) AS score_mean, MEDIAN(score_final) AS score_median,
                   STDDEV(score_final) AS score_std
            FROM read_parquet('{mp}')
            WHERE milieu IS NOT NULL AND date_calcul IS NOT NULL AND score_final IS NOT NULL
            GROUP BY DATE_TRUNC('week', date_calcul), milieu
            ORDER BY date_calcul
        """)
        a["score_ts_by_genre"] = qry(con, f"""
            SELECT DATE_TRUNC('week', date_calcul) AS date_calcul, genre_cm, COUNT(DISTINCT menage_ano) AS n_menages,
                   AVG(score_final) AS score_mean, MEDIAN(score_final) AS score_median,
                   STDDEV(score_final) AS score_std
            FROM read_parquet('{mp}')
            WHERE genre_cm IS NOT NULL AND date_calcul IS NOT NULL AND score_final IS NOT NULL
            GROUP BY DATE_TRUNC('week', date_calcul), genre_cm
            ORDER BY date_calcul
        """)
    else:
        a["score_ts_by_region"] = pd.DataFrame()
        a["score_ts_by_milieu"] = pd.DataFrame()
        a["score_ts_by_genre"] = pd.DataFrame()
    print(f"  score_ts_overall   = {len(a.get('score_ts_overall',   pd.DataFrame()))} rows")
    print(f"  score_ts_by_region = {len(a.get('score_ts_by_region', pd.DataFrame()))} rows")
    print(f"  score_ts_by_milieu = {len(a.get('score_ts_by_milieu', pd.DataFrame()))} rows")
    print(f"  score_ts_by_genre  = {len(a.get('score_ts_by_genre',  pd.DataFrame()))} rows")

# ── Near-threshold simulator timeseries ──────────────────────────────────────
section("Near-threshold simulator timeseries")

if parquet("near_threshold_timeseries"):
    # Built by build_near_threshold_timeseries() — uses running-state logic,
    # so n_total = truly active households that week (same as score_ts_overall n_menages).
    a["near_threshold_ts"] = qry(con, f"""
        SELECT *
        FROM read_parquet('{parquet("near_threshold_timeseries")}')
        ORDER BY week, programme
    """)
    print(f"  near_threshold_ts  = {len(a.get('near_threshold_ts', pd.DataFrame()))} week×programme rows")
else:
    a["near_threshold_ts"] = pd.DataFrame()
    print("  near_threshold_ts  = (parquet not found — re-run etl.py to generate)")

# ── Monthly flows ─────────────────────────────────────────────────────────────
section("Monthly flows (eligibility, beneficiaire & churn)")

if parquet("monthly_eligibility_flows"):
    a["monthly_eligibility_flows"] = qry(con, f"""
        SELECT * FROM read_parquet('{parquet("monthly_eligibility_flows")}')
        ORDER BY date
    """)
    print(f"  monthly_eligibility_flows  = {len(a['monthly_eligibility_flows'])} months")
else:
    a["monthly_eligibility_flows"] = pd.DataFrame()

if parquet("monthly_beneficiaire_flows"):
    a["monthly_beneficiaire_flows"] = qry(con, f"""
        SELECT * FROM read_parquet('{parquet("monthly_beneficiaire_flows")}')
        ORDER BY date
    """)
    print(f"  monthly_beneficiaire_flows = {len(a['monthly_beneficiaire_flows'])} months")
else:
    a["monthly_beneficiaire_flows"] = pd.DataFrame()

# ── Beneficiaire raw stats for Overview row ───────────────────────────────────
rbp = parquet("raw_beneficiaire")
if rbp:
    _bf_total = qry(con, f"SELECT COUNT(*) AS n FROM read_parquet('{rbp}')").iloc[0, 0]
    _bf_uniq  = qry(con, f"""
        SELECT COUNT(DISTINCT menage_ano) AS n FROM read_parquet('{rbp}')
        WHERE menage_ano IS NOT NULL
    """).iloc[0, 0]
    _bf_actif = qry(con, f"""
        SELECT COUNT(*) AS n FROM read_parquet('{rbp}')
        WHERE CAST(actif AS INTEGER) = 1
    """).iloc[0, 0]
    _bf_sz = round(Path(rbp).stat().st_size / 1e6, 1)
    a["beneficiaire_stats"] = {
        "n_total":  int(_bf_total),
        "n_uniq":   int(_bf_uniq),
        "n_actif":  int(_bf_actif),
        "size_mb":  _bf_sz,
    }
    print(f"  beneficiaire_stats: {int(_bf_total):,} rows | {int(_bf_uniq):,} unique menages | {int(_bf_actif):,} actif")
else:
    a["beneficiaire_stats"] = {}

if parquet("churn_timeline"):
    a["churn_timeline"] = qry(con, f"""
        SELECT * FROM read_parquet('{parquet("churn_timeline")}')
        ORDER BY date
    """)
    print(f"  churn_timeline             = {len(a['churn_timeline'])} month×programme rows")
else:
    a["churn_timeline"] = pd.DataFrame()

# ── Re-entry analysis ─────────────────────────────────────────────────────────
section("Re-entry analysis")

if parquet("reentry_detail"):
    a["reentry_detail"] = qry(con, f"""
        SELECT * FROM read_parquet('{parquet("reentry_detail")}')
    """)
    print(f"  reentry_detail  = {len(a['reentry_detail'])} menage×programme pairs")
else:
    a["reentry_detail"] = pd.DataFrame()

if parquet("reentry_summary"):
    a["reentry_summary"] = qry(con, f"""
        SELECT * FROM read_parquet('{parquet("reentry_summary")}')
    """)
    print(f"  reentry_summary = {len(a['reentry_summary'])} rows")
else:
    a["reentry_summary"] = pd.DataFrame()

# Compute reentry_stats: one summary row per programme for the KPI cards
# (total menages, how many re-entered at least once, max, and average among churners)
if not a.get("reentry_detail", pd.DataFrame()).empty:
    rd = a["reentry_detail"]
    stats_rows = []
    for prog in rd["programme"].unique():
        sub = rd[rd["programme"] == prog]
        total        = sub["menage_ano"].nunique()
        with_reentry = sub[sub["n_reentries"] > 0]["menage_ano"].nunique()
        max_re       = int(sub["n_reentries"].max())
        churners     = sub[sub["n_reentries"] > 0]["n_reentries"]
        avg_re       = round(float(churners.mean()), 2) if len(churners) > 0 else float("nan")
        pct          = round(with_reentry / total * 100, 1) if total > 0 else 0.0
        stats_rows.append({
            "programme":                    prog,
            "total_menages":                total,
            "n_avec_reentree":              with_reentry,
            "pct_avec_reentree":            pct,
            "max_reentrees":                max_re,
            "avg_reentrees_parmi_churners": avg_re,
        })
    a["reentry_stats"] = pd.DataFrame(stats_rows)
    print(f"  reentry_stats   = {len(a['reentry_stats'])} programme rows")
    for _, r in a["reentry_stats"].iterrows():
        print(f"    {r['programme']:<20}  total={r['total_menages']:,}  "
              f"re-entries={r['n_avec_reentree']:,} ({r['pct_avec_reentree']}%)")
else:
    a["reentry_stats"] = pd.DataFrame()

# ── Metadata & save ───────────────────────────────────────────────────────────
section("Saving cache")

a["_baked_at"] = datetime.datetime.now()
a["_snap_dir"]  = str(SNAP_DIR.resolve())

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)

size_kb = OUT_PATH.stat().st_size / 1024
print(f"\n    Cache saved to : {OUT_PATH.resolve()}")
print(f"    Cache size     : {size_kb:.1f} KB")
print(f"    Baked at       : {a['_baked_at'].strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n    Dashboard will now load in < 1 second on next startup.")
print()