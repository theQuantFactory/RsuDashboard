"""
RSU Analytics Dashboard
────────────────────────
Run:  streamlit run dashboard.py
      (from rsu_pipeline/ folder)

Fast startup: loads pre-baked cache from snapshots/dashboard_cache.pkl
if available. Falls back to live DuckDB aggregation if cache is missing.

To refresh the cache after a new pipeline run:
    python prebake_dashboard.py
"""

import pickle
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RSU Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #ffffff; color: #1a1a2e; }
[data-testid="stSidebar"] {
    background-color: #f7f8fa;
    border-right: 1px solid #e0e4ea;
}
.metric-card {
    background: transparent;
    border: none;
    border-left: 3px solid #e0e4ea;
    border-radius: 0;
    padding: 4px 14px;
    margin-bottom: 0;
    transition: border-left-color 0.2s;
}
.metric-card:hover { border-left-color: #1a56db; }
.metric-label {
    font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: #9ca3af; margin-bottom: 1px; line-height: 1.3;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.65rem;
    font-weight: 700; color: #111827; line-height: 1.1; letter-spacing: -0.02em;
}
.metric-sub { font-size: 0.69rem; color: #6b7280; margin-top: 1px; }
.warn  { color: #c2410c !important; }
.good  { color: #15803d !important; }
.metric-sub.warn { color: #c2410c !important; }
.metric-sub.good { color: #15803d !important; }
.sec {
    font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #6b7280; border-bottom: 1px solid #e0e4ea;
    padding-bottom: 6px; margin: 22px 0 10px 0;
}
.cache-badge {
    display: inline-block; font-size: 0.65rem; padding: 2px 8px;
    border-radius: 10px; margin-left: 8px; font-family: monospace;
}
.cache-hit  { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
.cache-miss { background: #fff7ed; color: #c2410c; border: 1px solid #fdba74; }
footer { visibility: hidden; }
hr { border-color: #e0e4ea; }
/* programme color classes */
.prog-amot { color: #1a56db !important; font-weight: 600; }
.prog-asd  { color: #0e9f6e !important; font-weight: 600; }
.prog-amoa { color: #d97706 !important; font-weight: 600; }
.prog-nc   { color: #9ca3af !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent


def _resolve_app_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if p.exists() or p.parent.exists():
        return p
    return APP_DIR / p


SNAP_DIR = _resolve_app_path("snapshots/csv")
CACHE_PATH = _resolve_app_path("snapshots/dashboard_cache.pkl")

PROG_COLORS = {
    "AMOT":          "#1a56db",
    "ASD":           "#0e9f6e",
    "AMOA":          "#d97706",
    "Non classifié": "#9ca3af",
}
PROG_FULL = {
    "AMOT":          "AMO Tadamon (AMOT)",
    "ASD":           "ASD — Aide Sociale Directe",
    "AMOA":          "AMO Achamil (AMOA)",
    "Non classifié": "Non classifié — Sans programme",
}
# Official eligibility thresholds
THRESHOLDS = {
    "AMOT": 9.3264284,
    "ASD":  9.743001,
}
PL = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#f7f8fa",
    font=dict(family="IBM Plex Sans", color="#6b7280", size=12),
    margin=dict(t=36, b=28, l=36, r=20),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "—"
    return f"{int(n):,}"

def card(fr, en, val, sub=None, cls=""):
    s = f'<div class="metric-sub {cls}">{sub}</div>' if sub else ""
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{fr} · {en}</div>'
            f'<div class="metric-value">{val}</div>{s}</div>')

def sec(fr, en=""):
    label = f"{fr} &nbsp;·&nbsp; {en}" if en else fr
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def ptitle(fr, en, sfr="", sen=""):
    sub = (f"<p style='color:#6b7280;font-size:0.83rem;margin-bottom:20px'>"
           f"{sfr} · {sen}</p>") if sfr else ""
    st.markdown(
        f"<h1 style='font-size:1.5rem;font-weight:600;color:#111827;"
        f"margin-bottom:4px'>{fr} · {en}</h1>{sub}",
        unsafe_allow_html=True)


# ── Load aggregations ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_agg():
    """
    Load pre-baked cache if available and up to date.
    Falls back to live DuckDB aggregation if cache is stale or missing.
    """
    cache_valid = False
    if CACHE_PATH.exists():
        cache_mtime = CACHE_PATH.stat().st_mtime
        if SNAP_DIR.exists():
            newest_snap = max(
                (p.stat().st_mtime for p in SNAP_DIR.glob("*.parquet")),
                default=0
            )
            cache_valid = cache_mtime >= newest_snap

    if cache_valid:
        with open(CACHE_PATH, "rb") as f:
            agg = pickle.load(f)
        agg["_source"] = "cache"
        return agg

    # ── Fallback: live DuckDB aggregation ─────────────────────────────────────
    try:
        import duckdb
    except ImportError:
        return {"_source": "error",
                "_error": "duckdb not installed and no cache found. "
                           "Run: pip install duckdb  then  python prebake_dashboard.py"}

    if not SNAP_DIR.exists():
        return {"_source": "error",
                "_error": f"Snapshot directory not found: {SNAP_DIR}"}

    def parquet(name):
        p = SNAP_DIR / f"{name}.parquet"
        return str(p) if p.exists() else None

    def qry(sql):
        try:
            return con.execute(sql).df()
        except Exception:
            return pd.DataFrame()

    con = duckdb.connect()
    a = {"_source": "live_duckdb"}

    rows = []
    for p in sorted(SNAP_DIR.glob("*.parquet")):
        n = qry(f"SELECT COUNT(*) FROM read_parquet('{p}')").iloc[0, 0]
        rows.append({
            "Fichier":     p.stem,
            "Lignes":      int(n),
            "Taille (MB)": round(p.stat().st_size / 1e6, 1),
            "Mis à jour":  datetime.datetime.fromtimestamp(
                               p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    a["files"] = pd.DataFrame(rows)
    a["last_update"] = max(
        datetime.datetime.fromtimestamp(p.stat().st_mtime)
        for p in SNAP_DIR.glob("*.parquet")
    )

    mp  = parquet("master_events")
    dp  = parquet("delta_frame")
    rmp = parquet("raw_menage")
    rsp = parquet("raw_scores")
    tp  = parquet("menage_trajectory")

    # ── Quality-check metrics ─────────────────────────────────────────────────
    if rsp:
        # Total raw score rows (before any dedup/filtering)
        a["qc_raw_score_rows"] = int(qry(
            f"SELECT COUNT(*) FROM read_parquet('{rsp}')"
        ).iloc[0, 0])

        # Duplicate score_id_ano count
        dup_res = qry(f"""
            SELECT COUNT(*) - COUNT(DISTINCT score_id_ano) AS n_dupes
            FROM read_parquet('{rsp}')
            WHERE score_id_ano IS NOT NULL
        """)
        a["qc_score_dupes"] = int(dup_res["n_dupes"].iloc[0]) if not dup_res.empty else 0

        # Aberrant scores (score_final > 15)
        if mp:
            ab_res = qry(f"""
                SELECT
                    (SELECT COUNT(*) FROM read_parquet('{rsp}')) -
                    (SELECT COUNT(*) FROM read_parquet('{mp}')) AS n_aberrant
            """)
            a["qc_aberrant_scores"] = max(int(ab_res.iloc[0, 0]) if not ab_res.empty else 0, 0)

        # Score period coverage
        period_res = qry(f"""
            SELECT
                MIN(date_calcul) AS date_min,
                MAX(date_calcul) AS date_max
            FROM read_parquet('{rsp}')
            WHERE date_calcul IS NOT NULL
        """)
        if not period_res.empty:
            a["qc_score_date_min"] = pd.to_datetime(period_res["date_min"].iloc[0])
            a["qc_score_date_max"] = pd.to_datetime(period_res["date_max"].iloc[0])

    if rmp:
        # Total raw menage rows
        a["qc_raw_menage_rows"] = int(qry(
            f"SELECT COUNT(*) FROM read_parquet('{rmp}')"
        ).iloc[0, 0])

        # Duplicate menage_ano
        dup_m = qry(f"""
            SELECT COUNT(*) - COUNT(DISTINCT menage_ano) AS n_dupes
            FROM read_parquet('{rmp}')
            WHERE menage_ano IS NOT NULL
        """)
        a["qc_menage_dupes"] = int(dup_m["n_dupes"].iloc[0]) if not dup_m.empty else 0

        # Menages missing a score entirely
        if mp:
            no_score = qry(f"""
                SELECT COUNT(*) AS n
                FROM read_parquet('{rmp}') m
                LEFT JOIN (
                    SELECT DISTINCT menage_ano FROM read_parquet('{mp}')
                ) s USING (menage_ano)
                WHERE s.menage_ano IS NULL
            """)
            a["qc_menages_no_score"] = int(no_score["n"].iloc[0]) if not no_score.empty else 0

        # Missing key demographic fields
        null_region = qry(f"""
            SELECT COUNT(*) AS n FROM read_parquet('{rmp}')
            WHERE region IS NULL OR TRIM(CAST(region AS VARCHAR)) = ''
        """)
        a["qc_null_region"] = int(null_region["n"].iloc[0]) if not null_region.empty else 0

        null_milieu = qry(f"""
            SELECT COUNT(*) AS n FROM read_parquet('{rmp}')
            WHERE milieu IS NULL OR TRIM(CAST(milieu AS VARCHAR)) = ''
        """)
        a["qc_null_milieu"] = int(null_milieu["n"].iloc[0]) if not null_milieu.empty else 0

    if mp:
        a["n_menages_uniq"] = int(qry(f"""
            SELECT COUNT(DISTINCT menage_ano) FROM read_parquet('{mp}')""").iloc[0,0])
        a["n_score_events"] = int(qry(f"""
            SELECT COUNT(*) FROM read_parquet('{parquet("raw_scores") or mp}')""").iloc[0,0])

        # Ménages par programme — exclude Non classifié
        a["prog_menages"] = qry(f"""
            SELECT programme,
                   COUNT(DISTINCT menage_ano) AS "Ménages uniques"
            FROM read_parquet('{mp}')
            WHERE programme != 'Non classifié'
            GROUP BY programme ORDER BY "Ménages uniques" DESC""")
        a["prog_menages"]["Nom complet"] = (
            a["prog_menages"]["programme"].map(PROG_FULL)
            .fillna(a["prog_menages"]["programme"]))

        nc = qry(f"""
            SELECT COUNT(DISTINCT menage_ano) AS n
            FROM read_parquet('{mp}')
            WHERE programme = 'Non classifié'
        """)
        a["n_non_classifie"] = int(nc["n"].iloc[0]) if not nc.empty else 0

        a["score_stats"] = qry(f"""
            SELECT programme AS "Programme",
                   COUNT(*) AS "Événements",
                   ROUND(AVG(score_final),4)    AS "Moyenne",
                   ROUND(MEDIAN(score_final),4) AS "Médiane",
                   ROUND(STDDEV(score_final),4) AS "Écart-type",
                   ROUND(MIN(score_final),4)    AS "Min",
                   ROUND(MAX(score_final),4)    AS "Max"
            FROM read_parquet('{mp}')
            GROUP BY programme ORDER BY programme""")
        a["score_hist"] = qry(f"""
            SELECT programme, ROUND(score_final*20)/20 AS bin, COUNT(*) AS n
            FROM read_parquet('{mp}')
            WHERE score_final IS NOT NULL
            GROUP BY programme, bin ORDER BY programme, bin""")
        elig = qry(f"""
            SELECT SUM(CAST(eligible AS INT)) AS ne, COUNT(*) AS nt
            FROM read_parquet('{mp}')""")
        ne, nt = int(elig["ne"].iloc[0]), int(elig["nt"].iloc[0])
        a["n_eligible"] = ne; a["n_not_eligible"] = nt - ne
        a["pct_eligible"] = round(ne/nt*100, 1) if nt else 0
        a["elig_by_prog"] = qry(f"""
            SELECT programme,
                   SUM(CAST(eligible AS INT)) AS eligible,
                   COUNT(*) AS total,
                   ROUND(100.0*SUM(CAST(eligible AS INT))/COUNT(*),1) AS "% éligible"
            FROM read_parquet('{mp}')
            GROUP BY programme ORDER BY programme""")
        a["elig_by_region"] = qry(f"""
            SELECT region AS "Région",
                   ROUND(100.0*AVG(CAST(eligible AS INT)),1) AS "% éligible"
            FROM read_parquet('{mp}')
            WHERE region IS NOT NULL
            GROUP BY region ORDER BY "% éligible" ASC""")
        a["elig_by_milieu"] = qry(f"""
            SELECT programme AS "Programme", milieu AS "Milieu",
                   ROUND(100.0*AVG(CAST(eligible AS INT)),1) AS "% éligible"
            FROM read_parquet('{mp}')
            WHERE milieu IS NOT NULL
            GROUP BY programme, milieu ORDER BY programme, milieu""")
        a["n_near_10"] = int(qry(f"""
            SELECT COALESCE(SUM(CAST("near_0.10" AS INT)),0)
            FROM read_parquet('{mp}')""").iloc[0,0])
        a["n_near_25"] = int(qry(f"""
            SELECT COALESCE(SUM(CAST("near_0.25" AS INT)),0)
            FROM read_parquet('{mp}')""").iloc[0,0])

        # Near-threshold timeseries — read from pre-built parquet (running-state based)
        ntt_path = str(SNAP_DIR / "near_threshold_timeseries.parquet")
        if Path(ntt_path).exists():
            a["near_threshold_ts"] = qry(f"""
                SELECT * FROM read_parquet('{ntt_path}') ORDER BY week, programme
            """)

    if rmp:
        a["n_menages"] = int(qry(f"SELECT COUNT(*) FROM read_parquet('{rmp}')").iloc[0,0])
        a["milieu"]       = qry(f"SELECT milieu AS 'Milieu', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') GROUP BY milieu ORDER BY 'Ménages' DESC")
        a["genre"]        = qry(f"SELECT genre_cm AS 'Genre', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') GROUP BY genre_cm ORDER BY 'Ménages' DESC")
        a["matrimonial"]  = qry(f"SELECT etat_matrimonial_cm AS 'État', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') GROUP BY etat_matrimonial_cm ORDER BY 'Ménages' DESC")
        a["region_dem"]   = qry(f"SELECT region AS 'Région', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') GROUP BY region ORDER BY 'Ménages' ASC")
        a["type_menage"]  = qry(f"SELECT type_menage AS 'Type', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') GROUP BY type_menage ORDER BY 'Ménages' DESC")
        taille = qry(f"SELECT taille_menage AS 'Personnes', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') WHERE taille_menage<=12 GROUP BY taille_menage ORDER BY taille_menage ASC")
        a["taille"] = taille
        prov = qry(f"SELECT province AS 'Province', COUNT(*) AS 'Ménages' FROM read_parquet('{rmp}') GROUP BY province ORDER BY 'Ménages' DESC LIMIT 20")
        a["top_provinces"] = prov.sort_values("Ménages")

    if tp:
        cols = qry(f"DESCRIBE SELECT * FROM read_parquet('{tp}')")[
            "column_name"].tolist()

        # Radiés who never came back:
        # est_radie = True AND statut_actuel != 'Inscription' AND statut_actuel != 'mise à jour du dossier'
        # i.e. their last known event type is still a radiation, not a subsequent re-inscription
        if "est_radie" in cols and "statut_actuel" in cols:
            rad_final = qry(f"""
                SELECT COUNT(*) AS n
                FROM read_parquet('{tp}')
                WHERE est_radie = TRUE
                  AND statut_actuel = 'radiation'
            """)
            a["n_radiated"] = int(rad_final["n"].iloc[0]) if not rad_final.empty else 0
        elif "est_radie" in cols:
            a["n_radiated"] = int(qry(f"""
                SELECT COALESCE(SUM(CAST(est_radie AS INT)),0)
                FROM read_parquet('{tp}')""").iloc[0,0])
        else:
            a["n_radiated"] = 0

        if "statut_actuel" in cols:
            slbl = {"stable_eligible":"Stable — éligible","stable_excluded":"Stable — exclu",
                    "gained":"A gagné l'éligibilité","lost":"A perdu l'éligibilité","fluctuating":"Fluctuant"}
            ts = qry(f"SELECT statut_actuel AS code, COUNT(*) AS 'Ménages' FROM read_parquet('{tp}') GROUP BY statut_actuel ORDER BY 'Ménages' DESC")
            ts["Statut"] = ts["code"].map(slbl).fillna(ts["code"])
            a["traj_status"] = ts
    else:
        a["n_radiated"] = 0

    if dp:
        dcols = qry(f"DESCRIBE SELECT * FROM read_parquet('{dp}')")[
            "column_name"].tolist()
        dg = qry(f"""
            SELECT ROUND(AVG(delta_ISE),4) AS mean, ROUND(STDDEV(delta_ISE),4) AS std,
                   ROUND(QUANTILE_CONT(ABS(delta_ISE),0.9),4) AS p90,
                   ROUND(QUANTILE_CONT(ABS(delta_ISE),0.99),4) AS p99
            FROM read_parquet('{dp}') WHERE delta_ISE IS NOT NULL""")
        a["delta_mean"] = float(dg["mean"].iloc[0])
        a["delta_std"]  = float(dg["std"].iloc[0])
        a["delta_p90"]  = float(dg["p90"].iloc[0])
        a["delta_p99"]  = float(dg["p99"].iloc[0])
        a["n_transitions"] = int(qry(f"SELECT COUNT(*) FROM read_parquet('{dp}')").iloc[0,0])
        a["vol_prog"] = qry(f"""
            SELECT programme, COUNT(*) AS n,
                   ROUND(AVG(delta_ISE),4) AS mean_ΔISE,
                   ROUND(STDDEV(delta_ISE),4) AS sigma_ΔISE,
                   ROUND(QUANTILE_CONT(ABS(delta_ISE),0.9),4) AS p90
            FROM read_parquet('{dp}') WHERE delta_ISE IS NOT NULL
            GROUP BY programme ORDER BY programme""")
        a["delta_box"] = qry(f"""
            SELECT programme,
                   ROUND(QUANTILE_CONT(delta_ISE,0.10),4) AS p10,
                   ROUND(QUANTILE_CONT(delta_ISE,0.25),4) AS q1,
                   ROUND(MEDIAN(delta_ISE),4) AS median,
                   ROUND(QUANTILE_CONT(delta_ISE,0.75),4) AS q3,
                   ROUND(QUANTILE_CONT(delta_ISE,0.90),4) AS p90,
                   ROUND(AVG(delta_ISE),4) AS mean
            FROM read_parquet('{dp}') WHERE delta_ISE IS NOT NULL
            GROUP BY programme""")
        if "region" in dcols:
            vr = qry(f"""
                SELECT programme, region,
                       ROUND(AVG(delta_ISE),4) AS mean_ΔISE,
                       ROUND(STDDEV(delta_ISE),4) AS sigma_ΔISE
                FROM read_parquet('{dp}')
                WHERE delta_ISE IS NOT NULL AND region IS NOT NULL
                GROUP BY programme, region ORDER BY programme, region""")
            a["vol_reg"] = vr
            if not vr.empty:
                a["vol_heatmap"] = vr.pivot(index="region",columns="programme",values="mean_ΔISE")
        if "status_change" in dcols:
            slbl = {"gained":"A gagné l'éligibilité","lost":"A perdu l'éligibilité","stable":"Stable"}
            sc = qry(f"""SELECT status_change AS code, COUNT(*) AS "Transitions"
                FROM read_parquet('{dp}') GROUP BY status_change ORDER BY "Transitions" DESC""")
            sc["Statut"] = sc["code"].map(slbl).fillna(sc["code"])
            a["status_counts"] = sc
            if "region" in dcols:
                sr = qry(f"""SELECT region, status_change, COUNT(*) AS "Transitions"
                    FROM read_parquet('{dp}')
                    WHERE status_change!='stable' AND region IS NOT NULL
                    GROUP BY region, status_change ORDER BY region, status_change""")
                sr["Statut"] = sr["status_change"].map(slbl).fillna(sr["status_change"])
                a["status_by_region"] = sr
        if "status_change" in dcols and "side_avant" in dcols:
            cp = qry(f"""
                SELECT programme,
                    SUM(CASE WHEN status_change='gained' THEN 1 ELSE 0 END) AS entries,
                    SUM(CASE WHEN status_change='lost'   THEN 1 ELSE 0 END) AS exits,
                    (SUM(CASE WHEN side_avant='eligible' THEN 1 ELSE 0 END)+
                     SUM(CASE WHEN side_apres='eligible' THEN 1 ELSE 0 END))/2.0 AS stock
                FROM read_parquet('{dp}') GROUP BY programme""")
            cp.columns = ["programme","Devenus éligibles (entrées)",
                          "Perdus l'éligibilité (sorties)","Stock moyen d'éligibles"]
            cp["Taux de rotation (churn)"] = (
                (cp["Devenus éligibles (entrées)"] + cp["Perdus l'éligibilité (sorties)"]) /
                cp["Stock moyen d'éligibles"]
            ).round(4)
            a["churn_prog"] = cp.dropna(subset=["Taux de rotation (churn)"])
            if "region" in dcols:
                cr = qry(f"""
                    SELECT programme, region,
                        SUM(CASE WHEN status_change='gained' THEN 1 ELSE 0 END) AS entries,
                        SUM(CASE WHEN status_change='lost'   THEN 1 ELSE 0 END) AS exits,
                        (SUM(CASE WHEN side_avant='eligible' THEN 1 ELSE 0 END)+
                         SUM(CASE WHEN side_apres='eligible' THEN 1 ELSE 0 END))/2.0 AS stock
                    FROM read_parquet('{dp}')
                    WHERE programme IN ('AMOT','ASD') AND region IS NOT NULL
                    GROUP BY programme, region""")
                cr["Taux de rotation (churn)"] = (
                    (cr["entries"]+cr["exits"])/cr["stock"]).round(4)
                a["churn_reg"] = cr.dropna(subset=["Taux de rotation (churn)"])

    a["_baked_at"] = None
    return a


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Chargement des données…"):
    agg = load_agg()

if agg.get("_source") == "error":
    st.error(agg.get("_error", "Unknown error"))
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    src = agg.get("_source", "")
    baked_at = agg.get("_baked_at")
    if src == "cache":
        badge_cls, badge_txt = "cache-hit", "cached"
        badge_tip = f"Loaded from pre-baked cache (baked {baked_at.strftime('%Y-%m-%d %H:%M') if baked_at else 'N/A'})"
    else:
        badge_cls, badge_txt = "cache-miss", "live query"
        badge_tip = "No cache found — ran live DuckDB queries. Run prebake_dashboard.py to speed this up."

    st.markdown(f"""
    <div style='padding:16px 0 8px'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:1.05rem;
                    color:#111827;font-weight:600'>RSU Analytics
            <span class='cache-badge {badge_cls}'>{badge_txt}</span>
        </div>
        <div style='font-size:0.7rem;color:#6b7280;margin-top:4px'>
            Tableau de bord · Dashboard</div>
    </div>""", unsafe_allow_html=True)

    if src != "cache":
        st.caption(badge_tip)

    page = st.radio("nav", [
        "Vue d'ensemble",
        "Démographie",
        "Scores et Volatilité",
        "Volatilité Temporelle",
        "Tendances Temporelles",
        "Simulateur de Seuil",
        " Churn",
        "Flux Bénéficiaire",
        "Delta Ajusté",
        "Explorateur de ménage",
    ], label_visibility="collapsed")

    st.markdown("---")
    if agg.get("last_update"):
        st.markdown(
            f"<div style='font-size:0.7rem;color:#6b7280;line-height:1.9'>"
            f"<div>Données mises à jour</div>"
            f"<div style='color:#1a56db;font-family:\"IBM Plex Mono\",monospace'>"
            f"{agg['last_update'].strftime('%Y-%m-%d %H:%M')}</div></div>",
            unsafe_allow_html=True)
    if baked_at:
        st.markdown(
            f"<div style='font-size:0.7rem;color:#6b7280;line-height:1.9;margin-top:8px'>"
            f"<div>Cache généré le</div>"
            f"<div style='color:#15803d;font-family:\"IBM Plex Mono\",monospace'>"
            f"{baked_at.strftime('%Y-%m-%d %H:%M')}</div></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Vue d'ensemble":
    ptitle("Vue d'ensemble", "Overview"
           )

    # ── helpers ───────────────────────────────────────────────────────────────
    df_qc = agg.get("qc_summary", pd.DataFrame())

    def _qc(m):
        r = df_qc[df_qc["metric"] == m]
        return r.iloc[0] if not r.empty else None

    def _int(v):
        try: return int(float(str(v)))
        except: return 0

    def _pct(v):
        try:
            f = float(str(v))
            return f if str(v) not in ("nan","None","") else None
        except: return None

    st.markdown("""
    <style>
    /* ── period card ── */
    .ov-period-card {
        background:#1a56db; border-radius:10px; padding:14px 20px;
        display:inline-block; color:#fff;
    }
    .ov-period-dates {
        font-family:'IBM Plex Mono',monospace; font-size:1.25rem;
        font-weight:700; line-height:1.2; letter-spacing:-0.01em; white-space:nowrap;
    }
    .ov-period-label {
        font-size:0.64rem; text-transform:uppercase; letter-spacing:0.12em;
        opacity:0.72; margin-bottom:4px;
    }
    .ov-period-jours {
        font-size:0.82rem; opacity:0.88; margin-top:4px; white-space:nowrap;
    }

    /* ── three-column row layout ── */
    .ov-section-hdr {
        display:grid; grid-template-columns:minmax(140px,180px) minmax(120px,160px) 1fr; gap:0;
        margin-bottom:6px; padding-bottom:6px; border-bottom:2px solid #111827;
    }
    .ov-section-hdr-cell {
        padding:0 14px; font-size:0.65rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.12em; color:#6b7280;
    }
    .ov-section-hdr-cell:first-child { padding-left:0; }

    .ov-row {
        display:grid;
        grid-template-columns:minmax(140px,180px) minmax(120px,160px) 1fr;
        gap:0;
        align-items:stretch;
        border-bottom:1px solid #e0e4ea;
        padding:12px 0;
    }
    .ov-row:last-child { border-bottom:none; }
    .ov-cell { padding:0 14px; }
    .ov-cell:first-child { padding-left:0; }
    .ov-cell:last-child  { padding-right:0; }
    .ov-cell-mid { border-left:1px solid #e0e4ea; border-right:1px solid #e0e4ea; }

    /* ── raw file cell ── */
    .ov-fname {
        font-family:'IBM Plex Mono',monospace; font-size:0.88rem;
        font-weight:700; margin-bottom:8px;
    }
    .ov-flines {
        font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
        font-weight:700; color:#1a56db; display:block; line-height:1;
    }
    .ov-flines-label {
        font-size:0.72rem; color:#6b7280; display:block; margin-top:2px;
        text-transform:uppercase; letter-spacing:0.06em;
    }
    .ov-fsize  { font-size:0.72rem; color:#9ca3af; display:block; margin-top:4px; }

    /* ── preprocessing prose (col 3) ── */
    .ov-expl-prose {
        font-size:0.95rem; color:#374151; line-height:1.85;
    }
    .ov-expl-prose .ov-hi-red    { color:#dc2626; font-weight:700; }
    .ov-expl-prose .ov-hi-orange { color:#d97706; font-weight:700; }
    .ov-expl-prose .ov-hi-green  { color:#15803d; font-weight:700; }

    /* ── summary / result cell ── */
    .ov-summary-after {
        margin-bottom:10px;
    }
    .ov-summary-num {
        font-family:'IBM Plex Mono',monospace; font-size:1.05rem;
        font-weight:700; line-height:1;  white-space:nowrap;
    }
    .ov-summary-lbl {
        font-size:0.78rem; color:#6b7280; margin-top:2px;
        text-transform:uppercase; letter-spacing:0.06em;
    }
    .ov-summary-sentence {
        font-size:0.92rem; color:#374151; line-height:1.75;
        border-top:1px dashed #e0e4ea; padding-top:8px; margin-top:4px;
    }
    .ov-summary-sentence strong { color:#111827; }
    .ov-hi-red    { color:#dc2626; font-weight:700; font-family:'IBM Plex Mono',monospace; }
    .ov-hi-orange { color:#d97706; font-weight:700; font-family:'IBM Plex Mono',monospace; }
    .ov-hi-green  { color:#15803d; font-weight:700; font-family:'IBM Plex Mono',monospace; }

    /* ── derived tables ── */
    .ov-derived-grid {
        display:grid; grid-template-columns:repeat(3,1fr); gap:12px;
        margin-top:4px;
    }
    .ov-derived-card {
        background:#f7f8fa; border:1px solid #e0e4ea; border-radius:8px;
        padding:16px 18px;
    }
    .ov-derived-name {
        font-family:'IBM Plex Mono',monospace; font-size:0.90rem;
        font-weight:700; color:#1a56db; margin-bottom:6px;
    }
    .ov-derived-desc { font-size:0.88rem; color:#374151; line-height:1.6; }
    .ov-derived-rows {
        font-family:'IBM Plex Mono',monospace; font-size:0.80rem;
        color:#9ca3af; margin-top:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Top bar: period card + title ──────────────────────────────────────────
    r_period = _qc("scores_period")
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(
            "<p style='color:#6b7280;font-size:0.82rem;margin:0;padding-top:6px'>"
            "Chaque fichier CSV reçu est présenté ligne par ligne avec le nettoyage "
            "appliqué et les anomalies détectées.</p>",
            unsafe_allow_html=True)
    with top_right:
        if r_period is not None:
            try:
                d_min  = pd.to_datetime(str(r_period["before"])).strftime("%d/%m/%Y")
                d_max  = pd.to_datetime(str(r_period["after"])).strftime("%d/%m/%Y")
                n_days = _int(r_period["dropped"])
                st.markdown(
                    f"<div class='ov-period-card'>"
                    f"  <div class='ov-period-label'>Période couverte · Jours couverts</div>"
                    f"  <div class='ov-period-dates'>{d_min} → {d_max}</div>"
                    f"  <div class='ov-period-jours'>({n_days:,} jours)</div>"
                    f"</div>",
                    unsafe_allow_html=True)
            except Exception:
                pass

    st.markdown("<br>", unsafe_allow_html=True)

    # ── helper renderers ──────────────────────────────────────────────────────
    df_files = agg.get("files", pd.DataFrame())

    def _file_meta(stem):
        if df_files.empty: return None, None
        m = df_files[df_files["Fichier"] == stem]
        if m.empty: return None, None
        r = m.iloc[0]
        return r.get("Lignes"), r.get("Taille (MB)")

    # Col 1: just filename + line count + size
    def _raw_cell(fname, color, stem):
        lines, size = _file_meta(stem)
        n_fmt = f"{int(lines):,}" if lines else "—"
        s_fmt = f"{float(size):.1f} MB" if size else ""
        return (
            f"<div class='ov-fname' style='color:{color}'>{fname}</div>"
            f"<span class='ov-flines'>{n_fmt}</span>"
            f"<span class='ov-flines-label'>lignes</span>"
            f"{'<span class=\"ov-fsize\">' + s_fmt + '</span>' if s_fmt else ''}"
        )

    def _raw_cell_n(fname, color, n_override):
        n_fmt = f"{int(n_override):,}" if n_override else "—"
        return (
            f"<div class='ov-fname' style='color:{color}'>{fname}</div>"
            f"<span class='ov-flines'>{n_fmt}</span>"
            f"<span class='ov-flines-label'>lignes</span>"
        )

    # Col 1: filename + RAW line count (before any cleaning)
    def _raw_cell(fname, color, n_raw, size_mb=None):
        n_fmt = fmt(n_raw) if n_raw else "—"
        s_fmt = f"{float(size_mb):.1f} MB" if size_mb else ""
        return (
            f"<div class='ov-fname' style='color:{color}'>{fname}</div>"
            f"<span class='ov-flines'>{n_fmt}</span>"
            f"<span class='ov-flines-label'>lignes brutes reçues</span>"
            f"{'<span class=\"ov-fsize\">' + s_fmt + '</span>' if s_fmt else ''}"
        )

    # Col 2: clean count only
    def _summary_cell(after_val, after_label, after_color):
        return (
            f"<div class='ov-summary-after'>"
            f"  <div class='ov-summary-num' style='color:{after_color}'>{after_val}</div>"
            f"  <div class='ov-summary-lbl'>{after_label}</div>"
            f"</div>"
        )

    # Col 3: plain prose sentence — no tags, no sub-steps
    def _expl_cell(sentence_html):
        return f"<div class='ov-expl-prose'>{sentence_html}</div>"

    def _row(raw_html, clean_html, expl_html):
        return (
            f"<div class='ov-row'>"
            f"<div class='ov-cell'>{raw_html}</div>"
            f"<div class='ov-cell ov-cell-mid'>{clean_html}</div>"
            f"<div class='ov-cell'>{expl_html}</div>"
            f"</div>"
        )

    # ── Column header ─────────────────────────────────────────────────────────
    st.markdown(
        "<div class='ov-section-hdr'>"
        "<div class='ov-section-hdr-cell'>Fichier reçu (brut)</div>"
        "<div class='ov-section-hdr-cell' style='padding-left:14px'>Fichier après nettoyage</div>"
        "<div class='ov-section-hdr-cell' style='padding-left:14px'>Explication du prétraitement</div>"
        "</div>",
        unsafe_allow_html=True)

    rows_html = ""

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 1 — menage.csv
    # ═══════════════════════════════════════════════════════════════════════════
    r_mc    = _qc("menage_raw_rows")
    r_md    = _qc("menage_duplicate_ids")
    r_ns    = _qc("menage_no_score")
    nm      = max(agg.get("n_menages", 1), 1)

    n_raw1  = _int(r_mc["before"]) if r_mc is not None else agg.get("qc_raw_menage_rows", 0)
    _, sz1  = _file_meta("raw_menage")
    raw1    = _raw_cell("menage.csv", "#374151", n_raw1, sz1)

    n_men   = agg.get("n_menages", 0)
    total_drop_m = max(int(n_raw1) - int(n_men), 0)
    hl1     = _summary_cell(fmt(n_men), "ménages uniques retenus", "#15803d")

    # prose explanation
    n_rad   = agg.get("n_radiated", 0)
    n_nc    = agg.get("n_non_classifie", 0)
    n_ns_v  = _int(r_ns["dropped"]) if r_ns is not None else int(agg.get("qc_menages_no_score", 0) or 0)
    pct_ns_v= _pct(r_ns.get("pct_dropped")) if r_ns is not None else ((n_ns_v / n_raw1 * 100) if n_raw1 else 0)
    pct_rad = n_rad / nm * 100
    parts1  = []
    nd = _int(r_md.get("dropped")) if r_md is not None else int(agg.get("qc_menage_dupes", 0) or 0)
    pct_d = (_pct(r_md.get("pct_dropped")) if r_md is not None else ((nd / n_raw1 * 100) if n_raw1 else 0)) or 0
    if nd > 0:
        parts1.append(f"Le fichier contenait <span class='ov-hi-red'>{fmt(nd)} doublons</span> "
                      f"({pct_d:.2f}% des lignes) qui ont été supprimés — lignes identiques retirées, "
                      f"premières occurrences conservées en cas d'ID en conflit.")
    else:
        parts1.append("Aucun doublon détecté dans le fichier.")
    if total_drop_m > 0:
        parts1.append(
            f"Au total, <span class='ov-hi-red'>{fmt(total_drop_m)} lignes</span> "
            f"ont été exclues au prétraitement "
            f"({(total_drop_m / n_raw1 * 100) if n_raw1 else 0:.2f}% du brut), "
            "ce qui explique le passage du brut au fichier de travail."
        )
    if n_ns_v > 0:
        parts1.append(f"<span class='ov-hi-red'>{fmt(n_ns_v)} ménages </span> "
                      f"n'ont aucun score ISE associé.")
    if n_rad > 0:
        parts1.append(f"<span class='ov-hi-red'>{fmt(n_rad)} ménages ({pct_rad:.1f}%)</span> "
                      f"ont été radiés définitivement.")
    if n_nc > 0:
        parts1.append(f"<span class='ov-hi-orange'>{fmt(n_nc)} ménages</span> "
                      f"ne sont rattachés à aucun programme (Non classifiés).")
    expl1 = _expl_cell(" ".join(parts1) if parts1 else "Aucune anomalie détectée.")
    rows_html += _row(raw1, hl1, expl1)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 2 — score.csv
    # ═══════════════════════════════════════════════════════════════════════════
    r_sc  = _qc("scores_raw_rows")
    r_sd  = _qc("scores_duplicate_rows")
    r_ab  = _qc("scores_aberrant")

    n_raw2 = _int(r_sc["before"]) if r_sc is not None else agg.get("qc_raw_score_rows", 0)
    _, sz2 = _file_meta("raw_scores")
    raw2   = _raw_cell("score.csv", "#374151", n_raw2, sz2)

    n_ev_clean = agg.get("n_score_events", 0)
    total_drop_s = max(int(n_raw2) - int(n_ev_clean), 0)
    hl2        = _summary_cell(fmt(n_ev_clean), "événements après nettoyage", "#15803d")

    parts2 = []
    if r_sc is not None:
        parts2.append(f"Le fichier a été chargé en blocs de 200 k lignes. "
                    f"Le score final retenu est <em>score_corrigé</em> s'il est renseigné, "
                    f"sinon <em>score_calculé</em>.")
    nd = _int(r_sd.get("dropped")) if r_sd is not None else int(agg.get("qc_score_dupes", 0) or 0)
    pct_d = (_pct(r_sd.get("pct_dropped")) if r_sd is not None else ((nd / n_raw2 * 100) if n_raw2 else 0)) or 0
    if nd > 0:
        parts2.append(f"<span class='ov-hi-red'>{fmt(nd)} doublons exacts</span> ont été supprimés "
                    f"({pct_d:.2f}% des lignes brutes).")
    else:
        parts2.append("Aucun doublon exact détecté.")

    na = _int(r_ab.get("dropped")) if r_ab is not None else int(agg.get("qc_aberrant_scores", 0) or 0)
    pct_a = (_pct(r_ab.get("pct_dropped")) if r_ab is not None else ((na / n_raw2 * 100) if n_raw2 else 0)) or 0
    if na > 0:
        parts2.append(f"<span class='ov-hi-red'>{fmt(na)} scores aberrants</span> (score > 15) "
                    f"ont été retirés ({pct_a:.2f}% post-déduplication).")
    else:
        parts2.append("Aucun score aberrant (> 15) détecté.")
    if total_drop_s > 0:
        parts2.append(
            f"Globalement, <span class='ov-hi-red'>{fmt(total_drop_s)} lignes</span> "
            f"ont été retirées entre brut et dataset final "
            f"({(total_drop_s / n_raw2 * 100) if n_raw2 else 0:.2f}%)."
        )

    # ── Programme enrichment note ──────────────────────────────────────────────
    n_nc = agg.get("n_non_classifie", 0)
    parts2.append(
        "Chaque événement de score a été enrichi avec le programme d'appartenance du ménage "
        "(jointure sur <em>menage_ano</em> avec les fichiers AMOT, ASD et AMOA). "
        + (f"<span class='ov-hi-red'>{fmt(n_nc)} ménages</span> sont absents de tout fichier programme "
        f"et ont été labellisés <em>Non classifié</em>."
        if n_nc > 0 else
        "Tous les ménages ont été appariés à un programme.")
    )

    expl2 = _expl_cell(" ".join(parts2) if parts2 else "Aucune anomalie détectée.")
    rows_html += _row(raw2, hl2, expl2)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROWS 3-5 — programme files (amot / asd / amoa)
    # ═══════════════════════════════════════════════════════════════════════════
    prog_files = [
        ("amot.csv",  "AMOT", "AMO Tadamon",          "programme_AMOT"),
        ("asd.csv",   "ASD",  "Aide Sociale Directe", "programme_ASD"),
        ("amoa.csv",  "AMOA", "AMO Achamil",           "programme_AMOA"),
    ]
    for fname, prog_key, prog_full, stem in prog_files:
        color  = PROG_COLORS.get(prog_key, "#6b7280")
        n_prog = agg.get(f"prog_raw_rows_{prog_key}", None)
        if n_prog is None:
            lines_val, _ = _file_meta(stem)
            n_prog = int(lines_val) if lines_val else 0

        raw_p = _raw_cell(fname, color, n_prog)

        pm = agg.get("prog_menages", pd.DataFrame())
        n_uniq = 0
        if not pm.empty:
            sub_pm = pm[pm["programme"] == prog_key]
            if not sub_pm.empty:
                n_uniq = int(sub_pm.iloc[0].get("Ménages uniques", 0))

        hl_p   = _summary_cell(fmt(n_uniq), "ménages uniques scorés", color)

        pct_match = (n_uniq / n_prog * 100) if n_prog else 0
        sentence_p = (
            f"Le fichier liste les ménages au programme {prog_full}. "
            f"Sur <span class='ov-hi-green'>{fmt(n_prog)} ménages</span> reçues "
            
        ) if n_prog else "Données non disponibles."
        expl_p = _expl_cell(sentence_p)
        rows_html += _row(raw_p, hl_p, expl_p)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 6 — beneficiaire.csv  
    # ═══════════════════════════════════════════════════════════════════════════
    bf_stats = agg.get("beneficiaire_stats", {})

    # Fallback: try to compute from raw DataFrame if stats weren't pre-baked
    if not bf_stats:
        df_benef_raw = agg.get("raw_beneficiaire", pd.DataFrame())
        _benef_lines, _benef_sz = _file_meta("raw_beneficiaire")
        n_benef_raw_fb = len(df_benef_raw) if not df_benef_raw.empty else (
            int(_benef_lines) if _benef_lines else 0)
        if n_benef_raw_fb > 0:
            bf_stats = {
                "n_total": n_benef_raw_fb,
                "n_uniq":  int(df_benef_raw["menage_ano"].nunique())
                           if not df_benef_raw.empty and "menage_ano" in df_benef_raw.columns else 0,
                "n_actif": int((df_benef_raw["actif"] == 1).sum())
                           if not df_benef_raw.empty and "actif" in df_benef_raw.columns else 0,
                "size_mb": float(_benef_sz) if _benef_sz else 0,
            }

    if bf_stats and bf_stats.get("n_total", 0) > 0:
        n_benef_total = bf_stats["n_total"]
        n_benef_uniq  = bf_stats.get("n_uniq", 0)
        benef_size_mb = bf_stats.get("size_mb", None)

        raw_b = _raw_cell("beneficiaire.csv", "#7c3aed", n_benef_total, benef_size_mb)

        avg_records = round(n_benef_total / n_benef_uniq, 1) if n_benef_uniq else 0
        hl_b = _summary_cell(fmt(n_benef_total), "changements de statut (aucune ligne supprimée)", "#7c3aed")

        expl_b = _expl_cell(
            "Ce fichier est une <strong>série temporelle de changements de statut</strong> : "
            "chaque ligne représente un changement de statut bénéficiaire d'un ménage "
            "à une date donnée (<em>date_insert</em>). "
            f"<span class='ov-hi-green'>{fmt(n_benef_uniq)} ménages distincts</span>. "
            
        )
        rows_html += _row(raw_b, hl_b, expl_b)

    st.markdown(rows_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # DERIVED TABLES
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.70rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.12em;color:#6b7280;padding-bottom:8px;"
        "border-bottom:2px solid #111827;margin-bottom:12px'>"
        "Tables dérivées produites par le pipeline · Derived tables</div>",
        unsafe_allow_html=True)

    DERIVED = [
        ("master_events",    "build",  "Événements enrichis",
         "scores ⟕ menage ⟕ programmes. 1 ligne / score × programme. "
         "Contient éligibilité, seuil, dist_threshold.", "raw_scores"),
        ("delta_frame",      "derive", "Transitions ΔISE",
         "Dédoublonnage sur (menage × date) puis shift par ménage. "
         "ΔISE = score[n] − score[n−1]. 1 ligne / paire consécutive.", "delta_frame"),
        ("menage_timeline",  "derive", "Timeline ménage",
         "Agrégat par (menage × programme) : premier score, dernier, "
         "min, max, trajectoire, éligibilité courante.", "menage_timeline"),
        ("menage_trajectory","derive", "Trajectoire complète",
         "Vie entière de chaque ménage : inscription, mises à jour, "
         "radiation éventuelle, delta total.", "menage_trajectory"),
        ("churn_timeline",   "derive", "Churn mensuel",
         "Éligibilité gap-fillée par (ménage × programme × mois). "
         "Pool, entrées, sorties, taux de churn.", "churn_timeline"),
        ("delta_frame",      "derive", "Ré-entrées",
         "Cycles perte → récupération d'éligibilité par ménage × programme. "
         "n_reentries par ménage.", "reentry_detail"),
    ]

    def _derived_card(name, tag_cls, title, desc, stem):
        lines, _ = _file_meta(stem)
        n_fmt = f"{int(lines):,} lignes" if lines else ""
        return (
            f"<div class='ov-derived-card'>"
            f"<div class='ov-proc-tag {tag_cls}' style='margin-bottom:6px'>"
            f"{name}</div>"
            f"<div style='font-size:0.82rem;font-weight:600;color:#111827;"
            f"margin-bottom:4px'>{title}</div>"
            f"<div style='font-size:0.76rem;color:#6b7280;line-height:1.5'>{desc}</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
            f"color:#9ca3af;margin-top:6px'>{n_fmt}</div>"
            f"</div>"
        )

    cards_html = "<div class='ov-derived-grid'>"
    for name, tag_cls, title, desc, stem in DERIVED:
        cards_html += _derived_card(name, tag_cls, title, desc, stem)
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── QC expander ───────────────────────────────────────────────────────────
    if not df_qc.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Rapport QC complet · Full QC report"):
            qcols = ["label_fr", "before", "after", "dropped", "pct_dropped", "note"]
            labels = {
                "label_fr": "Métrique",
                "before": "Avant nettoyage",
                "after": "Après nettoyage",
                "dropped": "Supprimés",
                "pct_dropped": "% supprimés",
                "note": "Explication",
            }
            avail = [c for c in qcols if c in df_qc.columns]
            tbl = df_qc[df_qc["metric"] != "scores_period"][avail].copy()
            tbl = tbl.rename(columns={c: labels[c] for c in avail})
            st.dataframe(tbl, width='stretch', hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Démographie":
    ptitle("Démographie", "Demographics")

    # ── dynamic summary banner ───────────────────────────────────────────────
    try:
        # milieu
        milieu_top = ""
        milieu_pct = ""
        if "milieu" in agg and not agg["milieu"].empty:
            df_m = agg["milieu"].sort_values("Ménages", ascending=False)
            milieu_top = str(df_m.iloc[0]["Milieu"]).lower()
            total_m = df_m["Ménages"].sum()
            pct_m   = df_m.iloc[0]["Ménages"] / total_m * 100 if total_m else 0
            milieu_pct = f"{pct_m:.0f}%"

        # genre
        genre_top = ""
        genre_pct = ""
        if "genre" in agg and not agg["genre"].empty:
            df_g = agg["genre"].sort_values("Ménages", ascending=False)
            genre_top = str(df_g.iloc[0]["Genre"]).lower()
            total_g = df_g["Ménages"].sum()
            pct_g   = df_g.iloc[0]["Ménages"] / total_g * 100 if total_g else 0
            genre_pct = f"{pct_g:.0f}%"

        # type de ménage
        type_top = ""
        if "type_menage" in agg and not agg["type_menage"].empty:
            df_t = agg["type_menage"].sort_values("Ménages", ascending=False)
            type_top = str(df_t.iloc[0]["Type"]).replace("_", " ").lower()

        # taille moyenne
        taille_avg = ""
        if "taille" in agg and not agg["taille"].empty:
            df_ta = agg["taille"]
            total_ta = df_ta["Ménages"].sum()
            if total_ta > 0:
                wmean = (df_ta["Personnes"] * df_ta["Ménages"]).sum() / total_ta
                taille_avg = f"{wmean:.1f}"

        # top regions covering ~70%
        top_regions_str = ""
        if "region_dem" in agg and not agg["region_dem"].empty:
            df_r = agg["region_dem"].sort_values("Ménages", ascending=False).reset_index(drop=True)
            total_r = df_r["Ménages"].sum()
            cumsum = 0
            regions = []
            for _, row in df_r.iterrows():
                regions.append(str(row["Région"]))
                cumsum += row["Ménages"]
                if cumsum / total_r >= 0.70:
                    break
            pct_covered = cumsum / total_r * 100 if total_r else 0
            if len(regions) <= 3:
                top_regions_str = " et ".join(regions)
            else:
                top_regions_str = ", ".join(regions[:-1]) + " et " + regions[-1]
            top_regions_str += f" ({pct_covered:.0f}% des ménages)"

        # top province
        top_province = ""
        if "top_provinces" in agg and not agg["top_provinces"].empty:
            df_p = agg["top_provinces"].sort_values("Ménages", ascending=False)
            top_province = str(df_p.iloc[0]["Province"]).capitalize()

        # build sentence
        parts = []
        if genre_top and genre_pct:
            parts.append(f"Les chefs de ménage sont majoritairement des <strong>{genre_top}s ({genre_pct})</strong>")
        if milieu_top and milieu_pct:
            parts.append(f"le milieu <strong>{milieu_top} ({milieu_pct})</strong> est légèrement dominant")
        if type_top:
            parts.append(f"le type de ménage le plus fréquent est le <strong>{type_top}</strong>")
        if taille_avg:
            parts.append(f"avec une taille moyenne de <strong>{taille_avg} personnes</strong>")
        if top_regions_str:
            parts.append(f"la concentration géographique se situe dans {top_regions_str}")
        if top_province:
            parts.append(f"la province la plus représentée étant <strong>{top_province}</strong>")

        if parts:
            sentence = " · ".join(parts) + "."
            st.markdown(
                f"<div style='background:#f7f8fa;border-left:3px solid #1a56db;"
                f"border-radius:0 6px 6px 0;padding:12px 18px;margin-bottom:16px;"
                f"font-size:0.92rem;color:#374151;line-height:1.7'>{sentence}</div>",
                unsafe_allow_html=True)
    except Exception:
        pass

    sec("Milieu, genre et situation matrimoniale",
        "Area, gender and marital status")
    c1, c2, c3 = st.columns(3)
    if "milieu" in agg:
        with c1:
            fig = px.pie(agg["milieu"], names="Milieu", values="Ménages", hole=0.55,
                         color_discrete_sequence=["#1a56db", "#0e9f6e"],
                         title="Milieu de résidence · Rural / Urbain")
            fig.update_layout(**PL, height=260)
            fig.update_traces(textinfo="percent+label", textfont_size=12)
            st.plotly_chart(fig, width='stretch')
    if "genre" in agg:
        with c2:
            fig = px.pie(agg["genre"], names="Genre", values="Ménages", hole=0.55,
                         color_discrete_sequence=["#1a56db", "#d97706"],
                         title="Genre du chef de ménage")
            fig.update_layout(**PL, height=260)
            fig.update_traces(textinfo="percent+label", textfont_size=12)
            st.plotly_chart(fig, width='stretch')
    if "matrimonial" in agg:
        with c3:
            fig = px.bar(agg["matrimonial"], x="Ménages", y="État",
                         orientation="h", color_discrete_sequence=["#1a56db"],
                         title="Situation matrimoniale")
            fig.update_layout(**PL, height=260,
                              xaxis_title="Ménages", yaxis_title="")
            st.plotly_chart(fig, width='stretch')

    sec("Répartition par région & Top 20 provinces", "Regional distribution & Top 20 provinces")
    col_reg, col_prov = st.columns(2)
    if "region_dem" in agg:
        with col_reg:
            fig = px.bar(agg["region_dem"], x="Ménages", y="Région",
                         orientation="h",
                         color="Ménages",
                         color_continuous_scale=["#bfdbfe", "#1a56db"],
                         title="Répartition par région")
            fig.update_layout(**PL, height=340, coloraxis_showscale=False,
                              xaxis_title="Ménages", yaxis_title="")
            st.plotly_chart(fig, width='stretch')
    if "top_provinces" in agg:
        with col_prov:
            fig = px.bar(agg["top_provinces"], x="Ménages", y="Province",
                         orientation="h",
                         color_discrete_sequence=["#d97706"],
                         title="Top 20 provinces")
            fig.update_layout(**PL, height=440,
                              xaxis_title="Ménages", yaxis_title="")
            st.plotly_chart(fig, width='stretch')

    sec("Type et taille du ménage", "Household type and size")
    c1, c2 = st.columns(2)
    if "type_menage" in agg:
        with c1:
            fig = px.bar(agg["type_menage"], x="Type", y="Ménages",
                         color_discrete_sequence=["#1a56db"],
                         title="Type de ménage")
            fig.update_layout(**PL, height=280, xaxis_title="", yaxis_title="Ménages",
                              xaxis_tickangle=-20)
            st.plotly_chart(fig, width='stretch')
    if "taille" in agg:
        with c2:
            fig = px.bar(agg["taille"], x="Personnes", y="Ménages",
                         color_discrete_sequence=["#0e9f6e"],
                         title="Taille du ménage (nombre de personnes)")
            fig.update_layout(**PL, height=280,
                              xaxis_title="Personnes dans le ménage",
                              yaxis_title="Ménages")
            st.plotly_chart(fig, width='stretch')




# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — SCORES & VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Scores et Volatilité":
    ptitle("Scores ISE & Volatilité", "ISE Scores & Volatility")

    # ── Seuils — prominent top cards ──────────────────────────────────────────
    st.markdown(
        "<div style='display:flex;gap:14px;margin-bottom:16px;flex-wrap:wrap'>"
        "  <div style='background:#1a56db;border-radius:10px;padding:14px 22px;"
        "display:flex;align-items:center;gap:16px;color:#fff;min-width:240px'>"
        "    <div style='font-family:IBM Plex Mono,monospace;font-size:2rem;"
        "font-weight:700;line-height:1;letter-spacing:-0.02em'>9.3264</div>"
        "    <div>"
        "      <div style='font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.12em;opacity:0.75;margin-bottom:3px'>Seuil éligibilité</div>"
        "      <div style='font-size:0.92rem;font-weight:600'>AMO Tadamon (AMOT)</div>"
        "    </div>"
        "  </div>"
        "  <div style='background:#0e9f6e;border-radius:10px;padding:14px 22px;"
        "display:flex;align-items:center;gap:16px;color:#fff;min-width:240px'>"
        "    <div style='font-family:IBM Plex Mono,monospace;font-size:2rem;"
        "font-weight:700;line-height:1;letter-spacing:-0.02em'>9.7430</div>"
        "    <div>"
        "      <div style='font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.12em;opacity:0.75;margin-bottom:3px'>Seuil éligibilité</div>"
        "      <div style='font-size:0.92rem;font-weight:600'>Aide Sociale Directe (ASD)</div>"
        "    </div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True)

    sec("Distribution des scores", "Score distribution")
    st.caption("Lignes pointillées = seuils d'éligibilité. "
               "Ménages à gauche du seuil = éligibles.")
    if "score_hist" in agg:
        fig = go.Figure()
        for prog, color in PROG_COLORS.items():
            sub = agg["score_hist"][agg["score_hist"]["programme"] == prog]
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                x=sub["bin"], y=sub["n"],
                name=PROG_FULL.get(prog, prog),
                marker_color=color, opacity=0.65,
                width=0.05,
            ))
        fig.add_vline(x=9.3264284, line_dash="dash", line_color="#1a56db",
                      annotation_text="Seuil AMOT 9.3264",
                      annotation_font_color="#1a56db")
        fig.add_vline(x=9.743001, line_dash="dash", line_color="#0e9f6e",
                      annotation_text="Seuil ASD 9.7430",
                      annotation_font_color="#0e9f6e")
        fig.update_layout(**PL, barmode="overlay", height=300,
                          xaxis_title="Score ISE",
                          yaxis_title="Nombre d'événements",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width='stretch')

    sec("Volatilité ΔISE", "ΔISE Volatility")
    st.caption(
        "**ΔISE** = variation du score entre deux événements consécutifs. "
        "Négatif = amélioration · Positif = dégradation.")

    # compact side-by-side stats + charts
    col_stats, col_vol, col_heatmap = st.columns([1, 1.2, 1.4])
    with col_stats:
        st.markdown(
            f"<div style='border-top:2px solid #111827;padding-top:12px;display:flex;"
            f"flex-direction:column;gap:18px'>"
            f"  <div class='metric-card'>"
            f"    <div class='metric-label'>Variation moyenne · Mean ΔISE</div>"
            f"    <div class='metric-value'>{agg.get('delta_mean','—')}</div>"
            f"    <div class='metric-sub'>Négatif = amélioration globale</div>"
            f"  </div>"
            f"  <div class='metric-card'>"
            f"    <div class='metric-label'>Écart-type · Std deviation</div>"
            f"    <div class='metric-value'>{agg.get('delta_std','—')}</div>"
            f"    <div class='metric-sub'>Instabilité globale des scores</div>"
            f"  </div>"
            f"  <div class='metric-card'>"
            f"    <div class='metric-label'>p90 |ΔISE| · 90th pct</div>"
            f"    <div class='metric-value'>{agg.get('delta_p90','—')}</div>"
            f"    <div class='metric-sub'>90% des ménages varient moins</div>"
            f"  </div>"
            f"  <div class='metric-card'>"
            f"    <div class='metric-label'>p99 |ΔISE| · 99th pct</div>"
            f"    <div class='metric-value'>{agg.get('delta_p99','—')}</div>"
            f"    <div class='metric-sub'>Variations extrêmes</div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True)

    if "vol_prog" in agg:
        with col_vol:
            fig = px.bar(agg["vol_prog"], x="programme", y="sigma_ΔISE",
                         color="programme", color_discrete_map=PROG_COLORS,
                         text="sigma_ΔISE",
                         title="Instabilité σ ΔISE par programme")
            # colour programme labels on x-axis
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(**PL, showlegend=False, height=280,
                              xaxis_title="", yaxis_title="Écart-type ΔISE")
            st.plotly_chart(fig, width='stretch')

    if "vol_heatmap" in agg:
        with col_heatmap:
            fig = px.imshow(agg["vol_heatmap"].round(4),
                            color_continuous_scale=["#dc2626", "#f9fafb", "#15803d"],
                            color_continuous_midpoint=0, aspect="auto",
                            title="ΔISE moyen — région × programme")
            fig.update_layout(**PL, height=280, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width='stretch')

    # ── stats table at bottom ─────────────────────────────────────────────────
    sec("Statistiques de score par programme", "Score statistics by programme")
    if "score_stats" in agg:
        def _colour_prog_col(df):
            styled = df.style
            def _highlight_prog(val):
                color_map = {"AMOT": "#1a56db", "ASD": "#0e9f6e",
                             "AMOA": "#d97706", "Non classifié": "#9ca3af"}
                c = color_map.get(val, "")
                return f"color:{c};font-weight:600" if c else ""
            styled = styled.applymap(_highlight_prog, subset=["Programme"])
            return styled
        try:
            st.dataframe(_colour_prog_col(agg["score_stats"]),
                         width='stretch', hide_index=True)
        except Exception:
            st.dataframe(agg["score_stats"], width='stretch', hide_index=True)




# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — TIME SERIES & TRENDS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Tendances Temporelles":
    ptitle("Tendances Temporelles", "Time Series & Trends")


    sec("Évolution des scores — Tous les ménages", "Overall score trends")

    if "score_ts_overall" in agg:
        df_ts_overall = agg["score_ts_overall"]
        if not df_ts_overall.empty:
            trend = (df_ts_overall.iloc[-1]['score_mean'] - df_ts_overall.iloc[0]['score_mean'])
            cls   = "good" if trend < 0 else "warn"
            cs1, cs2, cs3 = st.columns(3)
            with cs1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Moyenne (début) · Start mean</div>"
                    f"<div class='metric-value'>{df_ts_overall.iloc[0]['score_mean']:.4f}</div></div>",
                    unsafe_allow_html=True)
            with cs2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Moyenne (fin) · End mean</div>"
                    f"<div class='metric-value'>{df_ts_overall.iloc[-1]['score_mean']:.4f}</div></div>",
                    unsafe_allow_html=True)
            with cs3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Variation totale · Total drift</div>"
                    f"<div class='metric-value {cls}'>{trend:+.4f}</div>"
                    f"<div class='metric-sub'>Négatif = amélioration globale</div></div>",
                    unsafe_allow_html=True)

    if "score_ts_overall" in agg:
        df_ts_overall = agg["score_ts_overall"]
        if not df_ts_overall.empty and "score_std" in df_ts_overall.columns:
            mean_arr = df_ts_overall["score_mean"]
            std_arr  = df_ts_overall["score_std"]
            dates    = df_ts_overall["date_calcul"]
            has_pop  = "n_menages" in df_ts_overall.columns

            # ── Percentile selector ───────────────────────────────────────────
            PERCENTILE_OPTIONS = {
                "p10": ("score_p10", "p10", "#a855f7", "dot"),
                "p25": ("score_p25", "p25", "#06b6d4", "dashdot"),
                "p75": ("score_p75", "p75", "#f97316", "dashdot"),
                "p90": ("score_p90", "p90", "#ec4899", "dot"),
            }
            available_pcts = [k for k in PERCENTILE_OPTIONS if PERCENTILE_OPTIONS[k][0] in df_ts_overall.columns]
            selected_pcts  = st.multiselect(
                "Déciles à afficher sur le graphique · Percentiles to show",
                options=available_pcts,
                default=[],
                format_func=lambda k: PERCENTILE_OPTIONS[k][1],
                key="ts_overall_pct_sel",
            )

            from plotly.subplots import make_subplots as _make_subplots
            fig_band = _make_subplots(specs=[[{"secondary_y": True}]])

            fig_band.add_trace(go.Scatter(
                x=dates, y=(mean_arr - std_arr).round(6),
                mode="lines", name="Moyenne − 1σ",
                line=dict(color="rgba(26,86,219,0)"),
                hoverinfo="skip", showlegend=False,
            ), secondary_y=False)
            fig_band.add_trace(go.Scatter(
                x=dates, y=(mean_arr + std_arr).round(6),
                mode="lines", name="Bande ±1 σ",
                line=dict(color="rgba(26,86,219,0)"),
                fill="tonexty", fillcolor="rgba(26,86,219,0.18)",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>+1σ : %{y:.4f}<extra></extra>",
            ), secondary_y=False)
            fig_band.add_trace(go.Scatter(
                x=dates, y=mean_arr.round(6),
                mode="lines", name="Moyenne ISE",
                line=dict(color="#1a56db", width=2.5),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Moyenne : %{y:.4f}<extra></extra>",
            ), secondary_y=False)
            if "score_median" in df_ts_overall.columns:
                fig_band.add_trace(go.Scatter(
                    x=dates, y=df_ts_overall["score_median"].round(6),
                    mode="lines", name="Médiane",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Médiane : %{y:.4f}<extra></extra>",
                ), secondary_y=False)

            # ── Selected percentile lines ─────────────────────────────────────
            for pct_key in selected_pcts:
                col_name, label, color, dash = PERCENTILE_OPTIONS[pct_key]
                fig_band.add_trace(go.Scatter(
                    x=dates, y=df_ts_overall[col_name].round(6),
                    mode="lines", name=label,
                    line=dict(color=color, width=1.8, dash=dash),
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label} : %{{y:.4f}}<extra></extra>",
                ), secondary_y=False)

            if has_pop:
                fig_band.add_trace(go.Scatter(
                    x=dates, y=df_ts_overall["n_menages"],
                    mode="lines", name="Ménages actifs (échantillon)",
                    line=dict(color="rgba(107,114,128,0.55)", width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ménages actifs : %{y:,.0f}<extra></extra>",
                ), secondary_y=True)
            fig_band.add_hline(y=9.3264284,
                line=dict(color="#1a56db", width=1.2, dash="dash"),
                annotation_text="Seuil AMOT 9.3264",
                annotation_position="top right",
                annotation_font=dict(color="#1a56db", size=10))
            fig_band.add_hline(y=9.743001,
                line=dict(color="#0e9f6e", width=1.2, dash="dash"),
                annotation_text="Seuil ASD 9.7430",
                annotation_position="top right",
                annotation_font=dict(color="#0e9f6e", size=10))
            fig_band.update_layout(
                **PL, height=320,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
                title=dict(text="Évolution de la moyenne ± 1σ — tous ménages",
                           font=dict(size=12, color="#6b7280"), x=0),
                xaxis_title="Date",
            )
            fig_band.update_yaxes(title_text="Score ISE", secondary_y=False)
            if has_pop:
                fig_band.update_yaxes(
                    title_text="Ménages actifs",
                    secondary_y=True, showgrid=False,
                    tickfont=dict(color="rgba(107,114,128,0.7)"),
                    title_font=dict(color="rgba(107,114,128,0.7)"),
                )
            st.plotly_chart(fig_band, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Combined milieu + genre with selector ──────────────────────────────────
    sec("Comparaison par segment", "By segment (milieu / genre)")

    has_milieu = "score_ts_by_milieu" in agg and not agg["score_ts_by_milieu"].empty
    has_genre  = "score_ts_by_genre"  in agg and not agg["score_ts_by_genre"].empty

    seg_options = []
    if has_milieu: seg_options.append("Milieu (Urbain/Rural)")
    if has_genre:  seg_options.append("Genre (Homme/Femme)")

    if seg_options:
        cf1, cf2 = st.columns([1, 3])
        with cf1:
            seg_choice = st.selectbox("Segmentation", seg_options, key="seg_choice")
            if has_milieu and "Milieu" in seg_choice:
                milieu_opts = sorted(agg["score_ts_by_milieu"]["milieu"].unique())
                seg_filter  = st.multiselect("Filtrer", milieu_opts, default=milieu_opts, key="milieu_filter")
            elif has_genre and "Genre" in seg_choice:
                genre_opts = sorted(agg["score_ts_by_genre"]["genre_cm"].unique())
                seg_filter = st.multiselect("Filtrer", genre_opts, default=genre_opts, key="genre_filter")
            else:
                seg_filter = []

            # ── Percentile selector for segment chart ─────────────────────────
            SEG_PCT_OPTIONS = {
                "p10": ("score_p10", "p10", "#a855f7", "dot"),
                "p25": ("score_p25", "p25", "#06b6d4", "dashdot"),
                "p75": ("score_p75", "p75", "#f97316", "dashdot"),
                "p90": ("score_p90", "p90", "#ec4899", "dot"),
            }
            seg_df_ref = (agg.get("score_ts_by_milieu") if has_milieu and "Milieu" in seg_choice
                          else agg.get("score_ts_by_genre", pd.DataFrame()))
            if seg_df_ref is not None and not seg_df_ref.empty:
                avail_seg_pcts = [k for k in SEG_PCT_OPTIONS if SEG_PCT_OPTIONS[k][0] in seg_df_ref.columns]
            else:
                avail_seg_pcts = []
            selected_seg_pcts = st.multiselect(
                "Déciles · Percentiles",
                options=avail_seg_pcts,
                default=[],
                format_func=lambda k: SEG_PCT_OPTIONS[k][1],
                key="ts_seg_pct_sel",
            ) if avail_seg_pcts else []

        with cf2:
            fig_seg = go.Figure()
            if "Milieu" in seg_choice and has_milieu:
                milieu_colors = {"Urbain": "#1a56db", "Rural": "#d97706"}
                df_seg = agg["score_ts_by_milieu"]
                for seg in seg_filter:
                    sub   = df_seg[df_seg["milieu"] == seg].copy()
                    color = milieu_colors.get(seg, "#888888")
                    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                    mean_arr = sub["score_mean"]
                    std_arr  = sub["score_std"] if "score_std" in sub.columns else None
                    dates    = sub["date_calcul"]
                    if std_arr is not None:
                        fig_seg.add_trace(go.Scatter(x=dates, y=(mean_arr-std_arr).round(6),
                            mode="lines", line=dict(color=f"rgba({r},{g},{b},0)"),
                            hoverinfo="skip", showlegend=False))
                        fig_seg.add_trace(go.Scatter(x=dates, y=(mean_arr+std_arr).round(6),
                            mode="lines", line=dict(color=f"rgba({r},{g},{b},0)"),
                            fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.15)",
                            showlegend=False,
                            hovertemplate=f"<b>{seg}</b><br>+1σ : %{{y:.4f}}<extra></extra>"))
                    fig_seg.add_trace(go.Scatter(x=dates, y=mean_arr.round(6),
                        mode="lines", name=seg, line=dict(color=color, width=2.5),
                        hovertemplate=f"<b>{seg}</b> %{{x|%Y-%m-%d}}<br>Moy: %{{y:.4f}}<extra></extra>"))
                    # percentile lines for this segment
                    for pct_key in selected_seg_pcts:
                        col_name, label, pct_color, dash = SEG_PCT_OPTIONS[pct_key]
                        if col_name in sub.columns:
                            fig_seg.add_trace(go.Scatter(
                                x=dates, y=sub[col_name].round(6),
                                mode="lines", name=f"{seg} {label}",
                                line=dict(color=pct_color, width=1.5, dash=dash),
                                hovertemplate=f"<b>{seg} {label}</b> %{{x|%Y-%m-%d}}<br>{label}: %{{y:.4f}}<extra></extra>",
                            ))
            elif "Genre" in seg_choice and has_genre:
                genre_colors = {"Femme": "#e3342f", "Homme": "#1a56db"}
                df_seg = agg["score_ts_by_genre"]
                for seg in seg_filter:
                    sub   = df_seg[df_seg["genre_cm"] == seg].copy()
                    color = genre_colors.get(seg, "#888888")
                    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                    mean_arr = sub["score_mean"]
                    std_arr  = sub["score_std"] if "score_std" in sub.columns else None
                    dates    = sub["date_calcul"]
                    if std_arr is not None:
                        fig_seg.add_trace(go.Scatter(x=dates, y=(mean_arr-std_arr).round(6),
                            mode="lines", line=dict(color=f"rgba({r},{g},{b},0)"),
                            hoverinfo="skip", showlegend=False))
                        fig_seg.add_trace(go.Scatter(x=dates, y=(mean_arr+std_arr).round(6),
                            mode="lines", line=dict(color=f"rgba({r},{g},{b},0)"),
                            fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.15)",
                            showlegend=False,
                            hovertemplate=f"<b>{seg}</b><br>+1σ : %{{y:.4f}}<extra></extra>"))
                    fig_seg.add_trace(go.Scatter(x=dates, y=mean_arr.round(6),
                        mode="lines", name=seg, line=dict(color=color, width=2.5),
                        hovertemplate=f"<b>{seg}</b> %{{x|%Y-%m-%d}}<br>Moy: %{{y:.4f}}<extra></extra>"))
                    # percentile lines for this segment
                    for pct_key in selected_seg_pcts:
                        col_name, label, pct_color, dash = SEG_PCT_OPTIONS[pct_key]
                        if col_name in sub.columns:
                            fig_seg.add_trace(go.Scatter(
                                x=dates, y=sub[col_name].round(6),
                                mode="lines", name=f"{seg} {label}",
                                line=dict(color=pct_color, width=1.5, dash=dash),
                                hovertemplate=f"<b>{seg} {label}</b> %{{x|%Y-%m-%d}}<br>{label}: %{{y:.4f}}<extra></extra>",
                            ))

            fig_seg.add_hline(y=9.3264284, line=dict(color="#1a56db", width=1, dash="dash"),
                annotation_text="Seuil AMOT", annotation_font=dict(color="#1a56db", size=10))
            fig_seg.add_hline(y=9.743001, line=dict(color="#0e9f6e", width=1, dash="dash"),
                annotation_text="Seuil ASD", annotation_font=dict(color="#0e9f6e", size=10))
            fig_seg.update_layout(**PL, height=300,
                xaxis_title="Date", yaxis_title="Score ISE moyen",
                hovermode="x unified", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_seg, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — SIMULATEUR DE SEUIL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Simulateur de Seuil":
    ptitle(
        "Simulateur de Seuil",
        "Ménages proches du seuil d'éligibilité — évolution temporelle par programme",  
    )

    df_nt = agg.get("near_threshold_ts", pd.DataFrame())

    if df_nt is None or df_nt.empty:
        st.warning(
            "Données de proximité au seuil non disponibles. "
            "Exécutez `prebake_dashboard.py` après la dernière mise à jour du pipeline."
        )
        st.stop()

    df_nt = df_nt.copy()
    df_nt["week"] = pd.to_datetime(df_nt["week"])

    # ── Band definitions ──────────────────────────────────────────────────────
    # Each band has three views: net (|dist|≤ε), neg (eligible side), pos (ineligible side)
    BANDS = {
        "±0.10": {
            "n_col": "n_near_010", "mean_col": "mean_score_010", "median_col": "median_score_010",
            "n_neg": "n_near_010_neg", "mean_neg": "mean_score_010_neg", "median_neg": "median_score_010_neg",
            "n_pos": "n_near_010_pos", "mean_pos": "mean_score_010_pos", "median_pos": "median_score_010_pos",
            "color": "#1a56db", "color_rgb": (26,  86,  219),
        },
        "±0.25": {
            "n_col": "n_near_025", "mean_col": "mean_score_025", "median_col": "median_score_025",
            "n_neg": "n_near_025_neg", "mean_neg": "mean_score_025_neg", "median_neg": "median_score_025_neg",
            "n_pos": "n_near_025_pos", "mean_pos": "mean_score_025_pos", "median_pos": "median_score_025_pos",
            "color": "#d97706", "color_rgb": (217, 119,   6),
        },
        "±0.50": {
            "n_col": "n_near_050", "mean_col": "mean_score_050", "median_col": "median_score_050",
            "n_neg": "n_near_050_neg", "mean_neg": "mean_score_050_neg", "median_neg": "median_score_050_neg",
            "n_pos": "n_near_050_pos", "mean_pos": "mean_score_050_pos", "median_pos": "median_score_050_pos",
            "color": "#7c3aed", "color_rgb": (124,  58, 237),
        },
    }

    available_progs = sorted([p for p in df_nt["programme"].unique() if p != "Non classifié"])

    # ── Compact header: explanation + controls in two columns ─────────────────
    hdr_left, hdr_right = st.columns([2, 3])

    with hdr_left:
        st.markdown(
            "<div style='background:#f7f8fa;border:1px solid #e0e4ea;border-radius:8px;"
            "padding:10px 16px;font-size:0.78rem;color:#374151;line-height:1.65'>"
            "<strong>Comment lire ces graphiques :</strong><br>"
            "Un ménage est <em>proche du seuil</em> si son écart au seuil (dist = score − seuil) "
            "est dans la bande ε. "
            "<span style='color:#15803d;font-weight:600'>Côté éligible (dist &lt; 0)</span> : "
            "ménage sous le seuil, actuellement éligible, risque de le dépasser. "
            "<span style='color:#dc2626;font-weight:600'>Côté inéligible (dist &gt; 0)</span> : "
            "ménage au-dessus du seuil, proche de devenir éligible. "
            "Le <strong>net</strong> combine les deux côtés (|dist| ≤ ε)."
            "</div>",
            unsafe_allow_html=True,
        )

    with hdr_right:
        st.markdown(
            "<div style='background:#f7f8fa;border:1px solid #e0e4ea;border-radius:8px;"
            "padding:8px 14px'>",
            unsafe_allow_html=True,
        )
        rc1, rc2, rc3, rc4, rc5 = st.columns([1.6, 1, 1, 1.1, 1])
        with rc1:
            sel_prog = st.selectbox(
                "Programme",
                options=available_progs,
                format_func=lambda p: PROG_FULL.get(p, p),
                key="sim_prog",
            )
        with rc2:
            b010 = st.checkbox("±0.10", value=True,  key="sim_b010")
            b025 = st.checkbox("±0.25", value=True,  key="sim_b025")
            b050 = st.checkbox("±0.50", value=False, key="sim_b050")
        with rc3:
            show_mean   = st.checkbox("Moyenne",  value=True, key="sim_mean")
            show_median = st.checkbox("Médiane",  value=True, key="sim_median")
        with rc4:
            show_total = st.checkbox("N total (axe 2)", value=True,  key="sim_total")
            show_pct   = st.checkbox("Graphe % part",   value=False, key="sim_pct")
        with rc5:
            show_thresh = st.checkbox("Seuil",     value=True,  key="sim_thresh")
            show_split  = st.checkbox("Côtés ±",   value=True,  key="sim_split")
        st.markdown("</div>", unsafe_allow_html=True)

    selected_bands = [b for b, flag in [("±0.10", b010), ("±0.25", b025), ("±0.50", b050)] if flag]

    # ── Filter to selected programme ──────────────────────────────────────────
    df_prog    = df_nt[df_nt["programme"] == sel_prog].copy().sort_values("week")
    prog_color = PROG_COLORS.get(sel_prog, "#1a56db")
    prog_thr   = THRESHOLDS.get(sel_prog)

    # ── KPI strip (latest week) ───────────────────────────────────────────────
    if not df_prog.empty:
        latest_p  = df_prog.iloc[-1]
        n_total_l = int(latest_p.get("n_total", 0))
        kpi_cols  = st.columns(7)
        kpi_items = [
            ("N total actifs",    "n_total",       "#374151", "Ménages actifs cette semaine"),
            ("Net ±0.10",         "n_near_010",    "#1a56db", "Éligible + inéligible proches"),
            ("Éligibles ±0.10",   "n_near_010_neg","#15803d", "Sous le seuil, risque de sortir"),
            ("Inélig. ±0.10",     "n_near_010_pos","#dc2626", "Au-dessus, risque d'entrer"),
            ("Net ±0.25",         "n_near_025",    "#d97706", "Zone standard ±0.25"),
            ("Éligibles ±0.25",   "n_near_025_neg","#15803d", "Sous le seuil ±0.25"),
            ("Inélig. ±0.25",     "n_near_025_pos","#dc2626", "Au-dessus ±0.25"),
        ]
        for col_obj, (lbl, key, color, sub) in zip(kpi_cols, kpi_items):
            val = int(latest_p.get(key, 0)) if key in latest_p.index else 0
            with col_obj:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label' style='font-size:0.60rem'>{lbl}</div>"
                    f"<div class='metric-value' style='color:{color};font-size:1.25rem'>{fmt(val)}</div>"
                    f"<div class='metric-sub' style='font-size:0.60rem'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    from plotly.subplots import make_subplots as _msp_sim

    # ── Rows of charts: one row per selected band ─────────────────────────────
    # Each row: [Count chart (net + neg + pos)] | [Score mean/median chart]
    for band_key in selected_bands:
        cfg   = BANDS[band_key]
        color = cfg["color"]
        r, g, b_rgb = cfg["color_rgb"]

        sec(
            f"Bande {band_key} — {sel_prog}",
            f"Band {band_key} · eligible side (green) · ineligible side (red) · net (band color)",
        )

        col_left, col_right = st.columns(2)

        # ── Left: count chart (net + neg + pos) ──────────────────────────────
        with col_left:
            st.caption(
                f"<span style='color:{color};font-weight:600'>Net {band_key}</span> : |dist| ≤ {band_key[1:]} · "
                "<span style='color:#15803d;font-weight:600'>Côté éligible</span> : dist &lt; 0 (sous le seuil) · "
                "<span style='color:#dc2626;font-weight:600'>Côté inéligible</span> : dist &gt; 0 (au-dessus)",
                unsafe_allow_html=True,
            )
            fig_c = _msp_sim(specs=[[{"secondary_y": True}]])

            # Net trace (fill)
            n_col = cfg["n_col"]
            if n_col in df_prog.columns:
                fig_c.add_trace(go.Scatter(
                    x=df_prog["week"], y=df_prog[n_col],
                    mode="lines", name=f"Net {band_key}",
                    line=dict(color=color, width=2.5),
                    fill="tozeroy", fillcolor=f"rgba({r},{g},{b_rgb},0.07)",
                    hovertemplate=f"Net : %{{y:,}}<extra>Net {band_key}</extra>",
                ), secondary_y=False)

            # Eligible side (neg) — green
            n_neg = cfg["n_neg"]
            if show_split and n_neg in df_prog.columns:
                fig_c.add_trace(go.Scatter(
                    x=df_prog["week"], y=df_prog[n_neg],
                    mode="lines", name=f"Éligibles {band_key}",
                    line=dict(color="#15803d", width=2, dash="dot"),
                    hovertemplate=f"Éligibles : %{{y:,}}<extra>Côté éligible {band_key}</extra>",
                ), secondary_y=False)

            # Ineligible side (pos) — red
            n_pos = cfg["n_pos"]
            if show_split and n_pos in df_prog.columns:
                fig_c.add_trace(go.Scatter(
                    x=df_prog["week"], y=df_prog[n_pos],
                    mode="lines", name=f"Inéligibles {band_key}",
                    line=dict(color="#dc2626", width=2, dash="dash"),
                    hovertemplate=f"Inéligibles : %{{y:,}}<extra>Côté inéligible {band_key}</extra>",
                ), secondary_y=False)

            if show_total and "n_total" in df_prog.columns:
                fig_c.add_trace(go.Scatter(
                    x=df_prog["week"], y=df_prog["n_total"],
                    mode="lines", name="N total actifs",
                    line=dict(color="rgba(107,114,128,0.35)", width=1.2, dash="dot"),
                    hovertemplate="Total actifs : %{y:,}<extra></extra>",
                ), secondary_y=True)

            fig_c.update_layout(
                **PL, height=280, hovermode="x unified",
                xaxis_title="Semaine",
                legend=dict(orientation="h", y=1.14, font=dict(size=9)),
                title=dict(text=f"Ménages dans la zone {band_key} — {sel_prog}",
                           font=dict(size=11, color="#6b7280"), x=0),
            )
            fig_c.update_yaxes(title_text="Ménages", secondary_y=False)
            if show_total:
                fig_c.update_yaxes(
                    title_text="N total", secondary_y=True, showgrid=False,
                    tickfont=dict(color="rgba(107,114,128,0.5)"),
                    title_font=dict(color="rgba(107,114,128,0.5)"),
                )
            st.plotly_chart(fig_c, width="stretch")

        # ── Right: score mean/median (net + neg + pos) ────────────────────────
        with col_right:
            st.caption(
                "Score moyen (ligne pleine) et médian (pointillés) pour chaque sous-groupe. "
                "La ligne de seuil montre la frontière d'éligibilité.",
            )
            fig_s = go.Figure()

            # Net
            mean_col   = cfg["mean_col"]
            median_col = cfg["median_col"]
            if show_mean and mean_col in df_prog.columns:
                fig_s.add_trace(go.Scatter(
                    x=df_prog["week"], y=df_prog[mean_col],
                    mode="lines", name=f"Moy. net {band_key}",
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"Moy. net : %{{y:.4f}}<extra>{band_key}</extra>",
                ))
            if show_median and median_col in df_prog.columns:
                fig_s.add_trace(go.Scatter(
                    x=df_prog["week"], y=df_prog[median_col],
                    mode="lines", name=f"Méd. net {band_key}",
                    line=dict(color=f"rgba({r},{g},{b_rgb},0.65)", width=1.8, dash="dot"),
                    hovertemplate=f"Méd. net : %{{y:.4f}}<extra>{band_key}</extra>",
                ))

            # Eligible side (neg) — green shades
            if show_split:
                mean_neg   = cfg["mean_neg"]
                median_neg = cfg["median_neg"]
                if show_mean and mean_neg in df_prog.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_prog["week"], y=df_prog[mean_neg],
                        mode="lines", name="Moy. éligibles",
                        line=dict(color="#15803d", width=2),
                        hovertemplate="Moy. éligibles : %{y:.4f}<extra></extra>",
                    ))
                if show_median and median_neg in df_prog.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_prog["week"], y=df_prog[median_neg],
                        mode="lines", name="Méd. éligibles",
                        line=dict(color="#15803d", width=1.5, dash="dot"),
                        hovertemplate="Méd. éligibles : %{y:.4f}<extra></extra>",
                    ))

                # Ineligible side (pos) — red shades
                mean_pos   = cfg["mean_pos"]
                median_pos = cfg["median_pos"]
                if show_mean and mean_pos in df_prog.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_prog["week"], y=df_prog[mean_pos],
                        mode="lines", name="Moy. inéligibles",
                        line=dict(color="#dc2626", width=2),
                        hovertemplate="Moy. inéligibles : %{y:.4f}<extra></extra>",
                    ))
                if show_median and median_pos in df_prog.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_prog["week"], y=df_prog[median_pos],
                        mode="lines", name="Méd. inéligibles",
                        line=dict(color="#dc2626", width=1.5, dash="dot"),
                        hovertemplate="Méd. inéligibles : %{y:.4f}<extra></extra>",
                    ))

            if show_thresh and prog_thr is not None:
                fig_s.add_hline(
                    y=prog_thr,
                    line=dict(color=prog_color, width=1.4, dash="dash"),
                    annotation_text=f"Seuil {sel_prog} {prog_thr:.4f}",
                    annotation_position="top right",
                    annotation_font=dict(color=prog_color, size=10),
                )

            fig_s.update_layout(
                **PL, height=280, hovermode="x unified",
                xaxis_title="Semaine", yaxis_title="Score ISE",
                legend=dict(orientation="h", y=1.14, font=dict(size=9)),
                title=dict(text=f"Score ISE — ménages dans la zone {band_key} · {sel_prog}",
                           font=dict(size=11, color="#6b7280"), x=0),
            )
            st.plotly_chart(fig_s, width="stretch")

    # ── Optional % part chart (full width) ────────────────────────────────────
    if show_pct and selected_bands:
        st.markdown("<br>", unsafe_allow_html=True)
        sec("Part des ménages proches / total (%)", "Share of near-threshold households")
        fig_pct = go.Figure()
        for band_key in selected_bands:
            cfg   = BANDS[band_key]
            color = cfg["color"]
            r2, g2, b2 = cfg["color_rgb"]
            for col_key, lbl, lcolor, ldash in [
                (cfg["n_col"], f"Net {band_key}",        color,     "solid"),
                (cfg["n_neg"], f"Éligibles {band_key}",  "#15803d", "dot"),
                (cfg["n_pos"], f"Inéligibles {band_key}","#dc2626", "dash"),
            ]:
                if not show_split and col_key != cfg["n_col"]:
                    continue
                if col_key not in df_prog.columns:
                    continue
                pct_s = (df_prog[col_key] / df_prog["n_total"].replace(0, float("nan")) * 100).round(2)
                fig_pct.add_trace(go.Scatter(
                    x=df_prog["week"], y=pct_s,
                    mode="lines+markers", name=lbl,
                    line=dict(color=lcolor, width=2, dash=ldash),
                    marker=dict(size=3),
                    hovertemplate=f"{lbl} : %{{y:.2f}}%<extra></extra>",
                ))
        fig_pct.update_layout(
            **PL, height=240, hovermode="x unified",
            xaxis_title="Semaine", yaxis_title="% du total actifs",
            legend=dict(orientation="h", y=1.14, font=dict(size=9)),
            title=dict(text=f"Part hebdomadaire dans la zone de proximité — {sel_prog}",
                       font=dict(size=11, color="#6b7280"), x=0),
        )
        st.plotly_chart(fig_pct, width="stretch")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Données brutes · Raw data"):
        tbl_nt = df_prog.copy()
        tbl_nt["week"] = tbl_nt["week"].dt.strftime("%Y-%m-%d")
        # Add % columns for net bands
        for bk, cfg in BANDS.items():
            for col in [cfg["n_col"], cfg["n_neg"], cfg["n_pos"]]:
                if col in tbl_nt.columns:
                    tbl_nt[f"pct_{col}"] = (
                        tbl_nt[col] / tbl_nt["n_total"].replace(0, float("nan")) * 100
                    ).round(2)
        # Only show columns that exist
        display_cols = [c for c in tbl_nt.columns if not c.startswith("mean_score") and not c.startswith("median_score")]
        st.dataframe(tbl_nt[display_cols], width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ELIGIBILITY & CHURN
# ─────────────────────────────────────────────────────────────────────────────
elif page == " Churn":
    ptitle(" Churn",
           "Accès aux programmes et rotation mensuelle des ménages éligibles")

    if "churn_timeline" in agg and not agg["churn_timeline"].empty:
        df_ct = agg["churn_timeline"].copy()
        df_ct["date"] = pd.to_datetime(df_ct["date"])
        programmes_ct = sorted([p for p in df_ct["programme"].unique()
                                 if p not in ["Non classifié", "AMOA"]])

        sec("Évolution mensuelle du churn par programme",
            "Monthly churn timeline by programme")
        st.caption(
            "**Taux de churn** = ménages ayant perdu l'éligibilité ce mois / "
            "pool éligible du mois précédent. "
            "**Taux d'acquisition** = ménages devenus éligibles / pool du mois précédent. "
            "**Taux net** = (entrées − sorties) / pool — positif = pool en croissance.")

        prog_filter = st.multiselect(
            "Programmes à afficher",
            options=programmes_ct,
            default=[p for p in ["AMOT", "ASD"] if p in programmes_ct],
            key="churn_prog_filter",
        )
        if not prog_filter:
            prog_filter = programmes_ct

        df_ct_filt = df_ct[df_ct["programme"].isin(prog_filter)]

        # ── Two charts side by side in one line ───────────────────────────────
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            fig = go.Figure()
            for prog in prog_filter:
                sub   = df_ct_filt[df_ct_filt["programme"] == prog]
                color = PROG_COLORS.get(prog, "#888888")
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=(sub["churn_rate"] * 100).round(2),
                    name=f"{prog} — Churn", mode="lines+markers",
                    line=dict(color=color, width=2.5), marker=dict(size=4),
                    hovertemplate=f"<b>{prog}</b><br><b>%{{x|%Y-%m}}</b><br>Churn : %{{y:.2f}}%<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=(sub["acquisition_rate"] * 100).round(2),
                    name=f"{prog} — Acquisition", mode="lines+markers",
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=4, symbol="diamond"),
                    hovertemplate=f"<b>{prog}</b><br><b>%{{x|%Y-%m}}</b><br>Acquisition : %{{y:.2f}}%<extra></extra>",
                ))
            fig.add_hline(y=0, line_color="#d1d5db", line_width=1)
            fig.update_layout(**PL, height=280, xaxis_title="Mois", yaxis_title="Taux (%)",
                hovermode="x unified", legend=dict(orientation="h", y=1.14, font=dict(size=10)),
                title=dict(text="Churn (—) & Acquisition (···)",
                           font=dict(size=12, color="#6b7280"), x=0))
            st.plotly_chart(fig, width='stretch')

        with ch_col2:
            fig2 = go.Figure()
            for prog in prog_filter:
                sub   = df_ct_filt[df_ct_filt["programme"] == prog]
                color = PROG_COLORS.get(prog, "#888888")
                net   = (sub["net_rate"] * 100).round(2)
                fig2.add_trace(go.Bar(
                    x=sub["date"], y=net, name=prog,
                    marker_color=["#15803d" if v >= 0 else "#dc2626" for v in net],
                    hovertemplate=f"<b>{prog}</b><br><b>%{{x|%Y-%m}}</b><br>Net : %{{y:+.2f}}%<extra></extra>",
                ))
            fig2.add_hline(y=0, line_color="#374151", line_width=1.2)
            fig2.update_layout(**PL, height=280, barmode="group",
                xaxis_title="Mois", yaxis_title="Taux net (%)", hovermode="x unified",
                legend=dict(orientation="h", y=1.14, font=dict(size=10)),
                title=dict(text="Taux net — vert = croissance · rouge = déclin",
                           font=dict(size=12, color="#6b7280"), x=0))
            st.plotly_chart(fig2, width='stretch')

        # ── Absolute volumes — 4 charts in a 2×2 grid ────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        sec("Volumes mensuels absolus", "Monthly absolute volumes")
        st.caption(
            "Nombre réel de ménages uniques ayant gagné ou perdu l'éligibilité "
            "chaque mois, et taille du pool éligible au début et en fin de mois.")

        vol_cols = st.columns(len(prog_filter)) if len(prog_filter) <= 3 else st.columns(3)
        for pi, prog in enumerate(prog_filter):
            sub   = df_ct_filt[df_ct_filt["programme"] == prog].copy()
            color = PROG_COLORS.get(prog, "#888888")
            with vol_cols[pi % len(vol_cols)]:
                fig_pool = go.Figure()
                fig_pool.add_trace(go.Scatter(x=sub["date"], y=sub["pool_start"],
                    name="Pool début", mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%Y-%m}</b><br>Pool début : %{y:,}<extra></extra>"))
                fig_pool.add_trace(go.Scatter(x=sub["date"], y=sub["pool_end"],
                    name="Pool fin", mode="lines+markers",
                    line=dict(color=color, width=2.5), marker=dict(size=4),
                    hovertemplate="<b>%{x|%Y-%m}</b><br>Pool fin : %{y:,}<extra></extra>"))
                fig_pool.update_layout(**PL, height=260,
                    xaxis_title="Mois", yaxis_title="Ménages éligibles", hovermode="x unified",
                    legend=dict(orientation="h", y=1.14, font=dict(size=10)),
                    title=dict(text=f"{prog} — Pool éligible",
                               font=dict(size=12, color="#6b7280"), x=0))
                st.plotly_chart(fig_pool, width='stretch')

    if "reentry_summary" in agg and not agg["reentry_summary"].empty:
        st.markdown("<br>", unsafe_allow_html=True)
        sec("Analyse des ré-entrées", "Re-entry analysis")
        st.caption(
            "Un ménage 're-entre' quand il perd l'éligibilité puis la regagne. "
            "**Toujours éligible** = n'a jamais perdu · "
            "**Perdu sans retour** = a perdu l'éligibilité et ne l'a plus recouvrée · "
            "**1, 2, 3+** = nombre d'oscillations autour du seuil.")

        def _reentry_color(lbl):
            if lbl == "toujours éligible": return "#15803d"
            if lbl == "perdu sans retour": return "#6b7280"
            if lbl == "1":                 return "#f59e0b"
            return "#ef4444"

        def _reentry_order(lbl):
            order = {"toujours éligible": 0, "perdu sans retour": 1}
            try:    return order.get(lbl, 2 + int(lbl))
            except: return 99

        df_re = agg["reentry_summary"].copy()
        programmes_re = sorted([p for p in df_re["programme"].unique()
                                 if p not in ["Non classifié", "AMOA"]])

        # ── Programme selector + legend in one row ───────────────────────────
        sel_hdr_l, sel_hdr_r = st.columns([1, 3])
        with sel_hdr_l:
            re_prog_sel = st.selectbox(
                "Programme",
                options=programmes_re,
                format_func=lambda p: PROG_FULL.get(p, p),
                key="reentry_prog_sel",
            )
        with sel_hdr_r:
            st.markdown(
                "<div style='display:flex;gap:20px;flex-wrap:wrap;font-size:0.75rem;"
                "color:#374151;padding-top:30px;align-items:center'>"
                "<span style='display:inline-flex;align-items:center;gap:5px'>"
                "<span style='width:10px;height:10px;background:#15803d;border-radius:2px;display:inline-block'></span>"
                "<strong>Toujours éligible</strong></span>"
                "<span style='display:inline-flex;align-items:center;gap:5px'>"
                "<span style='width:10px;height:10px;background:#6b7280;border-radius:2px;display:inline-block'></span>"
                "<strong>Perdu sans retour</strong></span>"
                "<span style='display:inline-flex;align-items:center;gap:5px'>"
                "<span style='width:10px;height:10px;background:#f59e0b;border-radius:2px;display:inline-block'></span>"
                "<strong>1 ré-entrée</strong></span>"
                "<span style='display:inline-flex;align-items:center;gap:5px'>"
                "<span style='width:10px;height:10px;background:#ef4444;border-radius:2px;display:inline-block'></span>"
                "<strong>2+ ré-entrées</strong></span>"
                "</div>",
                unsafe_allow_html=True)

        # ── Stats sidebar + chart for selected programme ──────────────────────
        if "reentry_stats" in agg and not agg["reentry_stats"].empty:
            re_stats_df = agg["reentry_stats"][
                agg["reentry_stats"]["programme"] == re_prog_sel
            ]
            if not re_stats_df.empty:
                row   = re_stats_df.iloc[0]
                color = PROG_COLORS.get(re_prog_sel, "#888")
                avg   = row["avg_reentrees_parmi_churners"]

                sub = df_re[df_re["programme"] == re_prog_sel].copy()

                def _label(x):
                    if x in ("toujours éligible", "perdu sans retour"): return x
                    try:
                        n = int(x)
                        return str(n) if n <= 4 else "5+"
                    except: return str(x)

                sub["label"] = sub["n_reentries"].apply(_label)
                sub_grouped = (sub.groupby("label", observed=True)
                               .agg(n_menages=("n_menages", "sum"),
                                    pct_menages=("pct_menages", "sum"))
                               .reset_index())
                sub_grouped["_ord"] = sub_grouped["label"].apply(_reentry_order)
                sub_grouped = sub_grouped.sort_values("_ord").drop(columns=["_ord"])

                stat_col, chart_col = st.columns([1, 3])

                with stat_col:
                    st.markdown(
                        f"<div style='display:flex;flex-direction:column;gap:14px;padding-top:8px'>"
                        f"  <div class='metric-card'>"
                        f"    <div class='metric-label'>Total ménages</div>"
                        f"    <div class='metric-value' style='font-size:1.25rem'>{fmt(row['total_menages'])}</div>"
                        f"  </div>"
                        f"  <div class='metric-card'>"
                        f"    <div class='metric-label'>Avec ≥1 ré-entrée</div>"
                        f"    <div class='metric-value warn' style='font-size:1.25rem'>{fmt(row['n_avec_reentree'])}</div>"
                        f"    <div class='metric-sub warn'>{row['pct_avec_reentree']:.1f}% du total</div>"
                        f"  </div>"
                        f"  <div class='metric-card'>"
                        f"    <div class='metric-label'>Ré-entrées max</div>"
                        f"    <div class='metric-value' style='font-size:1.25rem'>{fmt(row['max_reentrees'])}</div>"
                        f"    <div class='metric-sub'>ménage le + instable</div>"
                        f"  </div>"
                        f"  <div class='metric-card'>"
                        f"    <div class='metric-label'>Moy. ré-entrées</div>"
                        f"    <div class='metric-value' style='font-size:1.25rem'>{f'{avg:.2f}' if pd.notna(avg) else '—'}</div>"
                        f"    <div class='metric-sub'>parmi les churners</div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True)

                with chart_col:
                    fig_re = go.Figure()
                    fig_re.add_trace(go.Bar(
                        x=sub_grouped["label"],
                        y=sub_grouped["n_menages"],
                        marker_color=[_reentry_color(l) for l in sub_grouped["label"]],
                        text=sub_grouped["pct_menages"].apply(lambda x: f"{x:.1f}%"),
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Ménages : %{y:,}<extra></extra>",
                    ))
                    fig_re.update_layout(**PL, height=300,
                        xaxis=dict(
                            title="",
                            type="category",
                            categoryorder="array",
                            categoryarray=sub_grouped["label"].tolist(),
                        ),
                        yaxis_title="Ménages",
                        title=dict(
                            text=f"{PROG_FULL.get(re_prog_sel, re_prog_sel)} — Distribution des ré-entrées",
                            font=dict(size=12, color="#6b7280"), x=0))
                    st.plotly_chart(fig_re, width='stretch')

    # ── All tables at the bottom ──────────────────────────────────────────────
    if "churn_timeline" in agg and not agg["churn_timeline"].empty:
        df_ct2      = agg["churn_timeline"].copy()
        df_ct2["date"] = pd.to_datetime(df_ct2["date"])
        programmes_ct2 = sorted([p for p in df_ct2["programme"].unique()
                                  if p not in ["Non classifié", "AMOA"]])
        prog_filter2 = [p for p in (prog_filter if "prog_filter" in dir() else programmes_ct2)
                        if p in programmes_ct2] or programmes_ct2
        df_ct_filt2 = df_ct2[df_ct2["programme"].isin(prog_filter2)]

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Tableau récapitulatif mensuel", "Monthly summary table")
        tbl_ct = df_ct_filt2[[
            "date", "programme", "pool_start", "pool_end",
            "entries", "exits", "churn_rate", "acquisition_rate", "net_rate",
        ]].copy()
        tbl_ct["date"] = tbl_ct["date"].dt.strftime("%Y-%m")
        tbl_ct["churn_rate"]       = (tbl_ct["churn_rate"] * 100).round(2)
        tbl_ct["acquisition_rate"] = (tbl_ct["acquisition_rate"] * 100).round(2)
        tbl_ct["net_rate"]         = (tbl_ct["net_rate"] * 100).round(2)
        tbl_ct.columns = ["Mois", "Programme", "Pool début", "Pool fin",
                          "Entrées", "Sorties", "Churn (%)", "Acquisition (%)", "Net (%)"]
        st.dataframe(tbl_ct, width="stretch", hide_index=True)

    if "reentry_summary" in agg and not agg["reentry_summary"].empty:
        df_re2 = agg["reentry_summary"].copy()
        programmes_re2 = sorted([p for p in df_re2["programme"].unique()
                                  if p not in ["Non classifié", "AMOA"]])
        st.markdown("<br>", unsafe_allow_html=True)
        sec("Tableau détaillé des ré-entrées", "Re-entry detail table")
        tbl_re = df_re2[df_re2["programme"].isin(programmes_re2)].copy()
        tbl_re.columns = ["Programme", "Catégorie", "Ménages", "% ménages"]
        st.dataframe(tbl_re, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — BENEFICIAIRE FLOWS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Flux Bénéficiaire":
    ptitle("Flux de Bénéficiaires", "Beneficiaire Flows",
           "Ménages Bénéficiaires vs non-Bénéficiaires par programme et par mois")

    st.info(
        "**Actif = 1 (Bénéficiaire)** : ménage Bénéficiaire au programme. "
        "**Actif = 0 (Non-Bénéficiaire)** : ménage non Bénéficiaires. "
        "Compte cumulative : menages qui restent bénéficiaires sont conservés, ceux qui deviennent non bénéficiaires sont retirés.")

    if "monthly_beneficiaire_flows" in agg and not agg["monthly_beneficiaire_flows"].empty:
        df_benef = agg["monthly_beneficiaire_flows"].copy()
        df_benef["date"] = pd.to_datetime(df_benef["date"])
        programmes = sorted(df_benef["programme"].unique())

        # ── Programme selector ────────────────────────────────────────────────
        sel_prog = st.selectbox(
            "Programme", programmes,
            format_func=lambda p: PROG_FULL.get(p, p),
            key="benef_prog_sel")

        df_prog = df_benef[df_benef["programme"] == sel_prog].copy()
        prog_color = PROG_COLORS.get(sel_prog, "#1a56db")

        if not df_prog.empty:
            # ── Compact stat cards (horizontal row) ───────────────────────────
            max_elig     = int(df_prog["unique_menages_eligible"].max())
            max_not_elig = int(df_prog["unique_menages_not_eligible"].max())
            became_elig  = int(df_prog["menages_became_eligible"].sum())
            became_inelig= int(df_prog["menages_became_ineligible"].sum())
            peak_elig_date    = df_prog.loc[df_prog["unique_menages_eligible"].idxmax(), "date"].strftime("%Y-%m") if len(df_prog) > 0 else ""
            peak_noelig_date  = df_prog.loc[df_prog["unique_menages_not_eligible"].idxmax(), "date"].strftime("%Y-%m") if len(df_prog) > 0 else ""

            st.markdown(
                f"<div style='display:flex;gap:0;flex-wrap:wrap;margin:14px 0 18px;"
                f"border-top:2px solid #111827;padding-top:14px'>"
                f"  <div class='metric-card' style='flex:1;min-width:150px'>"
                f"    <div class='metric-label'>Max Bénéficiaires · Peak beneficial</div>"
                f"    <div class='metric-value' style='color:{prog_color}'>{fmt(max_elig)}</div>"
                f"    <div class='metric-sub'>en {peak_elig_date}</div>"
                f"  </div>"
                f"  <div class='metric-card' style='flex:1;min-width:150px'>"
                f"    <div class='metric-label'>Max non-Bénéficiaires · Peak non-beneficial</div>"
                f"    <div class='metric-value'>{fmt(max_not_elig)}</div>"
                f"    <div class='metric-sub'>en {peak_noelig_date}</div>"
                f"  </div>"
                f"  <div class='metric-card' style='flex:1;min-width:150px'>"
                f"    <div class='metric-label'>Devenus Bénéficiaires · Became beneficial</div>"
                f"    <div class='metric-value good'>{fmt(became_elig)}</div>"
                f"    <div class='metric-sub'>Total cumulé</div>"
                f"  </div>"
                f"  <div class='metric-card' style='flex:1;min-width:150px'>"
                f"    <div class='metric-label'>Devenus non-Bénéficiaires · Became non-beneficial</div>"
                f"    <div class='metric-value warn'>{fmt(became_inelig)}</div>"
                f"    <div class='metric-sub'>Total cumulé</div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True)

            # ── Side-by-side: cumul chart + transitions chart ──────────────────
            sec(f"{sel_prog} — Évolution mensuelle", "Monthly flows")
            g1, g2 = st.columns(2)
            with g1:
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=df_prog["date"], y=df_prog["unique_menages_eligible"],
                    name="Bénéficiaires (cumul)", mode="lines",
                    line=dict(color="#0e9f6e", width=2.5), fill="tozeroy",
                    fillcolor="rgba(14,159,110,0.15)"))
                fig_cum.add_trace(go.Scatter(x=df_prog["date"], y=df_prog["unique_menages_not_eligible"],
                    name="Non-Bénéficiaires (cumul)", mode="lines",
                    line=dict(color="#c2410c", width=2.5), fill="tozeroy",
                    fillcolor="rgba(194,65,12,0.10)"))
                fig_cum.update_layout(**PL, height=280,
                    title=dict(text="Comptes cumulatifs mensuels",
                               font=dict(size=12, color="#6b7280"), x=0),
                    xaxis_title="Mois", yaxis_title="Ménages uniques", hovermode="x unified",
                    legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig_cum, width='stretch')

            with g2:
                fig_tr = go.Figure()
                fig_tr.add_trace(go.Bar(x=df_prog["date"], y=df_prog["menages_became_eligible"],
                    name="Devenus Bénéficiaires", marker_color="#0e9f6e"))
                fig_tr.add_trace(go.Bar(x=df_prog["date"], y=df_prog["menages_became_ineligible"],
                    name="Devenus non-Bénéficiaires", marker_color="#c2410c"))
                fig_tr.update_layout(**PL, height=280,
                    title=dict(text="Transitions de statut par mois",
                               font=dict(size=12, color="#6b7280"), x=0),
                    xaxis_title="Mois", yaxis_title="Ménages", barmode="group",
                    hovermode="x", legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig_tr, width='stretch')
    else:
        st.warning("Données de flux bénéficiaire non disponibles. "
                  "Assurez-vous que beneficiaire.csv a été traité.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — HOUSEHOLD EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explorateur de ménage":
    ptitle("Explorateur de ménage", "Household Explorer",
           "Consultez l'historique complet d'un ménage à partir de son identifiant",
           "Look up any household's full scoring history by ID")

    try:
        import duckdb as _duckdb
    except ImportError:
        st.error("duckdb non installé. Exécutez : pip install duckdb")
        st.stop()

    master_path = SNAP_DIR / "master_events.parquet"
    menage_path = SNAP_DIR / "raw_menage.parquet"

    if not master_path.exists():
        st.error(f"Fichier introuvable : {master_path}")
        st.stop()

    st.markdown(
        "<div style='margin-bottom:8px;font-size:0.82rem;color:#6b7280'>"
        "Entrez un identifiant ménage anonymisé (menage_ano) :</div>",
        unsafe_allow_html=True)

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        raw_input = st.text_input("menage_ano", label_visibility="collapsed",
                                  placeholder="ex: 123456789")
    with col_btn:
        st.button("Rechercher", width="stretch")

    if not raw_input.strip():
        st.markdown(
            "<div style='margin-top:60px;text-align:center;color:#9ca3af;"
            "font-size:0.9rem'>Entrez un identifiant pour commencer.</div>",
            unsafe_allow_html=True)
        st.stop()

    try:
        menage_id = int(raw_input.strip())
    except ValueError:
        st.error("L'identifiant doit être un nombre entier.")
        st.stop()

    @st.cache_data(show_spinner="Recherche en cours…", ttl=300)
    def fetch_household(mid: int):
        _con = _duckdb.connect()
        mp = str(master_path)
        events = _con.execute(f"""
            SELECT
                date_calcul,
                score_final                         AS score_final_corrige,
                score_calcule,
                score_corrige,
                was_corrected,
                type_demande                        AS canal,
                programme,
                eligible,
                ROUND(dist_threshold, 6)            AS dist_threshold,
                "near_0.10"                         AS near_10pct,
                region,
                milieu,
                taille_menage
            FROM read_parquet('{mp}')
            WHERE menage_ano = {mid}
            ORDER BY date_calcul ASC
        """).df()
        demo = pd.DataFrame()
        if menage_path.exists():
            rmp = str(menage_path)
            demo = _con.execute(f"""
                SELECT region, province, commune, milieu,
                       genre_cm, type_menage, taille_menage,
                       etat_matrimonial_cm, date_naissance_cm
                FROM read_parquet('{rmp}')
                WHERE menage_ano = {mid}
                LIMIT 1
            """).df()
        return events, demo

    events, demo = fetch_household(menage_id)

    if events.empty:
        st.warning(f"Aucun événement trouvé pour le ménage **{menage_id}**.")
        st.stop()

    events = events.copy()
    events["date_calcul"] = pd.to_datetime(events["date_calcul"])
    events = events.sort_values("date_calcul").reset_index(drop=True)
    events["delta_ise"] = events["score_final_corrige"].diff().round(6)

    latest_by_prog = (
        events.sort_values("date_calcul")
        .groupby("programme").last().reset_index()
        [["programme","score_final_corrige","eligible"]]
    )

    sec("Identité du ménage", "Household identity")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card("Identifiant", "Household ID", str(menage_id)),
                    unsafe_allow_html=True)
    with c2:
        n_ev = len(events)
        n_pr = events["programme"].nunique()
        st.markdown(card("Événements enregistrés", "Recorded events",
            str(n_ev), f"{n_pr} programme(s)"), unsafe_allow_html=True)
    with c3:
        d0 = events["date_calcul"].min().strftime("%d/%m/%Y")
        d1 = events["date_calcul"].max().strftime("%d/%m/%Y")
        st.markdown(card("Premier événement", "First event",
            d0, f"Dernier : {d1}"), unsafe_allow_html=True)
    with c4:
        s0 = events["score_final_corrige"].iloc[0]
        s1 = events["score_final_corrige"].iloc[-1]
        drift = round(s1 - s0, 4)
        cls = "good" if drift < 0 else ("warn" if drift > 0 else "")
        lbl = "Amélioration" if drift < 0 else ("Dégradation" if drift > 0 else "Stable")
        st.markdown(card("Drift total (premier à dernier)", "Total score drift",
            f"{drift:+.4f}", lbl, cls), unsafe_allow_html=True)

    if not demo.empty:
        d = demo.iloc[0]
        parts = []
        if pd.notna(d.get("region")):        parts.append(f"Région : {d['region']}")
        if pd.notna(d.get("province")):      parts.append(f"Province : {d['province']}")
        if pd.notna(d.get("milieu")):        parts.append(f"Milieu : {d['milieu']}")
        if pd.notna(d.get("genre_cm")):      parts.append(f"Chef : {d['genre_cm']}")
        if pd.notna(d.get("type_menage")):   parts.append(f"Type : {d['type_menage']}")
        if pd.notna(d.get("taille_menage")): parts.append(f"Taille : {int(d['taille_menage'])} pers.")
        if parts:
            st.markdown(
                "<div style='background:#f7f8fa;border:1px solid #e0e4ea;"
                "border-radius:6px;padding:10px 16px;margin:10px 0 4px;"
                f"font-size:0.8rem;color:#374151'>{' &nbsp;&nbsp;|&nbsp;&nbsp; '.join(parts)}</div>",
                unsafe_allow_html=True)

    if not latest_by_prog.empty:
        pills = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 8px'>"
        for _, row in latest_by_prog.iterrows():
            prog  = row["programme"]
            score = row["score_final_corrige"]
            elig  = row["eligible"]
            thr   = THRESHOLDS.get(prog)
            pc    = PROG_COLORS.get(prog, "#9ca3af")
            if thr is not None:
                status_txt = "Eligible" if elig else "Non eligible"
                bg = "#f0fdf4" if elig else "#fef2f2"
                bd = "#86efac" if elig else "#fca5a5"
                tc = "#15803d" if elig else "#dc2626"
            else:
                status_txt = "Seuil non défini"
                bg, bd, tc = "#f7f8fa", "#e0e4ea", "#6b7280"
            thr_txt = f" | seuil {thr}" if thr else ""
            pills += (
                f"<div style='background:{bg};border:1px solid {bd};"
                f"border-radius:20px;padding:6px 16px;font-size:0.78rem'>"
                f"<span style='color:{pc};font-weight:600'>{prog}</span>"
                f"&nbsp;&nbsp;<span style='color:{tc};font-weight:500'>{status_txt}</span>"
                f"&nbsp;&nbsp;<span style='color:#9ca3af;font-family:monospace'>"
                f"{score:.4f}{thr_txt}</span></div>"
            )
        pills += "</div>"
        st.markdown(pills, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Historique des scores", "Score history")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        fig = go.Figure()
        for prog in sorted(events["programme"].unique()):
            sub   = events[events["programme"] == prog]
            color = PROG_COLORS.get(prog, "#9ca3af")
            fig.add_trace(go.Scatter(
                x=sub["date_calcul"], y=sub["score_final_corrige"],
                mode="lines+markers", name=PROG_FULL.get(prog, prog),
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color,
                            symbol="circle-open" if not sub["was_corrected"].any()
                                   else "circle"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Score : <b>%{y:.4f}</b><br><extra></extra>",
            ))
        for prog, thr in THRESHOLDS.items():
            color = PROG_COLORS.get(prog, "#9ca3af")
            fig.add_hline(y=thr, line_dash="dot", line_color=color, line_width=1.2,
                annotation_text=f"Seuil {prog}  {thr}",
                annotation_font_color=color, annotation_font_size=10,
                annotation_position="bottom right")
        fig.update_layout(**PL, height=380,
            xaxis_title="Date de calcul", yaxis_title="Score ISE",
            legend=dict(orientation="h", y=1.1), hovermode="x unified",
            title=dict(text="Evolution du score ISE",
                       font=dict(size=11, color="#6b7280"), x=0))
        st.plotly_chart(fig, width="stretch")

    with col_right:
        ddf = events.dropna(subset=["delta_ise"]).copy()
        if not ddf.empty:
            colors_bar = ["#dc2626" if v > 0 else "#15803d" for v in ddf["delta_ise"]]
            fig2 = go.Figure(go.Bar(
                x=ddf["date_calcul"], y=ddf["delta_ise"], marker_color=colors_bar,
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>ΔISE : %{y:+.4f}<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color="#d1d5db", line_width=1)
            PL_tight = {k: v for k, v in PL.items() if k != "margin"}
            fig2.update_layout(**PL_tight, height=380,
                margin=dict(t=40, b=36, l=36, r=24),
                xaxis_title="Date de calcul", yaxis_title="ΔISE",
                title=dict(text="Variation ΔISE entre événements consécutifs  "
                               "(rouge = dégradation, vert = amélioration)",
                           font=dict(size=11, color="#6b7280"), x=0))
            st.plotly_chart(fig2, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Détail des événements", "Event details")

    canal_labels = {
        "mise à jour du dossier": "Mise à jour",
        "Inscription":            "Inscription",
        "radiation":              "Radiation",
        "révision de score":      "Révision",
        "Préinscription":         "Préinscription",
        "réclamation":            "Réclamation",
    }
    tbl = events[[
        "date_calcul", "programme", "canal",
        "score_final_corrige", "score_calcule",
        "delta_ise", "eligible",
        "dist_threshold", "was_corrected",
    ]].copy().sort_values("date_calcul", ascending=False)

    tbl["date_calcul"]   = tbl["date_calcul"].dt.strftime("%d/%m/%Y")
    tbl["canal"]         = tbl["canal"].map(canal_labels).fillna(tbl["canal"])
    tbl["eligible"]      = tbl["eligible"].map({True: "Oui", False: "Non"}).fillna("—")
    tbl["was_corrected"] = tbl["was_corrected"].map({True: "Oui", False: "Non"}).fillna("—")

    tbl.columns = ["Date", "Programme", "Canal", "Score final", "Score calculé",
                   "ΔISE", "Eligible", "Distance seuil", "Correction opérateur"]

    def highlight(row):
        styles = [""] * len(row)
        idx = list(row.index)
        if "Eligible" in idx:
            i = idx.index("Eligible")
            if row["Eligible"] == "Oui":
                styles[i] = "background-color:#f0fdf4;color:#15803d;font-weight:500"
            elif row["Eligible"] == "Non":
                styles[i] = "background-color:#fef2f2;color:#dc2626;font-weight:500"
        if "ΔISE" in idx:
            i = idx.index("ΔISE")
            try:
                v = float(row["ΔISE"])
                styles[i] = ("color:#dc2626;font-weight:500" if v > 0
                             else "color:#15803d;font-weight:500" if v < 0
                             else "")
            except (TypeError, ValueError):
                pass
        return styles

    styled = (
        tbl.style
        .apply(highlight, axis=1)
        .format({
            "Score final":    "{:.4f}",
            "Score calculé":  "{:.4f}",
            "ΔISE":           lambda x: f"{x:+.4f}" if pd.notna(x) else "—",
            "Distance seuil": lambda x: f"{x:+.6f}" if pd.notna(x) else "—",
        }, na_rep="—")
    )
    st.dataframe(styled, width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — DELTA AJUSTÉ
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Delta Ajusté":
    ptitle("Delta Ajusté", "Adjusted Delta",
           "Variation du score ISE ramenée à une journée · ΔISE / jours entre événements"
           )

    st.markdown(
        "<div style='background:#eff6ff;border-left:3px solid #1a56db;"
        "border-radius:0 6px 6px 0;padding:12px 18px;margin-bottom:16px;"
        "font-size:0.84rem;color:#374151;line-height:1.6'>"
        "<strong>Définition :</strong> "
        "<code>delta_ajusté = ΔISE / jours_entre_événements</code> — "
        "il mesure le <em>rythme quotidien</em> de variation du score ISE pour chaque "
        "transition entre deux événements consécutifs d'un même ménage.<br>"
        "<strong>Règle d'exclusion :</strong> les paires d'événements séparées de "
        "<strong>moins de 90 jours</strong> sont exclues  "
        
        "</div>",
        unsafe_allow_html=True)

    # ── Coverage cards — compact horizontal row ───────────────────────────────
    dd_g = agg.get("dd_global", pd.DataFrame())
    if not dd_g.empty:
        r = dd_g.iloc[0]
        n_total   = int(r.get("n_total",       0))
        n_valid   = int(r.get("n_valid",        0))
        n_excl    = int(r.get("n_excluded_90d", 0))
        pct_valid = n_valid / n_total * 100 if n_total else 0
        mean_dd   = float(r.get("mean",   0))
        median_dd = float(r.get("median", 0))
        std_dd    = float(r.get("std",    0))
        p10_dd    = float(r.get("p10",    0))
        p90_dd    = float(r.get("p90",    0))

        st.markdown(
            f"<div style='display:flex;gap:0;flex-wrap:wrap;margin-bottom:18px;"
            f"border-top:2px solid #111827;padding-top:14px'>"
            f"  <div class='metric-card' style='flex:1;min-width:130px'>"
            f"    <div class='metric-label'>Transitions valides · Valid</div>"
            f"    <div class='metric-value'>{fmt(n_valid)}</div>"
            f"    <div class='metric-sub'>{pct_valid:.1f}% · {fmt(n_excl)} exclues</div>"
            f"  </div>"
            f"  <div class='metric-card' style='flex:1;min-width:130px'>"
            f"    <div class='metric-label'>Moyenne Δ ajusté · Mean</div>"
            f"    <div class='metric-value {'good' if mean_dd < 0 else 'warn'}'>{mean_dd:+.6f}</div>"
            f"    <div class='metric-sub'>Négatif = amélioration nette</div>"
            f"  </div>"
            f"  <div class='metric-card' style='flex:1;min-width:130px'>"
            f"    <div class='metric-label'>Médiane · Median</div>"
            f"    <div class='metric-value'>{median_dd:+.6f}</div>"
            f"  </div>"
            f"  <div class='metric-card' style='flex:1;min-width:130px'>"
            f"    <div class='metric-label'>Écart-type · Std dev</div>"
            f"    <div class='metric-value'>{std_dd:.6f}</div>"
            f"    <div class='metric-sub'>Instabilité du rythme</div>"
            f"  </div>"
            f"  <div class='metric-card' style='flex:1;min-width:130px'>"
            f"    <div class='metric-label'>p10 / p90</div>"
            f"    <div class='metric-value'>{p10_dd:+.5f}</div>"
            f"    <div class='metric-sub'>p90: {p90_dd:+.5f}</div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True)

    # ── Interval distribution + scatter side by side ──────────────────────────
    sec("Répartition des intervalles & nuage ΔISE", "Intervals distribution & scatter")
    g1, g2 = st.columns(2)

    dd_days = agg.get("dd_days_dist", pd.DataFrame())
    with g1:
        st.caption("Barres rouges = intervalles > 90j exclus du calcul.")
        if not dd_days.empty:
            dd_days["exclu"] = ~dd_days["bucket"].str.contains("exclu")
            colors = ["#dc2626" if e else "#1a56db" for e in dd_days["exclu"]]
            fig = go.Figure(go.Bar(
                x=dd_days["bucket"], y=dd_days["n"],
                marker_color=colors,
                text=dd_days["n"].apply(fmt), textposition="outside",
            ))
            fig.update_layout(**PL, height=260,
                xaxis_title="Intervalle entre événements",
                yaxis_title="Transitions", showlegend=False)
            st.plotly_chart(fig, width='stretch')

    dd_sc = agg.get("dd_scatter_sample", pd.DataFrame())
    with g2:
        st.caption("Échantillon 5 000 transitions. Zone grisée = exclus (> 90j).")
        if not dd_sc.empty:
            fig = go.Figure()
            for prog in dd_sc["programme"].unique():
                if prog == "Non classifié": continue
                sub   = dd_sc[dd_sc["programme"] == prog]
                color = PROG_COLORS.get(prog, "#9ca3af")
                fig.add_trace(go.Scatter(
                    x=sub["days_between"], y=sub["delta_ISE"],
                    mode="markers", name=PROG_FULL.get(prog, prog),
                    marker=dict(color=color, size=4, opacity=0.45),
                    hovertemplate="Jours : %{x}<br>ΔISE : %{y:.4f}<extra></extra>",
                ))
            fig.add_vrect(x0=90, x1=dd_sc["days_between"].max()+5,
                fillcolor="rgba(156,163,175,0.12)", layer="below", line_width=0,
                annotation_text="Exclu (> 90j)", annotation_position="top left",
                annotation_font=dict(color="#9ca3af", size=10))
            fig.add_vline(x=90, line=dict(color="#9ca3af", width=1.5, dash="dash"))
            fig.add_hline(y=0, line=dict(color="#6b7280", width=1, dash="dot"))
            fig.update_layout(**PL, height=260,
                xaxis_title="Jours entre événements", yaxis_title="ΔISE brut", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width='stretch')

    # ── Monthly trend with programme selector ─────────────────────────────────
    sec("Évolution mensuelle du delta ajusté par programme",
        "Monthly trend of adjusted delta")
    st.caption("Moyenne du delta ajusté de toutes les transitions valides par mois.")

    dd_ts = agg.get("dd_timeseries", pd.DataFrame())
    if not dd_ts.empty:
        dd_ts["month"] = pd.to_datetime(dd_ts["month"])
        avail_progs = [p for p in dd_ts["programme"].unique() if p != "Non classifié"]

        tf1, tf2 = st.columns([1, 4])
        with tf1:
            prog_sel_dd = st.multiselect(
                "Programmes", avail_progs,
                default=avail_progs,
                key="dd_prog_sel",
                format_func=lambda p: PROG_FULL.get(p, p))
            

        with tf2:
            fig_ts = go.Figure()
            for prog in (prog_sel_dd or avail_progs):
                sub   = dd_ts[dd_ts["programme"] == prog].sort_values("month")
                color = PROG_COLORS.get(prog, "#9ca3af")
                fig_ts.add_trace(go.Scatter(
                    x=sub["month"], y=sub["mean_dd"].round(6),
                    mode="lines+markers", name=PROG_FULL.get(prog, prog),
                    line=dict(color=color, width=2), marker=dict(size=5),
                    hovertemplate=f"<b>{prog}</b> %{{x|%b %Y}}<br>Δ ajusté : %{{y:.6f}}<extra></extra>",
                ))
            fig_ts.add_hline(y=0, line=dict(color="#6b7280", width=1.2, dash="dash"),
                annotation_text="Stagnation (Δ=0)",
                annotation_font=dict(color="#6b7280", size=10))
            fig_ts.update_layout(**PL, height=300,
                xaxis_title="Mois", yaxis_title="Δ ajusté moyen (par jour)", hovermode="x unified",
                legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_ts, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — VOLATILITÉ TEMPORELLE  (Temporal Volatility)
elif page == "Volatilité Temporelle":
    ptitle(
        "Volatilité Temporelle",
        "Instabilité hebdomadaire des scores ISE — σ(ΔISE) et choc moyen |ΔISE|",

    )
 
    # ── Explanation banner ────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:#eff6ff;border-left:3px solid #1a56db;"
        "border-radius:0 6px 6px 0;padding:12px 18px;margin-bottom:20px;"
        "font-size:0.84rem;color:#374151;line-height:1.65'>"
        "<strong> σ(ΔISE) par semaine :</strong> "
        "écart-type de toutes les variations de score survenues cette semaine. "
        "Une valeur élevée signifie que les ménages changent de score de façon très hétérogène.<br>"
        "<strong> |ΔISE| moyen par semaine :</strong> "
        "taille moyenne du choc de score, sans tenir compte de la direction. "
        "Distingue aussi la moyenne des améliorations (ΔISE &lt; 0) "
        "et la moyenne des dégradations (ΔISE &gt; 0)."
        "</div>",
        unsafe_allow_html=True,
    )
 
    # ── Load pre-baked data ───────────────────────────────────────────────────
    df_vol = agg.get("delta_vol_weekly", pd.DataFrame())
    df_abs = agg.get("delta_abs_weekly", pd.DataFrame())
 
    if df_vol.empty or df_abs.empty:
        st.warning(
            "Données de volatilité hebdomadaire non disponibles. "
            "Assurez-vous d'avoir exécuté `prebake_dashboard.py` après avoir "
            "ajouté les requêtes `delta_vol_weekly` et `delta_abs_weekly`."
        )
        st.stop()
 
    # ── Date parsing ──────────────────────────────────────────────────────────
    df_vol = df_vol.copy()
    df_abs = df_abs.copy()
    df_vol["week"] = pd.to_datetime(df_vol["week"])
    df_abs["week"] = pd.to_datetime(df_abs["week"])
 
    # ── Optional date range filter ────────────────────────────────────────────
    min_date = df_vol["week"].min().date()
    max_date = df_vol["week"].max().date()
 
    fc1, fc2, _ = st.columns([1, 1, 2])
    with fc1:
        date_from = st.date_input("Début · From", value=min_date,
                                  min_value=min_date, max_value=max_date,
                                  key="vt_from")
    with fc2:
        date_to = st.date_input("Fin · To", value=max_date,
                                min_value=min_date, max_value=max_date,
                                key="vt_to")
 
    mask_vol = (df_vol["week"].dt.date >= date_from) & (df_vol["week"].dt.date <= date_to)
    mask_abs = (df_abs["week"].dt.date >= date_from) & (df_abs["week"].dt.date <= date_to)
    df_vol_f = df_vol[mask_vol].reset_index(drop=True)
    df_abs_f = df_abs[mask_abs].reset_index(drop=True)
 
    if df_vol_f.empty:
        st.warning("Aucune donnée dans la période sélectionnée.")
        st.stop()
 
    # ── KPI cards ─────────────────────────────────────────────────────────────
    overall_sigma     = round(float(df_vol_f["sigma_delta"].mean()), 4)
    peak_sigma_week   = df_vol_f.loc[df_vol_f["sigma_delta"].idxmax(), "week"].strftime("%d/%m/%Y")
    peak_sigma_val    = round(float(df_vol_f["sigma_delta"].max()), 4)
    overall_abs       = round(float(df_abs_f["mean_abs_delta"].mean()), 4)
    total_menages = int(df_vol_f["n_menages"].sum())
    n_weeks           = len(df_vol_f)
 
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>σ moyen (période) · Avg σ</div>"
            f"<div class='metric-value'>{overall_sigma}</div>"
            f"<div class='metric-sub'>Instabilité moyenne hebdomadaire</div>"
            f"</div>",
            unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Semaine la + instable · Peak σ week</div>"
            f"<div class='metric-value warn'>{peak_sigma_val}</div>"
            f"<div class='metric-sub'>Semaine du {peak_sigma_week}</div>"
            f"</div>",
            unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>|ΔISE| moyen · Avg shock</div>"
            f"<div class='metric-value'>{overall_abs}</div>"
            f"<div class='metric-sub'>Amplitude moyenne des chocs</div>"
            f"</div>",
            unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Ménages analysés · Total</div>"
            f"<div class='metric-value'>{fmt(total_menages)}</div>"
            f"<div class='metric-sub'>{n_weeks} semaines dans la période</div>"
            f"</div>",
            unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    from plotly.subplots import make_subplots as _msp
 
    # ─────────────────────────────────────────────────────────────────────────
    # CHART 1 — Option 2 : σ(ΔISE) per week
    # ─────────────────────────────────────────────────────────────────────────
    sec(" Instabilité hebdomadaire σ(ΔISE)")
    st.caption(
        "Chaque point = écart-type de toutes les variations de score survenues cette semaine. "
        "Un pic indique une semaine où les scores ont beaucoup bougé dans des directions très différentes."
    )
 
    # Percentile catalogue for chart 1 (raw ΔISE)
    # format: label → (col_name, line_color, dash_style)
    C1_PERCENTILES = {
        "p10 ΔISE":       ("p10_delta",    "#818cf8", "dot"),
        "p25 ΔISE":       ("p25_delta",    "#6366f1", "dashdot"),
        "médiane ΔISE":   ("median_delta", "#f59e0b", "dot"),
        "p75 ΔISE":       ("p75_delta",    "#f97316", "dashdot"),
        "p90 ΔISE":       ("p90_delta",    "#ef4444", "dot"),
    }
    # Band catalogue: label → (lower_col, upper_col, fill_rgba)
    C1_BANDS = {
        "Bande p10–p90": ("p10_delta", "p90_delta", "rgba(99,102,241,0.08)"),
        "Bande p25–p75": ("p25_delta", "p75_delta", "rgba(99,102,241,0.18)"),
    }
 
    d1c1, d1c2 = st.columns([1, 4])
    with d1c1:
        show_sigma   = st.checkbox("σ(ΔISE)",                value=True,  key="vt_c1_sigma")
        show_mean_c1 = st.checkbox("mean ΔISE",              value=True,  key="vt_c1_mean")
        show_n_c1    = st.checkbox("N ménages (axe 2)",  value=True,  key="vt_c1_n")
 
        st.markdown(
            "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;"
            "color:#9ca3af;margin:10px 0 3px'>Déciles · Percentiles</div>",
            unsafe_allow_html=True)
        c1_pct_sel = st.multiselect(
            "c1_pct", label_visibility="collapsed",
            options=[k for k, (col, *_) in C1_PERCENTILES.items() if col in df_vol_f.columns],
            default=[],
            key="vt_c1_pct",
        )
 
        st.markdown(
            "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;"
            "color:#9ca3af;margin:10px 0 3px'>Bandes · Bands</div>",
            unsafe_allow_html=True)
        available_c1_bands = [
            b for b, (lo, hi, _) in C1_BANDS.items()
            if lo in df_vol_f.columns and hi in df_vol_f.columns
        ]
        c1_band_sel = st.multiselect(
            "c1_bands", label_visibility="collapsed",
            options=available_c1_bands,
            default=["Bande p10–p90"] if "Bande p10–p90" in available_c1_bands else [],
            key="vt_c1_bands",
        )
 
    with d1c2:
        fig1 = _msp(specs=[[{"secondary_y": True}]])
 
        # Bands first (behind all lines)
        for band_label in c1_band_sel:
            lo_col, hi_col, fill_color = C1_BANDS[band_label]
            fig1.add_trace(go.Scatter(
                x=df_vol_f["week"], y=df_vol_f[lo_col],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False,
            ), secondary_y=False)
            fig1.add_trace(go.Scatter(
                x=df_vol_f["week"], y=df_vol_f[hi_col],
                mode="lines", name=band_label,
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor=fill_color,
                hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>{band_label} haut : %{{y:.6f}}<extra></extra>",
            ), secondary_y=False)
 
        # σ(ΔISE) primary line
        if show_sigma:
            fig1.add_trace(go.Scatter(
                x=df_vol_f["week"], y=df_vol_f["sigma_delta"],
                mode="lines+markers", name="σ(ΔISE)",
                line=dict(color="#1a56db", width=2.5), marker=dict(size=4),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>σ(ΔISE) : <b>%{y:.6f}</b><extra></extra>",
            ), secondary_y=False)
 
        # Selected percentile lines
        for pct_label in c1_pct_sel:
            col, color, dash = C1_PERCENTILES[pct_label]
            if col not in df_vol_f.columns:
                continue
            fig1.add_trace(go.Scatter(
                x=df_vol_f["week"], y=df_vol_f[col],
                mode="lines", name=pct_label,
                line=dict(color=color, width=1.6, dash=dash),
                hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>{pct_label} : %{{y:.6f}}<extra></extra>",
            ), secondary_y=False)
 
        # mean ΔISE + zero line
        if show_mean_c1:
            fig1.add_trace(go.Scatter(
                x=df_vol_f["week"], y=df_vol_f["mean_delta"],
                mode="lines", name="mean ΔISE",
                line=dict(color="#f59e0b", width=1.8, dash="dot"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>mean ΔISE : %{y:.6f}<extra></extra>",
            ), secondary_y=False)
            fig1.add_hline(y=0, line=dict(color="#d1d5db", width=1, dash="dash"))
 
        # N ménages on secondary y
        if show_n_c1:
            fig1.add_trace(go.Scatter(
                x=df_vol_f["week"], y=df_vol_f["n_menages"],
                mode="lines", name="Ménages",
                line=dict(color="rgba(107,114,128,0.40)", width=1.2, dash="dot"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Ménages : %{y:,}<extra></extra>",
            ), secondary_y=True)
 
        fig1.update_layout(
            **PL, height=340, hovermode="x unified",
            xaxis_title="Semaine",
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            title=dict(
                text="σ(ΔISE) par semaine — instabilité de la distribution des variations",
                font=dict(size=12, color="#6b7280"), x=0,
            ),
        )
        fig1.update_yaxes(title_text="σ(ΔISE)  /  ΔISE", secondary_y=False)
        if show_n_c1:
            fig1.update_yaxes(
                title_text="Ménages", secondary_y=True, showgrid=False,
                tickfont=dict(color="rgba(107,114,128,0.6)"),
                title_font=dict(color="rgba(107,114,128,0.6)"),
            )
        st.plotly_chart(fig1, width="stretch")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ─────────────────────────────────────────────────────────────────────────
    # CHART 2 — Option 4 : mean(|ΔISE|) per week
    # ─────────────────────────────────────────────────────────────────────────
    sec("Amplitude hebdomadaire |ΔISE|", "Weekly shock magnitude |ΔISE|")
    st.caption(
        "mean(|ΔISE|) = taille moyenne du choc de score indépendamment de la direction. "
    )
 
    # Percentile catalogue for chart 2 (absolute ΔISE)
    C2_PERCENTILES = {
        "p10 |ΔISE|":     ("p10_abs_delta",    "#818cf8", "dot"),
        "p25 |ΔISE|":     ("p25_abs_delta",    "#6366f1", "dashdot"),
        "médiane |ΔISE|": ("median_abs_delta", "#c084fc", "dot"),
        "p75 |ΔISE|":     ("p75_abs_delta",    "#f97316", "dashdot"),
        "p90 |ΔISE|":     ("p90_abs_delta",    "#ef4444", "dot"),
        "p99 |ΔISE|":     ("p99_abs_delta",    "#9f1239", "longdash"),
    }
    C2_BANDS = {
        "Bande p10–p90 |ΔISE|": ("p10_abs_delta", "p90_abs_delta", "rgba(124,58,237,0.08)"),
        "Bande p25–p75 |ΔISE|": ("p25_abs_delta", "p75_abs_delta", "rgba(124,58,237,0.18)"),
    }
 
    d2c1, d2c2 = st.columns([1, 4])
    with d2c1:
        show_mean_abs = st.checkbox("mean |ΔISE|",             value=True,  key="vt_c2_mean")
        show_split    = st.checkbox("Amélio. / dégradation",   value=True,  key="vt_split")
        show_n_c2     = st.checkbox("N ménages (axe 2)",   value=False, key="vt_c2_n")
 
        st.markdown(
            "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;"
            "color:#9ca3af;margin:10px 0 3px'>Déciles · Percentiles</div>",
            unsafe_allow_html=True)
        c2_pct_sel = st.multiselect(
            "c2_pct", label_visibility="collapsed",
            options=[k for k, (col, *_) in C2_PERCENTILES.items() if col in df_abs_f.columns],
            default=["médiane |ΔISE|"],
            key="vt_c2_pct",
        )
 
        st.markdown(
            "<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;"
            "color:#9ca3af;margin:10px 0 3px'>Bandes · Bands</div>",
            unsafe_allow_html=True)
        available_c2_bands = [
            b for b, (lo, hi, _) in C2_BANDS.items()
            if lo in df_abs_f.columns and hi in df_abs_f.columns
        ]
        c2_band_sel = st.multiselect(
            "c2_bands", label_visibility="collapsed",
            options=available_c2_bands,
            default=["Bande p25–p75 |ΔISE|"] if "Bande p25–p75 |ΔISE|" in available_c2_bands else [],
            key="vt_c2_bands",
        )
 
    with d2c2:
        fig2 = _msp(specs=[[{"secondary_y": True}]])
 
        # Bands first
        for band_label in c2_band_sel:
            lo_col, hi_col, fill_color = C2_BANDS[band_label]
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f[lo_col],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False,
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f[hi_col],
                mode="lines", name=band_label,
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor=fill_color,
                hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>{band_label} haut : %{{y:.6f}}<extra></extra>",
            ), secondary_y=False)
 
        # mean |ΔISE| primary line
        if show_mean_abs:
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f["mean_abs_delta"],
                mode="lines+markers", name="mean |ΔISE|",
                line=dict(color="#7c3aed", width=2.5), marker=dict(size=4),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>mean |ΔISE| : <b>%{y:.6f}</b><extra></extra>",
            ), secondary_y=False)
 
        # Selected percentile lines
        for pct_label in c2_pct_sel:
            col, color, dash = C2_PERCENTILES[pct_label]
            if col not in df_abs_f.columns:
                continue
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f[col],
                mode="lines", name=pct_label,
                line=dict(color=color, width=1.6, dash=dash),
                hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>{pct_label} : %{{y:.6f}}<extra></extra>",
            ), secondary_y=False)
 
        # Directional split
        if show_split:
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f["mean_improvement"],
                mode="lines", name="Amélioration moy. (ΔISE<0)",
                line=dict(color="#15803d", width=1.5, dash="dashdot"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Amélioration moy. : %{y:.6f}<extra></extra>",
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f["mean_degradation"],
                mode="lines", name="Dégradation moy. (ΔISE>0)",
                line=dict(color="#dc2626", width=1.5, dash="dashdot"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Dégradation moy. : %{y:.6f}<extra></extra>",
            ), secondary_y=False)
 
        # N ménages on secondary y
        if show_n_c2:
            fig2.add_trace(go.Scatter(
                x=df_abs_f["week"], y=df_abs_f["n_menages"],
                mode="lines", name="Ménages",
                line=dict(color="rgba(107,114,128,0.40)", width=1.2, dash="dot"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Ménages : %{y:,}<extra></extra>",
            ), secondary_y=True)
 
        fig2.update_layout(
            **PL, height=340, hovermode="x unified",
            xaxis_title="Semaine",
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            title=dict(
                text="Amplitude moyenne des chocs hebdomadaires — mean |ΔISE| et décomposition",
                font=dict(size=12, color="#6b7280"), x=0,
            ),
        )
        fig2.update_yaxes(title_text="|ΔISE|", secondary_y=False)
        if show_n_c2:
            fig2.update_yaxes(
                title_text="Ménages", secondary_y=True, showgrid=False,
                tickfont=dict(color="rgba(107,114,128,0.6)"),
                title_font=dict(color="rgba(107,114,128,0.6)"),
            )
        st.plotly_chart(fig2, width="stretch")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
 
    # ─────────────────────────────────────────────────────────────────────────
    # RAW DATA TABLE (collapsible)
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("Données brutes hebdomadaires · Raw weekly data"):
        # Merge vol + abs frames on week
        abs_cols = ["week", "mean_abs_delta", "median_abs_delta",
                    "p10_abs_delta", "p25_abs_delta", "p75_abs_delta",
                    "p90_abs_delta", "p99_abs_delta",
                    "mean_improvement", "mean_degradation",
                    "n_improving", "n_degrading"]
        tbl_merge = df_vol_f.merge(
            df_abs_f[[c for c in abs_cols if c in df_abs_f.columns]],
            on="week", how="left",
        )
        tbl_merge["week"] = tbl_merge["week"].dt.strftime("%Y-%m-%d")
        tbl_merge = tbl_merge.rename(columns={
            "week":             "Semaine",
            "n_menages":        "Ménages",
            "mean_delta":       "mean ΔISE",
            "median_delta":     "médiane ΔISE",
            "sigma_delta":      "σ(ΔISE)",
            "p10_delta":        "p10 ΔISE",
            "p25_delta":        "p25 ΔISE",
            "p75_delta":        "p75 ΔISE",
            "p90_delta":        "p90 ΔISE",
            "mean_abs_delta":   "mean |ΔISE|",
            "median_abs_delta": "médiane |ΔISE|",
            "p10_abs_delta":    "p10 |ΔISE|",
            "p25_abs_delta":    "p25 |ΔISE|",
            "p75_abs_delta":    "p75 |ΔISE|",
            "p90_abs_delta":    "p90 |ΔISE|",
            "p99_abs_delta":    "p99 |ΔISE|",
            "mean_improvement": "Amélio. moy.",
            "mean_degradation": "Dégrada. moy.",
            "n_improving":      "N amélio.",
            "n_degrading":      "N dégrada.",
        })
        st.dataframe(tbl_merge, width="stretch", hide_index=True)