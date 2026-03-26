# QFRsuDashboard

Standalone RSU dashboard app that depends on `qfpytoolbox` as a package.

## Prerequisites

- Python 3.10+
- `qfpytoolbox` available as installed package

Recommended setup from this project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e ../QFPyToolbox
```

Alternative (when toolbox is published):

```bash
python -m pip install qfpytoolbox
```

## Input data (supported formats)

Place raw files under:

`data/raw/`

Supported formats: `.csv`, `.arrow`, `.xlsx`

Expected logical sources:

- `menage` (e.g. `menage.csv`)
- `scores` (either `scores.csv` or `score.csv`) **required**
- `programmes` (either consolidated `programmes.csv` with `menage_ano, programme`, or split files `amot.csv`, `asd.csv`, `amoa.csv`)
- `beneficiaire` (optional)

## Run end-to-end

From this project root:

```bash
# 1) Build parquet snapshots from raw inputs
python pipeline.py --input-dir data/raw --output-dir snapshots/csv

# 2) Pre-bake cache for fast dashboard startup
python prebake_dashboard.py

# 3) Start Streamlit dashboard
streamlit run dashboard.py
```

## Architecture

- App scripts orchestrate pipeline + prebake + UI.
- Business compute is called from `qfpytoolbox` (package dependency), especially via:
  - `qfpytoolbox.rsu`
  - `qfpytoolbox.rsu_builder`
  - `qfpytoolbox.rsu_loaders`
