# MIMIC-IV labevents → grid images

Pipeline to turn `labevents.csv` into (lab_type, lab_value) grid images per specimen. Path defaults live in `scripts/config.py` (e.g. `MIMIC4/`, `lab_images/`).

## Setup (uv)

```bash
cd /path/to/Latte
uv sync
```

This creates a `.venv` and installs dependencies from `pyproject.toml` (and locks them in `uv.lock`). Then run scripts with:

```bash
uv run python scripts/run_pipeline.py --convert --skip-images
```

Or activate the venv and use `python` directly:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python scripts/run_pipeline.py --limit-specimens 100
```

**Without uv:** `pip install -r requirements.txt` still works (dependencies are listed in `pyproject.toml` and mirrored in `requirements.txt`).

## 1. Convert CSV to Parquet (one-time, streaming)

Reduces size and enables fast SQL over lab events without loading the full CSV:

```bash
python scripts/run_pipeline.py --convert --skip-images
```

Or only the conversion step:

```bash
# Lab events (default)
python scripts/csv_to_parquet.py
python scripts/csv_to_parquet.py lab --csv MIMIC4/labevents.csv --out MIMIC4/labevents.parquet

# Admissions (for --with-admissions)
python scripts/csv_to_parquet.py admissions
```

Uses DuckDB streaming (M4-style); set `M4_DUCKDB_MEM=4GB` and `M4_DUCKDB_THREADS=2` if needed.

## 2. Generate grid images per specimen

Each specimen becomes one `(H×W)` float array (`.npy`): cell = normalized lab value, 0 = missing.

```bash
# Test run: 100 specimens, 16×16 grid
python scripts/run_pipeline.py --limit-specimens 100

# Full run (after Parquet exists)
python scripts/run_pipeline.py
```

Output: `lab_images/specimen_<id>.npy` (default `--images-dir lab_images`). Progress is shown with a tqdm bar.

Options:

- `--grid H W` — grid shape (default 16 16)
- `--limit-specimens N` — process only N specimens
- `--no-top-k` — use first H×W itemids from `d_labitems` instead of top-K by frequency
- `--parquet PATH` / `--images-dir PATH` / `--mimic4-dir PATH` — paths
- `--with-admissions` — use admission-aware labs (join admissions, backfill hadm_id 0–24h before admit), then build images from `lab_events_with_adm.parquet`
- `--limit-lab-rows N` — with `--with-admissions`, process only subjects from the first N lab rows (good for fast testing). Without this, the full lab table is processed in **subject_id batches** so each batch fits in memory (default 1000 subjects/batch, ~2GB RAM). Tune with `--batch-size` or env `DUCKDB_MEMORY_LIMIT=1GB DUCKDB_THREADS=1`.
- `--force-admissions` — re-run the admission join even if `lab_events_with_adm.parquet` already exists and is up to date

### Admission-aware labs (optional)

To attach admission context and backfill `hadm_id` for labs in the 0–24h before admission:

```bash
# One-time: convert admissions.csv to Parquet (if not already)
python scripts/csv_to_parquet.py admissions

# Run pipeline with admission-aware lab table
python scripts/run_pipeline.py --with-admissions --limit-specimens 100

# Fast test (only load/sort 200k lab rows; then build 10 images)
python scripts/run_pipeline.py --with-admissions --limit-lab-rows 200000 --limit-specimens 10
```

This produces `MIMIC4/lab_events_with_adm.parquet` and uses it for the specimen→image step. You can also run the step alone:

```bash
python scripts/lab_with_admissions.py --lab-parquet MIMIC4/labevents.parquet --adm-parquet MIMIC4/admissions.parquet -o MIMIC4/lab_events_with_adm.parquet
```

Options: `--skip-vocab` to skip joining lab labels from `d_labitems`; `--filter-null-hadm` to drop rows with null `hadm_id` after backfill.

## 3. Run convert + images in one go

```bash
python scripts/run_pipeline.py --convert --limit-specimens 50
```

## Module overview

| File | Role |
|------|------|
| `config.py` | Shared paths (MIMIC4, Parquet files, lab_images) |
| `csv_to_parquet.py` | Stream `labevents.csv` or `admissions.csv` → Parquet via DuckDB |
| `lab_with_admissions.py` | Join labs + admissions, sort, backfill hadm_id 0–24h before admit; optional vocab join |
| `lab_grid.py` | Load `d_labitems`, build itemid → (row,col) index; optional top-K by count from Parquet |
| `specimen_to_image.py` | Query Parquet by specimen, fill grid, normalize with ref range, save `.npy` (with tqdm) |
| `run_pipeline.py` | CLI: `--convert`, `--skip-images`, `--with-admissions`, `--limit-specimens`, etc. |
| `pipeline_stats.py` | Read-only stats for each step (lab_parquet, admissions_parquet, lab_with_adm, lab_images); optional JSON output |
| `top_admissions.py` | List top admissions by lab event count (whole stay or `--first-24h`); uses lab_events_with_adm.parquet |

Grid semantics: each cell is one lab type (from a fixed set of itemids). **Value = raw lab value** (valuenum); missing = nan. **Layout:** by default the grid is ordered by **d_labitems category** (then itemid) so that related tests (e.g. Blood Gas, Chemistry) sit in contiguous regions; use `--no-sort-by-category` to use frequency or itemid order instead.

**Which 256 labs?** The grid has H×W cells (default 16×16 = 256). We **select** the top 256 **itemids by event count** from the lab Parquet (most common labs in your data). We **order** them by **d_labitems.category** then **itemid** so the same category appears in contiguous (row, col) blocks. So every specimen image uses the same fixed map: (row, col) → one lab type.

**Verify the layout:** Run `python scripts/show_grid_layout.py` to print the full (row, col) → category, itemid, label map and confirm category grouping. Use `--no-sort-by-category` or `--no-top-k` to match pipeline options.

## Inspect each pipeline step (statistics)

Use `pipeline_stats.py` to print summary statistics for any step **without re-running** the pipeline. Useful to validate outputs and compare across runs.

```bash
# All steps (default paths: MIMIC4/*.parquet, lab_images/)
python scripts/pipeline_stats.py

# One step
python scripts/pipeline_stats.py --step lab_parquet
python scripts/pipeline_stats.py --step admissions_parquet
python scripts/pipeline_stats.py --step lab_with_adm
python scripts/pipeline_stats.py --step lab_images

# Override path for a step
python scripts/pipeline_stats.py --step lab_with_adm --path MIMIC4/lab_events_with_adm.parquet

# Write stats to JSON (e.g. for tracking over time)
python scripts/pipeline_stats.py --output stats.json
python scripts/pipeline_stats.py --step lab_parquet --output lab_stats.json --brief
```

**What each step reports:**

| Step | Typical stats |
|------|----------------|
| `lab_parquet` | rows, columns, null counts (subject_id, hadm_id, specimen_id, itemid, charttime, valuenum), file size |
| `admissions_parquet` | rows, unique subject_id/hadm_id, null counts, file size |
| `lab_with_adm` | rows, unique subject_id/hadm_id/specimen_id, hadm_id null %, admittime null %, file size |
| `lab_images` | count of specimen_*.npy, shapes, whether all same shape |

Processing scripts also print a short **end-of-run** summary: `csv_to_parquet` prints row count and file size after writing; `lab_with_admissions` prints rows, unique hadm_id, and hadm_id null %.

## Inspect generated images

```bash
# List .npy files and print shape/stats for each
uv run python scripts/inspect_lab_images.py

# One specimen: stats + optional heatmap
uv run python scripts/inspect_lab_images.py --specimen 32
uv run python scripts/inspect_lab_images.py --specimen 32 --plot   # requires: uv add matplotlib
```

**Find admissions with the most lab events** (whole stay or first 24h):

```bash
# Top admissions by lab event count for the entire stay (no 24h limit)
python scripts/top_admissions.py --top 20

# Same but restricted to first 24 hours from admission
python scripts/top_admissions.py --top 20 --first-24h

# Custom Parquet path
python scripts/top_admissions.py --parquet MIMIC4/lab_events_with_adm.parquet --top 50
```

Output columns: `hadm_id`, `n_events`, `n_labs` (distinct itemids), `n_specimens`. Use these `hadm_id` values when building admission-level (T×H×W) videos.

**Find specimens with many lab results** (then build + inspect):

```bash
# List specimen IDs with the most lab events (from Parquet)
python scripts/inspect_specimen.py --find-rich 10

# Build .npy for one or more of those IDs (no need to run the full pipeline)
python scripts/build_specimen_npy.py --specimen 83925500
python scripts/build_specimen_npy.py --specimens 83925500 41650253 45668018

# Only list specimens that already have a .npy in lab_images/
python scripts/inspect_specimen.py --find-rich 10 --only-with-npy
```

**Show grid layout** (which 256 labs, in what order; verify category grouping):

```bash
python scripts/show_grid_layout.py
python scripts/show_grid_layout.py --compact
python scripts/show_grid_layout.py --no-sort-by-category
```

**Reconstruct a specimen .npy** (which lab is at each grid cell, and optional raw Parquet events):

```bash
# Map (row,col) → itemid, lab name, value (non-missing cells only)
python scripts/inspect_specimen.py --specimen 367826

# Also show raw events from Parquet (subject_id, charttime, itemid, valuenum, ref range)
python scripts/inspect_specimen.py --specimen 367826 --raw

# If you used a specific parquet or --no-top-k
python scripts/inspect_specimen.py --specimen 367826 --parquet MIMIC4/lab_events_with_adm.parquet --raw
python scripts/inspect_specimen.py --specimen 367826 --no-top-k --all-cells
```
