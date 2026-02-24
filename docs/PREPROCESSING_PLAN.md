# Preprocessing improvement plan (labmae_meds–inspired)

Plan to add admission-aware processing, optional vocab labels, and better UX (progress, config). Your data: `labevents` (Parquet) has `subject_id`, `hadm_id`, `specimen_id`, `charttime`, `itemid`, `valuenum`, ref ranges; `admissions.csv` has `subject_id`, `hadm_id`, `admittime`, `edregtime`, etc.

---

## Phase 1: Foundation (paths + progress)

**Goal:** Single place for paths, and progress bars for long runs.

| Task | Details |
|------|--------|
| **1.1 Paths / config** | Add a small `scripts/config.py` (or a section in `run_pipeline.py`) with `ROOT`, `DEFAULT_MIMIC4`, `DEFAULT_PARQUET`, `DEFAULT_ADMISSIONS_CSV`, `DEFAULT_LAB_IMAGES`, so all scripts and the README use the same defaults. |
| **1.2 Progress in specimen loop** | In `specimen_to_image.py`, wrap the specimen iteration in `tqdm` (e.g. `for spec_id, events in tqdm(iter_specimens_..., desc="Specimens")`) and add `tqdm` to `pyproject.toml` / requirements if missing. |

**Deliverables:** `scripts/config.py` (or consolidated defaults), `tqdm` in specimen loop, README updated to reference same paths.

---

## Phase 2: Admission-aware lab events (optional stage)

**Goal:** Optionally produce a “lab events with admission context” dataset: join admissions, sort by subject + charttime, backfill `hadm_id` for labs in the 0–24h before admission (like labmae_meds), then optionally filter to rows with valid `hadm_id`.

| Task | Details |
|------|--------|
| **2.1 Admissions in a good format** | Convert `admissions.csv` → `admissions.parquet` (DuckDB or pandas), or add a small script `scripts/csv_to_parquet.py`-style for admissions only. Keep columns: `subject_id`, `hadm_id`, `admittime`, (optional: `edregtime`, `race`). |
| **2.2 New script: `scripts/lab_with_admissions.py`** | Inputs: labevents Parquet, admissions Parquet. Steps: (1) Join lab events with admissions on `hadm_id` (left join), keep admission cols (e.g. `admittime`). (2) Sort by `subject_id`, `charttime`. (3) Backfill `admittime` per subject (backward fill). (4) Mark rows in 0–24h before admission; backfill `hadm_id` for those when `hadm_id` is null; then drop the helper column. (5) Optionally filter to `hadm_id` not null. (6) Write `lab_events_with_adm.parquet`. Use Polars for the heavy sort/backfill (or pandas if you prefer; Polars is faster). |
| **2.3 Pipeline integration** | Add a flag to `run_pipeline.py`, e.g. `--with-admissions`, that (1) requires `admissions.parquet` (or builds it from CSV once), (2) runs `lab_with_admissions.py` to produce `lab_events_with_adm.parquet`, (3) uses that Parquet (instead of raw labevents) for the specimen→image step. Document in README. |

**Deliverables:** `admissions.parquet` (or script to create it), `scripts/lab_with_admissions.py`, `run_pipeline.py --with-admissions`, README update.

---

## Phase 3: Optional vocab / labels

**Goal:** Optionally attach human-readable lab labels (e.g. from `d_labitems`) so exports and logs can show lab names.

| Task | Details |
|------|--------|
| **3.1 Vocab join in one place** | In `lab_with_admissions.py` or a small `scripts/add_lab_labels.py`: read `d_labitems.csv`, select `itemid` and label column (e.g. `label` or `fluid`/`category` as needed). Join to the lab table on `itemid` → add a `lab_label` column. Make this optional (e.g. `--vocab-csv` or `--skip-vocab`). |
| **3.2 Use where it helps** | Use labels in (a) any new export (e.g. a sample CSV or Parquet with labels), (b) logging (e.g. top-K itemids printed by name in `lab_grid.py` or in inspect script). No need to change `.npy` format. |

**Deliverables:** Optional `--vocab-csv` / `--skip-vocab` in the script that does the join; optional `lab_label` in exported Parquet; optional label output in inspect/top-K logging.

---

## Phase 4: Optional extras (when needed)

| Task | Details |
|------|--------|
| **4.1 SQL files for complex queries** | If you add more DuckDB logic (e.g. top-K, filters), put raw SQL in `scripts/sql/*.sql` and load with `Path(...).read_text()` so queries are easier to tune and reuse. |
| **4.2 BigQuery path** | If you ever use MIMIC-IV on BigQuery: a script that runs SQL files and streams results to Parquet (like labmae_meds `bigquery.py`) can live under `scripts/` or `scripts/bigquery/` and stay separate from the local CSV/Parquet pipeline. |

---

## Dependency order

```
Phase 1 (paths + tqdm)     →  no dependencies
Phase 2 (admissions)       →  Phase 1 (use shared paths)
Phase 3 (vocab)           →  Phase 1; can build on Phase 2 or on raw lab Parquet
Phase 4                   →  anytime
```

---

## Suggested implementation order

1. **Phase 1.1 + 1.2** (config + tqdm) — quick wins, no new data.
2. **Phase 2.1** (admissions → Parquet) — small script or one-off conversion.
3. **Phase 2.2** (lab_with_admissions.py) — core logic; test on a small Parquet slice first.
4. **Phase 2.3** (--with-admissions in run_pipeline) — wire up and document.
5. **Phase 3** (vocab join + labels in export/logs) — optional and incremental.

---

## File layout after plan

```
Latte/
  MIMIC4/
    labevents.parquet
    admissions.csv
    admissions.parquet   # Phase 2.1
    d_labitems.csv
  scripts/
    config.py            # Phase 1.1 (optional)
    run_pipeline.py      # Phase 1.1, 2.3
    csv_to_parquet.py
    lab_with_admissions.py  # Phase 2.2
    lab_grid.py
    specimen_to_image.py   # Phase 1.2
    inspect_lab_images.py
    sql/                  # Phase 4.1 (optional)
      top_itemids.sql
  docs/
    PREPROCESSING_PLAN.md # this file
```

You can implement Phase 1 first, then Phase 2 when you need admission-level analysis or backfilled `hadm_id`.
