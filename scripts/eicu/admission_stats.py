"""
Distribution analysis of the eICU lab_events_with_adm dataset.

Mirrors `scripts/admission_stats.py`, but adapts to the eICU extract:

  - `patientunitstayid` is the admission/stay identifier (`hadm_id` equivalent)
  - `lab_code` is the lab identifier (`itemid` equivalent)
  - `labresult` is the numeric value (`valuenum` equivalent)
  - there is usually no `specimen_id`, so collections are grouped by
    `labresultoffset` (minutes from admission) within each stay

Usage:
    uv run python scripts/eicu/admission_stats.py
    uv run python scripts/eicu/admission_stats.py --parquet eICU/lab_events_with_adm.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_LAB_WITH_ADM_PARQUET
from schema_utils import _col_expr, _specimen_expr, resolve_schema_columns


def _pct(arr: list[float], ps: list[int] = [5, 25, 50, 75, 95]) -> dict[str, float]:
    """Compute percentiles from a sorted list."""
    import math

    n = len(arr)
    if n == 0:
        return {f"p{p}": 0 for p in ps}
    result = {}
    for p in ps:
        k = (p / 100) * (n - 1)
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        if lo == hi:
            result[f"p{p}"] = arr[lo]
        else:
            result[f"p{p}"] = arr[lo] + (k - lo) * (arr[hi] - arr[lo])
    return result


def _print_dist(name: str, values: list[float]) -> None:
    """Print distribution summary for a metric."""
    if not values:
        print(f"\n  {name}: (no data)")
        return
    values_sorted = sorted(values)
    n = len(values_sorted)
    total = sum(values_sorted)
    mean = total / n
    pcts = _pct(values_sorted)
    print(f"\n  {name}  (n={n:,})")
    print(f"    mean={mean:,.2f}  median={pcts['p50']:,.2f}")
    print(
        f"    p5={pcts['p5']:,.2f}  p25={pcts['p25']:,.2f}  "
        f"p75={pcts['p75']:,.2f}  p95={pcts['p95']:,.2f}"
    )
    print(f"    min={values_sorted[0]:,.2f}  max={values_sorted[-1]:,.2f}")


def _collection_key_source(cols: dict[str, str | None]) -> tuple[str, str]:
    """Describe which field is being used as the collection/specimen surrogate."""
    if cols.get("specimen_id"):
        return cols["specimen_id"], "native specimen_id"
    if cols.get("labresultoffset"):
        return cols["labresultoffset"], "surrogate grouped by relative minute offset"
    return cols["charttime"], "surrogate grouped by charttime"


def _duration_hours_expr(cols: dict[str, str | None]) -> str:
    """Duration expression in hours, preferring eICU's native offset column."""
    if cols.get("labresultoffset"):
        offset = _col_expr(cols, "labresultoffset")
        return f"(MAX({offset}) - MIN({offset})) / 60.0"
    chart = _col_expr(cols, "charttime")
    return f"EXTRACT(EPOCH FROM (MAX({chart}) - MIN({chart}))) / 3600.0"


def _collection_time_expr(cols: dict[str, str | None]) -> str:
    """Per-collection time coordinate in hours, preferring relative offsets."""
    if cols.get("labresultoffset"):
        offset = _col_expr(cols, "labresultoffset")
        return f"MIN({offset}) / 60.0"
    chart = _col_expr(cols, "charttime")
    return f"EXTRACT(EPOCH FROM MIN({chart})) / 3600.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribution stats for eICU lab_events_with_adm.")
    parser.add_argument("--parquet", type=Path, default=None)
    args = parser.parse_args()

    pq_path = args.parquet or DEFAULT_LAB_WITH_ADM_PARQUET
    if not pq_path.exists():
        print(f"Parquet not found: {pq_path}")
        return

    import duckdb

    pq = pq_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    cols = resolve_schema_columns(con, pq)
    collection_key = _specimen_expr(cols, prefer_offset=True)
    collection_col, collection_desc = _collection_key_source(cols)
    hadm = _col_expr(cols, "hadm_id")
    item = _col_expr(cols, "itemid")
    value = _col_expr(cols, "valuenum")
    chart = _col_expr(cols, "charttime")
    admit = _col_expr(cols, "admittime")
    duration_expr = _duration_hours_expr(cols)
    collection_time_expr = _collection_time_expr(cols)

    print("=" * 65)
    print("Schema / structure notes")
    print("=" * 65)
    print(f"  Admission/stay key: {cols['hadm_id']} (MIMIC `hadm_id` equivalent)")
    print(f"  Lab code column:    {cols['itemid']} (MIMIC `itemid` equivalent)")
    print(f"  Numeric value:      {cols['valuenum']} (MIMIC `valuenum` equivalent)")
    print(f"  Collection key:     {collection_col} ({collection_desc})")
    if cols.get("specimen_id") is None:
        print("  Note: no native `specimen_id`; same-minute labs within a stay are merged")
    if cols.get("labresultoffset"):
        print(
            f"  Timing model:       {cols['labresultoffset']} is the native minute offset; "
            f"{cols['charttime']} / {cols['admittime']} are reconstructed anchor timestamps"
        )
    else:
        print(f"  Timing model:       using {cols['charttime']} timestamps directly")

    row = con.execute(
        f"""
        SELECT COUNT(*) AS n_events,
               COUNT(DISTINCT {hadm}) AS n_admissions,
               COUNT(DISTINCT {item}) AS n_lab_codes,
               COUNT(DISTINCT CASE
                   WHEN {hadm} IS NOT NULL AND {collection_key} IS NOT NULL
                   THEN CAST({hadm} AS VARCHAR) || '|' || {collection_key}
               END) AS n_collections,
               SUM(CASE WHEN {hadm} IS NULL THEN 1 ELSE 0 END) AS hadm_null,
               SUM(CASE WHEN {value} IS NULL THEN 1 ELSE 0 END) AS value_null,
               SUM(CASE WHEN {chart} IS NULL THEN 1 ELSE 0 END) AS chart_null,
               SUM(CASE WHEN {admit} IS NULL THEN 1 ELSE 0 END) AS admit_null
        FROM read_parquet('{pq}')
        """
    ).fetchone()
    n_events, n_adm, n_lab_codes, n_collections, hadm_null, value_null, chart_null, admit_null = (
        int(x) for x in row
    )

    print("\n" + "=" * 65)
    print("Dataset-level totals")
    print("=" * 65)
    print(f"  Total lab events:   {n_events:,}")
    print(f"  Total ICU stays:    {n_adm:,}")
    print(f"  Total collections:  {n_collections:,}")
    print(f"  Distinct lab codes: {n_lab_codes:,}")
    print(f"  {cols['hadm_id']} null:     {hadm_null:,} ({100 * hadm_null / n_events:.1f}%)")
    print(f"  {cols['valuenum']} null:    {value_null:,} ({100 * value_null / n_events:.1f}%)")
    print(f"  {cols['charttime']} null:   {chart_null:,} ({100 * chart_null / n_events:.1f}%)")
    print(f"  {cols['admittime']} null:   {admit_null:,} ({100 * admit_null / n_events:.1f}%)")
    if cols.get("subject_id"):
        n_subjects = con.execute(
            f"SELECT COUNT(DISTINCT {_col_expr(cols, 'subject_id')}) FROM read_parquet('{pq}')"
        ).fetchone()[0]
        print(f"  Distinct patients:  {int(n_subjects):,}")

    if cols.get("labresultoffset"):
        offset = _col_expr(cols, "labresultoffset")
        neg_row = con.execute(
            f"""
            SELECT SUM(CASE WHEN {offset} < 0 THEN 1 ELSE 0 END) AS neg_events,
                   COUNT(DISTINCT CASE WHEN {offset} < 0 THEN {hadm} END) AS neg_adm
            FROM read_parquet('{pq}')
            """
        ).fetchone()
        neg_events = int(neg_row[0] or 0)
        neg_adm = int(neg_row[1] or 0)
        print(
            f"  Negative {cols['labresultoffset']}: {neg_events:,} "
            f"({100 * neg_events / n_events:.1f}% of events)"
        )
        print(
            f"  Stays with negative {cols['labresultoffset']}: {neg_adm:,} "
            f"({100 * neg_adm / n_adm:.1f}% of stays)"
        )

    print("\n" + "=" * 65)
    print("Per-collection (per merged draw / frame surrogate)")
    print("=" * 65)

    rows = con.execute(
        f"""
        SELECT COUNT(DISTINCT {item}) AS n_labs,
               COUNT(*) AS n_events
        FROM read_parquet('{pq}')
        WHERE {hadm} IS NOT NULL AND {collection_key} IS NOT NULL
        GROUP BY {hadm}, {collection_key}
        """
    ).fetchall()
    labs_per_collection = [float(row[0]) for row in rows]
    events_per_collection = [float(row[1]) for row in rows]
    vocab_fill_pct = [100.0 * value / max(n_lab_codes, 1) for value in labs_per_collection]

    _print_dist("n_labs / collection (distinct lab codes per merged draw)", labs_per_collection)
    _print_dist("n_events / collection (total rows per merged draw)", events_per_collection)
    _print_dist(
        f"lab-code coverage % / collection (of {n_lab_codes} observed lab codes)",
        vocab_fill_pct,
    )

    print("\n" + "=" * 65)
    print("Per-stay (per video candidate)")
    print("=" * 65)

    rows = con.execute(
        f"""
        SELECT COUNT(*) AS n_events,
               COUNT(DISTINCT {item}) AS n_labs,
               COUNT(DISTINCT {collection_key}) AS n_collections,
               {duration_expr} AS hours
        FROM read_parquet('{pq}')
        WHERE {hadm} IS NOT NULL
          AND {item} IS NOT NULL
          AND {collection_key} IS NOT NULL
        GROUP BY {hadm}
        """
    ).fetchall()

    events_per_adm = [float(row[0]) for row in rows]
    labs_per_adm = [float(row[1]) for row in rows]
    collections_per_adm = [float(row[2]) for row in rows]
    hours_per_adm = [float(row[3]) if row[3] is not None else 0.0 for row in rows]
    collections_per_hour = [
        count / hours for count, hours in zip(collections_per_adm, hours_per_adm) if hours > 0
    ]

    _print_dist("n_events / stay (total lab rows per video candidate)", events_per_adm)
    _print_dist("n_labs / stay (distinct lab codes per video candidate)", labs_per_adm)
    _print_dist("n_collections / stay (merged draws per video candidate)", collections_per_adm)
    _print_dist("hours / stay (first→last collection span)", hours_per_adm)
    _print_dist("collections / hour (sampling frequency)", collections_per_hour)

    print("\n" + "=" * 65)
    print("Time gaps between consecutive collections (per stay)")
    print("=" * 65)

    rows = con.execute(
        f"""
        WITH collections AS (
            SELECT {hadm} AS hadm_id_key,
                   {collection_key} AS collection_key,
                   {collection_time_expr} AS collection_time_hours
            FROM read_parquet('{pq}')
            WHERE {hadm} IS NOT NULL AND {collection_key} IS NOT NULL
            GROUP BY {hadm}, {collection_key}
        ),
        with_lag AS (
            SELECT *,
                   LAG(collection_time_hours) OVER (
                       PARTITION BY hadm_id_key
                       ORDER BY collection_time_hours
                   ) AS prev_collection_time_hours
            FROM collections
        )
        SELECT collection_time_hours - prev_collection_time_hours AS gap_hours
        FROM with_lag
        WHERE prev_collection_time_hours IS NOT NULL
        """
    ).fetchall()
    gaps = [float(row[0]) for row in rows]
    _print_dist("hours between consecutive collections", gaps)

    if cols.get("labresultoffset"):
        print("\n" + "=" * 65)
        print("Offset-based timing relative to admission")
        print("=" * 65)

        offset = _col_expr(cols, "labresultoffset")
        rows = con.execute(
            f"""
            SELECT MIN({offset}) / 60.0 AS first_hours,
                   MAX({offset}) / 60.0 AS last_hours
            FROM read_parquet('{pq}')
            WHERE {hadm} IS NOT NULL AND {offset} IS NOT NULL
            GROUP BY {hadm}
            """
        ).fetchall()
        first_offsets = [float(row[0]) for row in rows if row[0] is not None]
        last_offsets = [float(row[1]) for row in rows if row[1] is not None]
        _print_dist("first collection offset / stay (hours from admittime)", first_offsets)
        _print_dist("last collection offset / stay (hours from admittime)", last_offsets)

    con.close()
    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
