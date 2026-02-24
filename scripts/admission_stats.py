"""
Distribution analysis of the lab_events_with_adm dataset.

Prints summary statistics (count, mean, median, percentiles, min, max) for:

  Dataset level:
    - Total admissions, total lab events, total specimens

  Per-specimen (= per image / per frame):
    - n_labs / specimen        — how many lab types per frame

  Per-admission (= per video):
    - n_events / admission     — total lab events per video
    - n_labs / admission       — distinct lab types per video
    - n_specimens / admission  — frames per video
    - hours / admission        — duration of each admission (first→last charttime)
    - specimens / hour         — sampling frequency per admission

  Additional:
    - hours between specimens  — time gap between consecutive specimens (per admission)
    - n_events / specimen      — how many values per frame (including repeats of same lab)
    - grid fill % / specimen   — what fraction of 256 grid cells are filled per frame
    - valuenum null %          — how many lab events have no numeric value

Usage:
    python scripts/admission_stats.py
    python scripts/admission_stats.py --parquet MIMIC4/lab_events_with_adm.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_LAB_WITH_ADM_PARQUET


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
    print(f"    p5={pcts['p5']:,.2f}  p25={pcts['p25']:,.2f}  p75={pcts['p75']:,.2f}  p95={pcts['p95']:,.2f}")
    print(f"    min={values_sorted[0]:,.2f}  max={values_sorted[-1]:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribution stats for lab_events_with_adm.")
    parser.add_argument("--parquet", type=Path, default=None)
    args = parser.parse_args()

    pq_path = args.parquet or DEFAULT_LAB_WITH_ADM_PARQUET
    if not pq_path.exists():
        print(f"Parquet not found: {pq_path}")
        return

    import duckdb

    pq = pq_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()

    # ===== Dataset-level totals =====
    row = con.execute(f"""
        SELECT COUNT(*) AS n_events,
               COUNT(DISTINCT hadm_id) AS n_admissions,
               COUNT(DISTINCT specimen_id) AS n_specimens,
               SUM(CASE WHEN hadm_id IS NULL THEN 1 ELSE 0 END) AS hadm_null,
               SUM(CASE WHEN valuenum IS NULL THEN 1 ELSE 0 END) AS valuenum_null
        FROM read_parquet('{pq}')
    """).fetchone()
    n_events, n_adm, n_spec, hadm_null, valuenum_null = (int(x) for x in row)

    print("=" * 65)
    print("Dataset-level totals")
    print("=" * 65)
    print(f"  Total lab events:   {n_events:,}")
    print(f"  Total admissions:   {n_adm:,}")
    print(f"  Total specimens:    {n_spec:,}")
    print(f"  hadm_id null:       {hadm_null:,} ({100*hadm_null/n_events:.1f}%)")
    print(f"  valuenum null:      {valuenum_null:,} ({100*valuenum_null/n_events:.1f}%)")

    # ===== Per-specimen stats (= per image / per frame) =====
    print("\n" + "=" * 65)
    print("Per-specimen (per frame / per image)")
    print("=" * 65)

    # n_labs / specimen
    rows = con.execute(f"""
        SELECT COUNT(DISTINCT itemid) AS n_labs,
               COUNT(*) AS n_events
        FROM read_parquet('{pq}')
        WHERE specimen_id IS NOT NULL AND hadm_id IS NOT NULL
        GROUP BY specimen_id
    """).fetchall()
    labs_per_spec = [float(r[0]) for r in rows]
    events_per_spec = [float(r[1]) for r in rows]

    _print_dist("n_labs / specimen (distinct lab types per frame)", labs_per_spec)
    _print_dist("n_events / specimen (total values per frame)", events_per_spec)

    # grid fill % (out of 256)
    fill_pct = [100.0 * v / 256.0 for v in labs_per_spec]
    _print_dist("grid fill % / specimen (of 256 cells)", fill_pct)

    # ===== Per-admission stats (= per video) =====
    print("\n" + "=" * 65)
    print("Per-admission (per video)")
    print("=" * 65)

    # n_events, n_labs, n_specimens per admission
    rows = con.execute(f"""
        SELECT COUNT(*) AS n_events,
               COUNT(DISTINCT itemid) AS n_labs,
               COUNT(DISTINCT specimen_id) AS n_specimens,
               EXTRACT(EPOCH FROM (MAX(charttime) - MIN(charttime))) / 3600.0 AS hours
        FROM read_parquet('{pq}')
        WHERE hadm_id IS NOT NULL AND charttime IS NOT NULL
        GROUP BY hadm_id
    """).fetchall()

    events_per_adm = [float(r[0]) for r in rows]
    labs_per_adm = [float(r[1]) for r in rows]
    specs_per_adm = [float(r[2]) for r in rows]
    hours_per_adm = [float(r[3]) if r[3] is not None else 0.0 for r in rows]
    specs_per_hour = [s / h if h > 0 else 0.0 for s, h in zip(specs_per_adm, hours_per_adm)]

    _print_dist("n_events / admission (total lab events per video)", events_per_adm)
    _print_dist("n_labs / admission (distinct lab types per video)", labs_per_adm)
    _print_dist("n_specimens / admission (frames per video)", specs_per_adm)
    _print_dist("hours / admission (duration first→last charttime)", hours_per_adm)
    _print_dist("specimens / hour (sampling frequency)", [v for v in specs_per_hour if v > 0])

    # ===== Time gaps between consecutive specimens per admission =====
    print("\n" + "=" * 65)
    print("Time gaps between consecutive specimens (per admission)")
    print("=" * 65)

    rows = con.execute(f"""
        WITH spec_times AS (
            SELECT hadm_id, specimen_id, MIN(charttime) AS spec_time
            FROM read_parquet('{pq}')
            WHERE hadm_id IS NOT NULL AND specimen_id IS NOT NULL AND charttime IS NOT NULL
            GROUP BY hadm_id, specimen_id
        ),
        with_lag AS (
            SELECT *,
                LAG(spec_time) OVER (PARTITION BY hadm_id ORDER BY spec_time) AS prev_time
            FROM spec_times
        )
        SELECT EXTRACT(EPOCH FROM (spec_time - prev_time)) / 3600.0 AS gap_hours
        FROM with_lag
        WHERE prev_time IS NOT NULL
    """).fetchall()

    gaps = [float(r[0]) for r in rows]
    _print_dist("hours between consecutive specimens", gaps)

    con.close()
    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
