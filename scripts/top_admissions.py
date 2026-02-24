"""
List admissions (hadm_id) by lab event count — for the whole stay or first 24h only.
Uses lab_events_with_adm.parquet (must have hadm_id, charttime, admittime).

Usage:
  python scripts/top_admissions.py --top 20
  python scripts/top_admissions.py --top 50 --first-24h
  python scripts/top_admissions.py --parquet MIMIC4/lab_events_with_adm.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_LAB_WITH_ADM_PARQUET, DEFAULT_MIMIC4


def _choose_parquet(parquet_path: Path | None) -> Path:
    if parquet_path is not None and parquet_path.exists():
        return Path(parquet_path)
    return DEFAULT_LAB_WITH_ADM_PARQUET


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List top admissions by lab event count (whole stay or first 24h)."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of admissions to list (default: 20)",
    )
    parser.add_argument(
        "--first-24h",
        action="store_true",
        help="Restrict to lab events in the first 24 hours from admission only",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Lab events with admission Parquet (default: lab_events_with_adm.parquet)",
    )
    args = parser.parse_args()

    parquet_path = _choose_parquet(args.parquet)
    if not parquet_path.exists():
        print(f"Parquet not found: {parquet_path}")
        return

    import duckdb

    pq = parquet_path.as_posix()
    con = duckdb.connect()

    if args.first_24h:
        # Filter: charttime >= admittime AND charttime < admittime + 24h
        sql = f"""
        WITH first_24h AS (
            SELECT hadm_id, subject_id, charttime, admittime, itemid, specimen_id
            FROM read_parquet('{pq}')
            WHERE hadm_id IS NOT NULL
              AND admittime IS NOT NULL
              AND charttime IS NOT NULL
              AND charttime >= admittime
              AND charttime < admittime + INTERVAL '24 hours'
        )
        SELECT hadm_id,
               COUNT(*) AS n_events,
               COUNT(DISTINCT itemid) AS n_labs,
               COUNT(DISTINCT specimen_id) AS n_specimens
        FROM first_24h
        GROUP BY hadm_id
        ORDER BY n_events DESC
        LIMIT {args.top}
        """
        label = "first 24h"
    else:
        sql = f"""
        SELECT hadm_id,
               COUNT(*) AS n_events,
               COUNT(DISTINCT itemid) AS n_labs,
               COUNT(DISTINCT specimen_id) AS n_specimens
        FROM read_parquet('{pq}')
        WHERE hadm_id IS NOT NULL
        GROUP BY hadm_id
        ORDER BY n_events DESC
        LIMIT {args.top}
        """
        label = "whole stay"

    try:
        rows = con.execute(sql).fetchall()
        # Total distinct admissions in the data (for clarity when fewer than --top)
        if args.first_24h:
            total_sql = f"""
            SELECT COUNT(DISTINCT hadm_id) FROM (
                SELECT hadm_id FROM read_parquet('{pq}')
                WHERE hadm_id IS NOT NULL AND admittime IS NOT NULL AND charttime IS NOT NULL
                  AND charttime >= admittime AND charttime < admittime + INTERVAL '24 hours'
            ) t
            """
        else:
            total_sql = f"SELECT COUNT(DISTINCT hadm_id) FROM read_parquet('{pq}') WHERE hadm_id IS NOT NULL"
        total = con.execute(total_sql).fetchone()[0]
    finally:
        con.close()

    print(f"Parquet: {parquet_path}")
    if len(rows) < total:
        print(f"Top {len(rows)} of {total} admissions by lab event count ({label})")
    else:
        print(f"Top {len(rows)} admissions by lab event count ({label})")
    print("-" * 60)
    for r in rows:
        hadm_id, n_events, n_labs, n_specimens = r
        print(f"  hadm_id={hadm_id}  n_events={n_events}  n_labs={n_labs}  n_specimens={n_specimens}")
    print("-" * 60)
    print("Use these hadm_id values when building admission-level (T×H×W) videos.")


if __name__ == "__main__":
    main()
