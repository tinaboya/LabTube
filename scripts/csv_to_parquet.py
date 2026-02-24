"""
Convert MIMIC-IV labevents.csv and admissions.csv to Parquet using DuckDB (streaming, low memory).
Borrows approach from M4: read_csv_auto + COPY TO Parquet with MIMIC null handling.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb

# Allow importing config when run as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def labevents_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    *,
    memory_limit: str = "4GB",
    threads: int = 2,
) -> Path:
    """
    Stream labevents CSV to Parquet. Does not load full file into memory.

    Args:
        csv_path: Path to labevents.csv
        parquet_path: Output path for labevents.parquet
        memory_limit: DuckDB memory_limit (e.g. "4GB")
        threads: DuckDB thread count for the conversion

    Returns:
        Path to the written Parquet file.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA threads={threads}")
        # MIMIC uses '___' for masked values; treat as null for consistency
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    sample_size=-1,
                    auto_detect=true,
                    nullstr=['', 'NULL', 'NA', 'N/A', '___'],
                    ignore_errors=false
                )
            )
            TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
    finally:
        con.close()

    return parquet_path


def admissions_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    *,
    memory_limit: str = "4GB",
    threads: int = 2,
) -> Path:
    """
    Convert admissions.csv to Parquet. Keeps subject_id, hadm_id, admittime, edregtime, race.

    Args:
        csv_path: Path to admissions.csv
        parquet_path: Output path for admissions.parquet
        memory_limit: DuckDB memory_limit
        threads: DuckDB thread count

    Returns:
        Path to the written Parquet file.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA threads={threads}")
        con.execute(
            f"""
            COPY (
                SELECT subject_id, hadm_id, admittime, edregtime, race
                FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    sample_size=-1,
                    auto_detect=true,
                    nullstr=['', 'NULL', 'NA', 'N/A', '___'],
                    ignore_errors=false
                )
            )
            TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
    finally:
        con.close()

    return parquet_path


def main() -> None:
    import argparse

    from config import DEFAULT_ADMISSIONS_CSV, DEFAULT_ADMISSIONS_PARQUET, DEFAULT_MIMIC4

    parser = argparse.ArgumentParser(description="Convert MIMIC-IV CSV to Parquet")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["lab", "admissions"],
        default="lab",
        help="lab: labevents.csv → labevents.parquet; admissions: admissions.csv → admissions.parquet",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input CSV (default: MIMIC4/labevents.csv or MIMIC4/admissions.csv by mode)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Parquet path",
    )
    parser.add_argument("--memory", type=str, default=os.environ.get("M4_DUCKDB_MEM", "4GB"))
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("M4_DUCKDB_THREADS", "2")),
    )
    args = parser.parse_args()

    if args.mode == "lab":
        csv_path = args.csv or (DEFAULT_MIMIC4 / "labevents.csv")
        out_path = args.out or (DEFAULT_MIMIC4 / "labevents.parquet")
        out = labevents_csv_to_parquet(
            csv_path, out_path, memory_limit=args.memory, threads=args.threads
        )
    else:
        csv_path = args.csv or DEFAULT_ADMISSIONS_CSV
        out_path = args.out or DEFAULT_ADMISSIONS_PARQUET
        out = admissions_csv_to_parquet(
            csv_path, out_path, memory_limit=args.memory, threads=args.threads
        )
    print(f"Wrote {out}")
    # Quick stats (read back for row count and size)
    try:
        from pipeline_stats import get_stats_lab_parquet, get_stats_admissions_parquet, format_stats
        getter = get_stats_lab_parquet if args.mode == "lab" else get_stats_admissions_parquet
        d = getter(out)
        if d:
            print(format_stats(d, verbose=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
