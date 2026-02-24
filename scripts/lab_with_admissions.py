"""
Join lab events with admissions, sort by subject+charttime, and backfill hadm_id
for labs in the 0–24h before admission (labmae_meds-style). Optionally join lab vocab labels.

Full runs use a **chunked** approach: process subject_ids in batches so each batch
fits in memory, then merge the batch Parquet files into one output. This avoids OOM
on machines with limited RAM (e.g. 16GB laptop with 158M lab rows).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Allow importing config when run as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_D_LABITEMS, DEFAULT_MIMIC4


def _q(s: str) -> str:
    """Escape single quotes for SQL string literals."""
    return s.replace("'", "''")


def _backfill_sql(
    lab_source: str,
    adm_parquet: str,
    *,
    vocab_csv: str | None = None,
    filter_null_hadm: bool = False,
) -> str:
    """Return the full SELECT SQL (no COPY wrapper) for join + sort + backfill.

    ``lab_source`` is the FROM clause for lab rows, e.g.
        read_parquet('...')
        (SELECT * FROM read_parquet('...') WHERE subject_id IN (...))
    """
    core = f"""
    WITH joined AS (
      SELECT lab.*, adm.admittime
      FROM {lab_source} lab
      LEFT JOIN read_parquet('{adm_parquet}') adm ON lab.hadm_id = adm.hadm_id
    ),
    with_filled AS (
      SELECT *,
        LAST_VALUE(admittime IGNORE NULLS) OVER (
          PARTITION BY subject_id ORDER BY charttime NULLS LAST
          ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        ) AS admittime_filled,
        (
          (LAST_VALUE(admittime IGNORE NULLS) OVER (
            PARTITION BY subject_id ORDER BY charttime NULLS LAST
            ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
          ) >= charttime)
          AND (LAST_VALUE(admittime IGNORE NULLS) OVER (
            PARTITION BY subject_id ORDER BY charttime NULLS LAST
            ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
          ) - charttime <= INTERVAL 24 HOURS)
        ) AS before_admit_24h
      FROM joined
    ),
    with_hadm AS (
      SELECT *,
        CASE
          WHEN before_admit_24h AND hadm_id IS NULL THEN
            LAST_VALUE(hadm_id IGNORE NULLS) OVER (
              PARTITION BY subject_id ORDER BY charttime NULLS LAST
              ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            )
          ELSE hadm_id
        END AS hadm_id_filled
      FROM with_filled
    ),
    result AS (
      SELECT * EXCLUDE (hadm_id, admittime, before_admit_24h, admittime_filled, hadm_id_filled),
        hadm_id_filled AS hadm_id,
        admittime_filled AS admittime
      FROM with_hadm
    )
    """
    where = " WHERE hadm_id IS NOT NULL" if filter_null_hadm else ""
    order = " ORDER BY subject_id, charttime"

    if vocab_csv:
        final = f"""
    SELECT result.*, vocab.label AS lab_label
    FROM result
    LEFT JOIN read_csv_auto('{vocab_csv}') vocab ON result.itemid = vocab.itemid
    {where}{order}"""
    else:
        final = f"SELECT result.* FROM result{where}{order}"

    return core + final


def _run_chunked_path(
    lab_path: Path,
    adm_path: Path,
    out_path: Path,
    *,
    batch_size: int = 1000,
    limit_rows: int | None = None,
    vocab_csv_path: Path | None = None,
    skip_vocab: bool = False,
    filter_null_hadm: bool = False,
) -> Path:
    """Process lab→admission join in subject_id batches so each batch fits in memory.

    1. Get sorted distinct subject_ids from lab parquet.
    2. For each batch of ``batch_size`` subject_ids: run join+sort+backfill in DuckDB,
       write a temp Parquet file.
    3. Merge all temp Parquets into one output (simple concatenation; already sorted).
    """
    import duckdb

    lp = _q(lab_path.resolve().as_posix())
    ap = _q(adm_path.resolve().as_posix())
    out_posix = out_path.resolve().as_posix()
    vp = _q(vocab_csv_path.resolve().as_posix()) if (not skip_vocab and vocab_csv_path and vocab_csv_path.exists()) else None

    memory_limit = os.environ.get("DUCKDB_MEMORY_LIMIT", "2GB")
    threads = os.environ.get("DUCKDB_THREADS", "2")

    # --- Step 1: get sorted distinct subject_ids ---
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        lab_source_for_ids = f"(SELECT * FROM read_parquet('{lp}') LIMIT {limit_rows})" if limit_rows else f"read_parquet('{lp}')"
        rows = con.execute(
            f"SELECT DISTINCT subject_id FROM {lab_source_for_ids} WHERE subject_id IS NOT NULL ORDER BY subject_id"
        ).fetchall()
    finally:
        con.close()

    all_subject_ids = [int(r[0]) for r in rows]
    n_subjects = len(all_subject_ids)
    n_batches = (n_subjects + batch_size - 1) // batch_size
    print(f"Chunked processing: {n_subjects:,} subject_ids in {n_batches} batches (batch_size={batch_size})")

    # --- Step 2: process each batch ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="latte_adm_", dir=out_path.parent))
    try:
        batch_files: list[str] = []
        for i in range(0, n_subjects, batch_size):
            batch_ids = all_subject_ids[i : i + batch_size]
            batch_num = i // batch_size + 1

            # Build comma-separated list of subject_ids for IN clause
            ids_csv = ",".join(str(sid) for sid in batch_ids)
            if limit_rows is not None:
                lab_src = f"(SELECT * FROM read_parquet('{lp}') WHERE subject_id IN ({ids_csv}) LIMIT {limit_rows})"
            else:
                lab_src = f"(SELECT * FROM read_parquet('{lp}') WHERE subject_id IN ({ids_csv}))"

            sql = _backfill_sql(lab_src, ap, vocab_csv=vp, filter_null_hadm=filter_null_hadm)

            batch_file = tmp_dir / f"batch_{batch_num:05d}.parquet"
            batch_files.append(_q(batch_file.resolve().as_posix()))

            con = duckdb.connect()
            try:
                con.execute(f"SET memory_limit='{memory_limit}'")
                con.execute(f"SET threads={threads}")
                con.execute("SET preserve_insertion_order=false")
                con.execute(f"COPY ({sql}) TO '{_q(batch_file.resolve().as_posix())}' (FORMAT PARQUET)")
            finally:
                con.close()

            print(f"  Batch {batch_num}/{n_batches}: subject_ids {batch_ids[0]}–{batch_ids[-1]} done")

        # --- Step 3: merge all batch files into one output ---
        print("Merging batches...")
        batch_glob = _q((tmp_dir / "batch_*.parquet").resolve().as_posix())
        con = duckdb.connect()
        try:
            con.execute(f"SET memory_limit='{memory_limit}'")
            con.execute(f"SET threads={threads}")
            con.execute("SET preserve_insertion_order=false")
            # Batches are already sorted by (subject_id, charttime) and subject_ids
            # are in ascending order across batches, so simple union preserves global order.
            con.execute(f"COPY (SELECT * FROM read_parquet('{batch_glob}')) TO '{_q(out_posix)}' (FORMAT PARQUET)")
        finally:
            con.close()
    finally:
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Wrote {out_path}")

    # --- Stats ---
    con = duckdb.connect()
    try:
        row = con.execute(
            f"SELECT COUNT(*) AS n, COUNT(DISTINCT hadm_id) AS u, "
            f"SUM(CASE WHEN hadm_id IS NULL THEN 1 ELSE 0 END) AS nulls "
            f"FROM read_parquet('{_q(out_posix)}')"
        ).fetchone()
        n, u, nulls = int(row[0]), int(row[1]), int(row[2])
    finally:
        con.close()
    print(f"  Stats: rows={n:,}, unique hadm_id={u:,}, hadm_id null={nulls:,} ({100 * nulls / n:.1f}%)")
    return out_path


def run_lab_with_admissions(
    lab_parquet_path: str | Path,
    admissions_parquet_path: str | Path,
    output_path: str | Path,
    *,
    vocab_csv_path: str | Path | None = None,
    skip_vocab: bool = False,
    filter_null_hadm: bool = False,
    limit_rows: int | None = None,
    batch_size: int = 1000,
) -> Path:
    """
    Read lab + admissions Parquet, join, sort, backfill hadm_id for 0–24h before admit, write Parquet.

    By default uses **chunked processing** (subject_id batches) so the full 158M-row
    lab table can be processed on a machine with limited RAM (e.g. 16GB laptop).

    Args:
        lab_parquet_path: labevents.parquet (must have subject_id, hadm_id, charttime, specimen_id, itemid, etc.)
        admissions_parquet_path: admissions.parquet (hadm_id, admittime)
        output_path: where to write lab_events_with_adm.parquet
        vocab_csv_path: if set and not skip_vocab, join d_labitems to add lab_label
        skip_vocab: if True, do not join vocab
        filter_null_hadm: if True, drop rows where hadm_id is null after backfill
        limit_rows: if set, only load and process this many lab rows (for fast testing)
        batch_size: number of subject_ids per batch (default 1000; lower = less memory)

    Returns:
        Path to the written Parquet file.
    """
    lab_path = Path(lab_parquet_path)
    adm_path = Path(admissions_parquet_path)
    out_path = Path(output_path)
    vc = Path(vocab_csv_path) if vocab_csv_path else None

    return _run_chunked_path(
        lab_path,
        adm_path,
        out_path,
        batch_size=batch_size,
        limit_rows=limit_rows,
        vocab_csv_path=vc,
        skip_vocab=skip_vocab,
        filter_null_hadm=filter_null_hadm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join lab events with admissions and backfill hadm_id for 0–24h before admit."
    )
    parser.add_argument(
        "--lab-parquet",
        type=Path,
        default=DEFAULT_MIMIC4 / "labevents.parquet",
        help="Lab events Parquet path",
    )
    parser.add_argument(
        "--adm-parquet",
        type=Path,
        default=DEFAULT_MIMIC4 / "admissions.parquet",
        help="Admissions Parquet path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_MIMIC4 / "lab_events_with_adm.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--vocab-csv",
        type=Path,
        default=DEFAULT_D_LABITEMS,
        help="Lab vocabulary CSV (d_labitems) to add lab_label; use with --no-skip-vocab",
    )
    parser.add_argument(
        "--skip-vocab",
        action="store_true",
        help="Do not join lab labels",
    )
    parser.add_argument(
        "--filter-null-hadm",
        action="store_true",
        help="Drop rows with null hadm_id after backfill",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        metavar="N",
        help="Only load and process first N lab rows (for fast testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of subject_ids per batch (default: 1000; lower = less memory)",
    )
    args = parser.parse_args()

    run_lab_with_admissions(
        args.lab_parquet,
        args.adm_parquet,
        args.output,
        vocab_csv_path=args.vocab_csv,
        skip_vocab=args.skip_vocab,
        filter_null_hadm=args.filter_null_hadm,
        limit_rows=args.limit_rows,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
