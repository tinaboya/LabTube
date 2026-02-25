from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from google.cloud import bigquery
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
SQL_DIR = SCRIPT_DIR / "sql"
PARQUET_SUFFIXES = {".parquet", ".pq"}

DATASET_CONFIG = {
    "mimic-iv": {
        "prefix": "mimiciv_",
        "default_sql": "mimiciv_lab_events.sql",
        "default_output_dir": PROJECT_ROOT / "MIMIC4",
        "output_by_sql": {
            "mimiciv_lab_events": PROJECT_ROOT / "MIMIC4" / "labevents.parquet",
            "mimiciv_all_admissions": PROJECT_ROOT / "MIMIC4" / "admissions.parquet",
            "mimiciv_vocab_labs": PROJECT_ROOT / "MIMIC4" / "d_labitems.csv",
        },
    },
    "eicu": {
        "prefix": "eicu_",
        "default_sql": "eicu_lab_events_with_adm.sql",
        "default_output_dir": PROJECT_ROOT / "eICU",
        "output_by_sql": {
            "eicu_lab_events_with_adm": PROJECT_ROOT / "eICU" / "lab_events_with_adm.parquet",
            "eicu_vocab_labs": PROJECT_ROOT / "eICU" / "vocab_labs.csv",
        },
    },
}


def get_client(project_id: str | None = None) -> bigquery.Client:
    effective_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or None
    # Uses Application Default Credentials (e.g., gcloud auth application-default login).
    return bigquery.Client(project=effective_project)


def load_sql(sql_path: Path) -> str:
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    return sql_path.read_text(encoding="utf-8")


def available_sql_files(dataset: str) -> list[Path]:
    prefix = DATASET_CONFIG[dataset]["prefix"]
    return sorted(SQL_DIR.glob(f"{prefix}*.sql"))


def resolve_sql_path(dataset: str, sql_selector: str | None) -> Path:
    if not sql_selector:
        return SQL_DIR / DATASET_CONFIG[dataset]["default_sql"]

    candidate = Path(sql_selector)
    if candidate.is_file():
        return candidate

    candidate_names = [candidate.name]
    if not candidate.name.endswith(".sql"):
        candidate_names.append(f"{candidate.name}.sql")

    for name in candidate_names:
        match = SQL_DIR / name
        if match.is_file():
            return match

    options = ", ".join(path.name for path in available_sql_files(dataset))
    raise FileNotFoundError(
        f"Could not find SQL '{sql_selector}'. Checked path directly and under {SQL_DIR}. "
        f"Available for dataset '{dataset}': {options}"
    )


def validate_sql_for_dataset(dataset: str, sql_path: Path) -> None:
    dataset_prefix = DATASET_CONFIG[dataset]["prefix"]
    if sql_path.name.startswith(dataset_prefix):
        return

    for other_dataset, config in DATASET_CONFIG.items():
        if other_dataset != dataset and sql_path.name.startswith(config["prefix"]):
            raise ValueError(
                f"SQL '{sql_path.name}' looks like '{other_dataset}' SQL but dataset is '{dataset}'."
            )


def default_output_path(dataset: str, sql_path: Path) -> Path:
    output_by_sql = DATASET_CONFIG[dataset]["output_by_sql"]
    return output_by_sql.get(
        sql_path.stem,
        DATASET_CONFIG[dataset]["default_output_dir"] / f"{sql_path.stem}.parquet",
    )


def get_bqstorage_client(use_bqstorage: bool):
    if not use_bqstorage:
        return None
    try:
        from google.cloud import bigquery_storage
        return bigquery_storage.BigQueryReadClient()
    except (ModuleNotFoundError, ImportError):
        try:
            from google.cloud import bigquery_storage_v1
            return bigquery_storage_v1.BigQueryReadClient()
        except (ModuleNotFoundError, ImportError):
            print(
                "google-cloud-bigquery-storage is unavailable; "
                "falling back to REST. Install/upgrade with:\n"
                "  pip install -U google-cloud-bigquery-storage"
            )
            return None


def infer_output_format(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        return "parquet"
    if suffix == ".csv":
        return "csv"
    raise ValueError(
        f"Unsupported output extension '{output_path.suffix}'. "
        "Use .parquet/.pq or .csv."
    )


def write_query_results_to_parquet(
    sql_text: str,
    output_path: Path,
    client: bigquery.Client,
    *,
    chunk_size: int,
    use_bqstorage: bool = True,
) -> int:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Executing query...")
    job = client.query(sql_text)
    result = job.result(page_size=chunk_size)
    print(f"Query complete. Job ID: {job.job_id}")

    bqstorage_client = get_bqstorage_client(use_bqstorage)
    batches = result.to_arrow_iterable(bqstorage_client=bqstorage_client)

    writer = None
    row_count = 0
    try:
        for batch in tqdm(batches, desc="Writing Parquet batches", unit="batch"):
            if isinstance(batch, pa.RecordBatch):
                n_rows = batch.num_rows
                if n_rows == 0:
                    continue
                table = pa.Table.from_batches([batch])
            else:
                n_rows = batch.num_rows
                if n_rows == 0:
                    continue
                table = batch

            if writer is None:
                writer = pq.ParquetWriter(output_path.as_posix(), table.schema, compression="zstd")
            writer.write_table(table)
            row_count += n_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Preserve schema (column names) for empty results.
        pd.DataFrame(columns=[field.name for field in result.schema]).to_parquet(
            output_path, index=False
        )

    return row_count


def write_query_results_to_csv(
    sql_text: str,
    output_path: Path,
    client: bigquery.Client,
    *,
    chunk_size: int,
    use_bqstorage: bool = True,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Executing query...")
    job = client.query(sql_text)
    result = job.result(page_size=chunk_size)
    print(f"Query complete. Job ID: {job.job_id}")

    bqstorage_client = get_bqstorage_client(use_bqstorage)
    frames = result.to_dataframe_iterable(bqstorage_client=bqstorage_client)

    wrote_header = False
    row_count = 0
    for frame in tqdm(frames, desc="Writing CSV chunks", unit="chunk"):
        if frame.empty:
            continue
        frame.to_csv(
            output_path,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
        )
        row_count += len(frame)
        wrote_header = True

    if not wrote_header:
        columns = [field.name for field in result.schema]
        pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    return row_count


def write_query_results(
    sql_text: str,
    output_path: Path,
    client: bigquery.Client,
    *,
    chunk_size: int,
    output_format: str,
    use_bqstorage: bool = True,
) -> int:
    if output_format == "parquet":
        return write_query_results_to_parquet(
            sql_text,
            output_path,
            client,
            chunk_size=chunk_size,
            use_bqstorage=use_bqstorage,
        )
    if output_format == "csv":
        return write_query_results_to_csv(
            sql_text,
            output_path,
            client,
            chunk_size=chunk_size,
            use_bqstorage=use_bqstorage,
        )
    raise ValueError(f"Unsupported output format: {output_format}")


def print_sql_options(dataset: str) -> None:
    options = available_sql_files(dataset)
    if not options:
        print(f"No SQL files found under {SQL_DIR} for dataset '{dataset}'.")
        return
    print(f"Available SQL files for dataset '{dataset}':")
    for path in options:
        print(f" - {path.name}")


def main() -> None:
    parser = ArgumentParser(description="Download query results from BigQuery using SQL in scripts/sql.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIG.keys()),
        default="mimic-iv",
        help="Dataset family used to select SQL defaults and output naming.",
    )
    parser.add_argument(
        "sql_path",
        nargs="?",
        default=None,
        help="Optional SQL path/name (for backward compatibility).",
    )
    parser.add_argument(
        "--sql",
        type=str,
        default=None,
        help="SQL file name/stem in scripts/sql, or explicit path. Example: mimiciv_lab_events.sql",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (.parquet/.pq or .csv). If omitted, a dataset-aware default is used.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default=None,
        help="Output format override. If omitted, inferred from --output extension.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="BigQuery billing project. Defaults to GOOGLE_CLOUD_PROJECT from environment.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk when streaming query results from BigQuery.",
    )
    parser.add_argument(
        "--list-sql",
        action="store_true",
        help="List available SQL files for the selected dataset and exit.",
    )
    parser.add_argument(
        "--no-bqstorage",
        action="store_true",
        help="Disable the BigQuery Storage API reader.",
    )

    args = parser.parse_args()

    if args.sql and args.sql_path:
        parser.error("Use either positional sql_path or --sql, not both.")

    if args.list_sql:
        print_sql_options(args.dataset)
        return

    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    sql_selector = args.sql or args.sql_path
    try:
        sql_path = resolve_sql_path(args.dataset, sql_selector)
        validate_sql_for_dataset(args.dataset, sql_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output_path = args.output or default_output_path(args.dataset, sql_path)
    output_path = output_path.resolve()
    try:
        output_format = args.format or infer_output_format(output_path)
    except ValueError as exc:
        parser.error(str(exc))

    load_dotenv(DOTENV_PATH, override=False)
    sql_text = load_sql(sql_path)
    try:
        client = get_client(project_id=args.project)
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Dataset: {args.dataset}")
    print(f"SQL: {sql_path.resolve()}")
    print(f"Output ({output_format.upper()}): {output_path}")

    row_count = write_query_results(
        sql_text,
        output_path,
        client,
        chunk_size=args.chunk_size,
        output_format=output_format,
        use_bqstorage=not args.no_bqstorage,
    )
    print(f"Wrote {row_count:,} rows to {output_path}")


if __name__ == "__main__":
    main()
