"""
Inspect pipeline steps and print summary statistics for each artifact.
Run after (or instead of) processing to validate outputs and compare across runs.

Usage:
  python scripts/pipeline_stats.py                    # all steps (default paths)
  python scripts/pipeline_stats.py --step lab_parquet
  python scripts/pipeline_stats.py --step lab_with_adm --path MIMIC4/lab_events_with_adm.parquet
  python scripts/pipeline_stats.py --output stats.json  # write JSON for each step
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing config when run as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_ADMISSIONS_PARQUET,
    DEFAULT_LAB_IMAGES,
    DEFAULT_LAB_WITH_ADM_PARQUET,
    DEFAULT_PARQUET,
    ROOT,
)


def _file_size_mb(p: Path) -> float | None:
    if not p.exists():
        return None
    return p.stat().st_size / (1024 * 1024)


def get_stats_lab_parquet(path: Path) -> dict | None:
    """Stats for labevents.parquet."""
    path = Path(path)
    if not path.exists():
        return None
    import polars as pl

    df = pl.read_parquet(path)
    n = len(df)
    cols = df.columns
    nulls = {c: df[c].null_count() for c in cols if df[c].null_count() > 0}
    # Key columns for reporting
    key_cols = ["subject_id", "hadm_id", "specimen_id", "itemid", "charttime", "valuenum"]
    key_nulls = {c: nulls.get(c, 0) for c in key_cols if c in cols}
    return {
        "step": "lab_parquet",
        "path": str(path),
        "rows": n,
        "columns": len(cols),
        "column_names": cols,
        "null_counts_key": key_nulls,
        "null_pct_key": {k: round(100 * v / n, 2) for k, v in key_nulls.items()} if n else {},
        "file_size_mb": round(_file_size_mb(path) or 0, 2),
    }


def get_stats_admissions_parquet(path: Path) -> dict | None:
    """Stats for admissions.parquet."""
    path = Path(path)
    if not path.exists():
        return None
    import polars as pl

    df = pl.read_parquet(path)
    n = len(df)
    cols = df.columns
    nulls = {c: df[c].null_count() for c in cols if df[c].null_count() > 0}
    return {
        "step": "admissions_parquet",
        "path": str(path),
        "rows": n,
        "columns": len(cols),
        "column_names": cols,
        "null_counts": nulls,
        "unique_subject_id": df["subject_id"].n_unique() if "subject_id" in cols else None,
        "unique_hadm_id": df["hadm_id"].n_unique() if "hadm_id" in cols else None,
        "file_size_mb": round(_file_size_mb(path) or 0, 2),
    }


def get_stats_lab_with_adm(path: Path) -> dict | None:
    """Stats for lab_events_with_adm.parquet (after join + backfill)."""
    path = Path(path)
    if not path.exists():
        return None
    import polars as pl

    df = pl.read_parquet(path)
    n = len(df)
    cols = df.columns
    hadm_null = df["hadm_id"].null_count() if "hadm_id" in cols else 0
    spec_null = df["specimen_id"].null_count() if "specimen_id" in cols else 0
    adm_null = df["admittime"].null_count() if "admittime" in cols else 0
    return {
        "step": "lab_with_adm",
        "path": str(path),
        "rows": n,
        "columns": len(cols),
        "unique_subject_id": df["subject_id"].n_unique() if "subject_id" in cols else None,
        "unique_hadm_id": df["hadm_id"].n_unique() if "hadm_id" in cols else None,
        "unique_specimen_id": df["specimen_id"].n_unique() if "specimen_id" in cols else None,
        "hadm_id_null_count": hadm_null,
        "hadm_id_null_pct": round(100 * hadm_null / n, 2) if n else 0,
        "specimen_id_null_count": spec_null,
        "admittime_null_count": adm_null,
        "admittime_null_pct": round(100 * adm_null / n, 2) if n else 0,
        "file_size_mb": round(_file_size_mb(path) or 0, 2),
    }


def get_stats_lab_images(dir_path: Path) -> dict | None:
    """Stats for lab_images/ (specimen_*.npy)."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return None
    import numpy as np

    npy_files = sorted(dir_path.glob("specimen_*.npy"))
    if not npy_files:
        return {
            "step": "lab_images",
            "path": str(dir_path),
            "count": 0,
            "message": "No specimen_*.npy files found",
        }
    shapes = []
    for f in npy_files:
        try:
            arr = np.load(f)
            shapes.append(arr.shape)
        except Exception:
            shapes.append(None)
    valid_shapes = [s for s in shapes if s is not None]
    return {
        "step": "lab_images",
        "path": str(dir_path),
        "count": len(npy_files),
        "shapes_sample": valid_shapes[:5] if valid_shapes else [],
        "shape_unique": list({s for s in valid_shapes}),
        "all_same_shape": len(set(valid_shapes)) <= 1 if valid_shapes else False,
    }


def format_stats(d: dict | None, verbose: bool = True) -> str:
    """Turn a stats dict into a readable block. If d is None, return a missing-file message."""
    if d is None:
        return "(file or directory not found)"
    step = d.get("step", "?")
    lines = [f"--- {step} ---", f"  path: {d.get('path', '?')}"]
    for k, v in d.items():
        if k in ("step", "path", "column_names"):
            continue
        if k == "null_counts_key" or k == "null_counts" or k == "null_pct_key":
            if verbose and v:
                lines.append(f"  {k}: {v}")
            continue
        if v is not None:
            lines.append(f"  {k}: {v}")
    if verbose and "column_names" in d:
        lines.append(f"  columns ({len(d['column_names'])}): {d['column_names'][:10]}{'...' if len(d['column_names']) > 10 else ''}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print summary statistics for each pipeline step (read-only)."
    )
    parser.add_argument(
        "--step",
        choices=["all", "lab_parquet", "admissions_parquet", "lab_with_adm", "lab_images"],
        default="all",
        help="Which step to report (default: all)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Override path for the chosen step (e.g. path to a specific parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write stats JSON to this file (one object per step key)",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Omit null counts and long lists",
    )
    args = parser.parse_args()

    steps_run = []
    if args.step == "all":
        steps_run = ["lab_parquet", "admissions_parquet", "lab_with_adm", "lab_images"]
    else:
        steps_run = [args.step]

    getters = {
        "lab_parquet": (get_stats_lab_parquet, args.path or DEFAULT_PARQUET),
        "admissions_parquet": (get_stats_admissions_parquet, args.path or DEFAULT_ADMISSIONS_PARQUET),
        "lab_with_adm": (get_stats_lab_with_adm, args.path or DEFAULT_LAB_WITH_ADM_PARQUET),
        "lab_images": (get_stats_lab_images, args.path or DEFAULT_LAB_IMAGES),
    }

    out_data = {}
    for name in steps_run:
        getter, path = getters[name]
        path = path if isinstance(path, Path) else ROOT / path
        d = getter(path)
        out_data[name] = d
        print(format_stats(d, verbose=not args.brief))
        print()

    if args.output:
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # JSON-serializable: drop any non-serializable values
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sanitize(x) for x in obj]
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            return str(obj)

        with open(args.output, "w") as f:
            json.dump(sanitize(out_data), f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
