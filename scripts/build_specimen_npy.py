"""
Build .npy grid(s) for one or more specimen IDs by querying Parquet only for those IDs.
Uses the same grid layout and options as the main pipeline (top-K itemids, raw values, etc.).

Usage:
  python scripts/build_specimen_npy.py --specimen 83925500
  python scripts/build_specimen_npy.py --specimens 83925500 41650253 45668018
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_LAB_IMAGES,
    DEFAULT_LAB_WITH_ADM_PARQUET,
    DEFAULT_MIMIC4,
    DEFAULT_PARQUET,
)
from lab_grid import build_itemid_to_grid_index, get_top_itemids_from_parquet
from specimen_to_image import specimen_events_to_grid


def _choose_parquet(parquet_path: Path | None) -> Path:
    if parquet_path is not None and parquet_path.exists():
        return Path(parquet_path)
    if DEFAULT_LAB_WITH_ADM_PARQUET.exists():
        return DEFAULT_LAB_WITH_ADM_PARQUET
    return DEFAULT_PARQUET


def fetch_events_for_specimen(parquet_path: Path, specimen_id: int) -> pd.DataFrame | None:
    """Load all lab events for one specimen_id from Parquet."""
    import duckdb

    pq = Path(parquet_path).as_posix()
    con = duckdb.connect()
    try:
        df = con.execute(
            """
            SELECT specimen_id, subject_id, charttime, itemid, valuenum, value,
                   ref_range_lower, ref_range_upper
            FROM read_parquet(?)
            WHERE specimen_id = ?
            """,
            [pq, specimen_id],
        ).fetchdf()
    finally:
        con.close()
    return df if df is not None and len(df) > 0 else None


def build_one_specimen(
    specimen_id: int,
    parquet_path: Path,
    d_labitems_path: Path,
    output_dir: Path,
    *,
    grid_shape: tuple[int, int] = (16, 16),
    use_top_k: bool = True,
    missing_fill: float = np.nan,
    use_raw_values: bool = True,
    itemid_to_rc: dict | None = None,
) -> Path | None:
    """
    Build .npy for one specimen_id. If itemid_to_rc is provided, reuse it; else build from parquet + d_labitems.
    Returns output path or None if specimen has no events.
    """
    events = fetch_events_for_specimen(parquet_path, specimen_id)
    if events is None:
        return None

    n_cells = grid_shape[0] * grid_shape[1]
    if itemid_to_rc is None:
        if use_top_k and parquet_path.exists():
            itemids = get_top_itemids_from_parquet(parquet_path, n_cells)
        else:
            itemids = None
        itemid_to_rc, _, _ = build_itemid_to_grid_index(
            d_labitems_path, grid_shape=grid_shape, itemids=itemids
        )

    img = specimen_events_to_grid(
        events,
        itemid_to_rc,
        grid_shape,
        missing_fill=missing_fill,
        use_raw_values=use_raw_values,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"specimen_{specimen_id}.npy"
    np.save(out_path, img)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build .npy grid(s) for specific specimen ID(s) from Parquet."
    )
    parser.add_argument(
        "--specimen",
        type=int,
        metavar="ID",
        help="Single specimen ID (e.g. 83925500)",
    )
    parser.add_argument(
        "--specimens",
        type=int,
        nargs="+",
        metavar="ID",
        help="Multiple specimen IDs",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Lab events Parquet (default: lab_events_with_adm.parquet if exists else labevents.parquet)",
    )
    parser.add_argument(
        "--mimic4-dir",
        type=Path,
        default=DEFAULT_MIMIC4,
        help="MIMIC-IV dir for d_labitems.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_LAB_IMAGES,
        help="Output directory for specimen_<id>.npy",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("H", "W"),
        help="Grid shape (default: 16 16)",
    )
    parser.add_argument(
        "--no-top-k",
        action="store_true",
        help="Use first H*W itemids from d_labitems (match pipeline --no-top-k)",
    )
    parser.add_argument(
        "--no-sort-by-category",
        action="store_true",
        help="Order grid by frequency (or itemid) instead of by d_labitems category",
    )
    args = parser.parse_args()

    ids = []
    if args.specimen is not None:
        ids.append(args.specimen)
    if args.specimens:
        ids.extend(args.specimens)
    if not ids:
        parser.error("Provide at least one of --specimen ID or --specimens ID1 ID2 ...")

    parquet_path = _choose_parquet(args.parquet)
    d_labitems_path = args.mimic4_dir / "d_labitems.csv"
    grid_shape = tuple(args.grid)

    if not parquet_path.exists():
        print(f"Parquet not found: {parquet_path}")
        return
    if not d_labitems_path.exists():
        print(f"d_labitems not found: {d_labitems_path}")
        return

    # Build grid index once and reuse for all specimens
    n_cells = grid_shape[0] * grid_shape[1]
    if not args.no_top_k:
        itemids = get_top_itemids_from_parquet(parquet_path, n_cells)
    else:
        itemids = None
    itemid_to_rc, _, _ = build_itemid_to_grid_index(
        d_labitems_path, grid_shape=grid_shape, itemids=itemids, sort_by_category=not args.no_sort_by_category
    )

    written = []
    for specimen_id in ids:
        out = build_one_specimen(
            specimen_id,
            parquet_path,
            d_labitems_path,
            args.output_dir,
            grid_shape=grid_shape,
            use_top_k=False,  # we already built itemid_to_rc
            missing_fill=np.nan,
            use_raw_values=True,
            itemid_to_rc=itemid_to_rc,
        )
        if out is not None:
            written.append(out)
            print(f"Wrote {out}")
        else:
            print(f"No events for specimen_id={specimen_id}, skipped.")

    print(f"Built {len(written)} / {len(ids)} specimen(s). Inspect: python scripts/inspect_specimen.py --specimen <id> --raw")


if __name__ == "__main__":
    main()
