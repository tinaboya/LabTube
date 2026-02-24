"""
Reconstruct and inspect a specimen .npy: which lab is at each (row,col), and optional raw events from Parquet.

Usage:
  python scripts/inspect_specimen.py --find-rich 10
  python scripts/inspect_specimen.py --find-rich 10 --only-with-npy
  python scripts/inspect_specimen.py --specimen 367826
  python scripts/inspect_specimen.py --specimen 367826 --raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_LAB_IMAGES, DEFAULT_LAB_WITH_ADM_PARQUET, DEFAULT_MIMIC4, DEFAULT_PARQUET
from lab_grid import build_itemid_to_grid_index, get_top_itemids_from_parquet


def _choose_parquet(parquet_path: Path | None) -> Path:
    if parquet_path is not None and parquet_path.exists():
        return Path(parquet_path)
    if DEFAULT_LAB_WITH_ADM_PARQUET.exists():
        return DEFAULT_LAB_WITH_ADM_PARQUET
    return DEFAULT_PARQUET


def reconstruct_grid(
    npy_path: Path,
    d_labitems_path: Path,
    parquet_path: Path,
    grid_shape: tuple[int, int] = (16, 16),
    use_top_k: bool = True,
    sort_by_category: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int, str, float]]]:
    """
    Load .npy and build (row, col) -> itemid, label. Return array and list of (r, c, itemid, label, value).
    """
    arr = np.load(npy_path)
    H, W = grid_shape
    n_cells = H * W

    if use_top_k and parquet_path.exists():
        itemids = get_top_itemids_from_parquet(parquet_path, n_cells)
    else:
        itemids = None
    itemid_to_rc, index_to_itemid, df_items = build_itemid_to_grid_index(
        d_labitems_path, grid_shape=grid_shape, itemids=itemids, sort_by_category=sort_by_category
    )
    # index -> itemid; (r,c) -> index = r*W + c
    itemid_to_label = dict(zip(df_items["itemid"].astype(int), df_items["label"].astype(str)))

    rows = []
    for r in range(H):
        for c in range(W):
            idx = r * W + c
            iid = index_to_itemid[idx] if idx < len(index_to_itemid) else 0
            val = float(arr[r, c])
            label = itemid_to_label.get(int(iid), f"itemid_{iid}") if iid else "(empty)"
            rows.append((r, c, int(iid) if iid else 0, label, val))
    return arr, rows


def find_specimens_with_most_labs(
    parquet_path: Path,
    top_n: int = 20,
    only_with_npy: bool = False,
    images_dir: Path | None = None,
) -> list[tuple[int, int, int]]:
    """
    Query parquet for specimen_ids with the most lab events (and most distinct itemids).
    Returns list of (specimen_id, n_events, n_distinct_itemids), sorted by n_events desc.
    """
    import duckdb

    images_dir = images_dir or DEFAULT_LAB_IMAGES
    pq = Path(parquet_path).as_posix()
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            SELECT specimen_id,
                   COUNT(*) AS n_events,
                   COUNT(DISTINCT itemid) AS n_labs
            FROM read_parquet('{pq}')
            WHERE specimen_id IS NOT NULL
            GROUP BY specimen_id
            ORDER BY n_events DESC
            LIMIT {top_n * 2}
            """
        )
        rows = con.fetchall()
    finally:
        con.close()
    result = [(int(r[0]), int(r[1]), int(r[2])) for r in rows]
    if only_with_npy and images_dir.exists():
        result = [(sid, ne, nl) for sid, ne, nl in result if (images_dir / f"specimen_{sid}.npy").exists()]
    return result[:top_n]


def fetch_raw_events(parquet_path: Path, specimen_id: int) -> list[dict] | None:
    """Return list of event dicts for this specimen from Parquet, or None if not found."""
    import duckdb

    pq = Path(parquet_path).as_posix()
    con = duckdb.connect()
    try:
        df = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            WHERE specimen_id = ?
            ORDER BY charttime, itemid
            """,
            [pq, specimen_id],
        ).fetchdf()
    finally:
        con.close()
    if df is None or len(df) == 0:
        return None
    return df.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct specimen .npy: map (row,col) to lab name and value; optionally show raw Parquet events."
    )
    parser.add_argument(
        "--find-rich",
        type=int,
        metavar="N",
        default=None,
        help="Find N specimen IDs with the most lab events (from Parquet); do not inspect a single specimen",
    )
    parser.add_argument(
        "--only-with-npy",
        action="store_true",
        help="With --find-rich: only list specimens that already have a .npy in lab_images/",
    )
    parser.add_argument("--specimen", type=int, default=None, help="Specimen ID to inspect (e.g. 367826)")
    parser.add_argument(
        "--npy",
        type=Path,
        default=None,
        help="Path to specimen_<id>.npy (default: lab_images/specimen_<id>.npy)",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Lab events Parquet used to build the grid (default: lab_events_with_adm.parquet if exists else labevents.parquet)",
    )
    parser.add_argument(
        "--mimic4-dir",
        type=Path,
        default=DEFAULT_MIMIC4,
        help="MIMIC-IV dir for d_labitems.csv",
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
        help="Use first H*W itemids from d_labitems (match --no-top-k in pipeline)",
    )
    parser.add_argument(
        "--no-sort-by-category",
        action="store_true",
        help="Use frequency/itemid order for grid (match when .npy was built with --no-sort-by-category)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also query Parquet and print raw events for this specimen (subject_id, charttime, itemid, valuenum, etc.)",
    )
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="Print all grid cells; by default only non-missing (value > 0) cells",
    )
    args = parser.parse_args()

    parquet_path = _choose_parquet(args.parquet)
    if args.find_rich is not None:
        top_n = max(1, args.find_rich)
        rows = find_specimens_with_most_labs(
            parquet_path,
            top_n=top_n,
            only_with_npy=args.only_with_npy,
        )
        print(f"Parquet: {parquet_path}")
        print(f"Top {len(rows)} specimen(s) by lab event count:")
        print("-" * 50)
        for specimen_id, n_events, n_labs in rows:
            npy_exists = (DEFAULT_LAB_IMAGES / f"specimen_{specimen_id}.npy").exists()
            mark = "  [has .npy]" if npy_exists else ""
            print(f"  specimen_id={specimen_id}  n_events={n_events}  n_labs={n_labs}{mark}")
        print("-" * 50)
        print("Inspect one: python scripts/inspect_specimen.py --specimen <id> --raw")
        if not rows and args.only_with_npy:
            print("No specimens with .npy found. Run pipeline without --only-with-npy to see top from Parquet.")
        return

    if args.specimen is None:
        parser.error("Either --specimen <id> or --find-rich N is required")
    specimen_id = args.specimen
    npy_path = args.npy or (DEFAULT_LAB_IMAGES / f"specimen_{specimen_id}.npy")
    d_labitems_path = args.mimic4_dir / "d_labitems.csv"
    parquet_path = _choose_parquet(args.parquet)
    grid_shape = tuple(args.grid)

    if not npy_path.exists():
        print(f"Not found: {npy_path}")
        return
    if not d_labitems_path.exists():
        print(f"Not found: {d_labitems_path}")
        return

    arr, rows = reconstruct_grid(
        npy_path,
        d_labitems_path,
        parquet_path,
        grid_shape=grid_shape,
        use_top_k=not args.no_top_k,
        sort_by_category=not args.no_sort_by_category,
    )

    print(f"Specimen ID: {specimen_id}")
    print(f"NPY path:   {npy_path}")
    print(f"Grid shape: {arr.shape}  (expected {grid_shape})")
    print(f"Parquet:    {parquet_path}")
    print()
    print("Reconstructed grid (row, col) → itemid, label, value (missing = nan)")
    print("-" * 80)
    def _is_missing(v: float) -> bool:
        return np.isnan(v) if isinstance(v, (float, np.floating)) else False
    if args.all_cells:
        to_show = rows
    else:
        to_show = [x for x in rows if not _is_missing(x[4])]
    for r, c, iid, label, val in to_show:
        val_str = f"{val:.4f}" if not _is_missing(val) else "nan"
        print(f"  ({r:2}, {c:2})  itemid={iid:6}  {label[:50]:50}  {val_str}")
    print("-" * 80)
    print(f"Non-missing cells: {sum(1 for x in rows if not _is_missing(x[4]))} / {len(rows)}")
    print()

    if args.raw:
        events = fetch_raw_events(parquet_path, specimen_id)
        if not events:
            print("No raw events found in Parquet for this specimen_id.")
            return
        # First row for subject_id, charttime, hadm_id, admittime
        first = events[0]
        keys = list(first.keys())
        meta_keys = [k for k in ["subject_id", "hadm_id", "charttime", "admittime"] if k in keys]
        print("Raw events from Parquet")
        print("Metadata (first event):")
        for k in meta_keys:
            print(f"  {k}: {first.get(k)}")
        print()
        print("Events (itemid, valuenum, value, ref_range_lower, ref_range_upper):")
        for ev in events:
            iid = ev.get("itemid")
            vn = ev.get("valuenum")
            v = ev.get("value")
            rlo = ev.get("ref_range_lower")
            rhi = ev.get("ref_range_upper")
            print(f"  itemid={iid}  valuenum={vn}  value={v!r}  ref=[{rlo}, {rhi}]")
        print(f"Total events: {len(events)}")


if __name__ == "__main__":
    main()
