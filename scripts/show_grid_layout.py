"""
Print how the lab grid is built: which 256 labs are chosen, in what order, and where each (row,col) is.
Run this to verify category-based ordering and to see the full map used for every specimen image.

Usage:
  python scripts/show_grid_layout.py
  python scripts/show_grid_layout.py --no-sort-by-category
  python scripts/show_grid_layout.py --no-top-k --grid 8 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_LAB_WITH_ADM_PARQUET, DEFAULT_MIMIC4, DEFAULT_PARQUET
from lab_grid import build_itemid_to_grid_index, get_top_itemids_from_parquet


def _choose_parquet(parquet_path: Path | None) -> Path:
    if parquet_path is not None and parquet_path.exists():
        return Path(parquet_path)
    if DEFAULT_LAB_WITH_ADM_PARQUET.exists():
        return DEFAULT_LAB_WITH_ADM_PARQUET
    return DEFAULT_PARQUET


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show which labs fill the grid and in what order (same logic as pipeline)."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Lab events Parquet for top-K (default: lab_events_with_adm.parquet if exists else labevents.parquet)",
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
        help="Use first H*W itemids from d_labitems instead of top-K by frequency",
    )
    parser.add_argument(
        "--no-sort-by-category",
        action="store_true",
        help="Show frequency/itemid order instead of category order",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="One line per cell (row col category itemid label); default is grouped by category",
    )
    args = parser.parse_args()

    parquet_path = _choose_parquet(args.parquet)
    d_labitems_path = args.mimic4_dir / "d_labitems.csv"
    grid_shape = tuple(args.grid)
    H, W = grid_shape
    n_cells = H * W

    if not d_labitems_path.exists():
        print(f"d_labitems not found: {d_labitems_path}")
        return

    # Same logic as pipeline
    if not args.no_top_k and parquet_path.exists():
        itemids = get_top_itemids_from_parquet(parquet_path, n_cells)
        selection = f"Top {n_cells} itemids by event count from {parquet_path.name}"
    else:
        itemids = None
        selection = f"First {n_cells} itemids from d_labitems (by itemid)"

    order = "category then itemid (same category adjacent)" if not args.no_sort_by_category else "frequency order (or itemid)"
    print("How the grid is built")
    print("=" * 60)
    print("1) Which labs (the 256):", selection)
    print("2) Order:", order)
    print("3) Layout: row-major. Index 0 -> (0,0), 1 -> (0,1), ...", H, "x", W)
    print("=" * 60)

    itemid_to_rc, index_to_itemid, chosen = build_itemid_to_grid_index(
        d_labitems_path,
        grid_shape=grid_shape,
        itemids=itemids,
        sort_by_category=not args.no_sort_by_category,
    )

    if "category" not in chosen.columns:
        chosen["category"] = "(no category)"
    chosen = chosen.assign(
        row=[idx // W for idx in range(len(chosen))],
        col=[idx % W for idx in range(len(chosen))],
    )

    if args.compact:
        print("\n(row, col)  category           itemid   label")
        print("-" * 60)
        for _, r in chosen.iterrows():
            print(f"  ({r['row']:2}, {r['col']:2})   {str(r['category']):18} {int(r['itemid']):6}   {str(r['label'])[:40]}")
    else:
        print("\nGrid by category (row, col) -> itemid label")
        print("-" * 60)
        for cat, grp in chosen.groupby("category", sort=False):
            print(f"\n  [{cat}]")
            for _, r in grp.iterrows():
                print(f"    ({r['row']:2}, {r['col']:2})  itemid={int(r['itemid']):6}  {str(r['label'])[:50]}")

    print("\nDone. Every specimen .npy uses this same (row,col) -> itemid map.")


if __name__ == "__main__":
    main()
