"""
End-to-end pipeline: convert labevents CSV → Parquet (optional), then specimen → grid images.
Run from project root: python scripts/run_pipeline.py --convert
Then: python scripts/run_pipeline.py --limit-specimens 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing scripts in same dir when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_ADMISSIONS_PARQUET,
    DEFAULT_LAB_IMAGES,
    DEFAULT_LAB_WITH_ADM_PARQUET,
    DEFAULT_MIMIC4,
    DEFAULT_PARQUET,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-IV labevents: CSV→Parquet and/or specimen→grid images"
    )
    parser.add_argument(
        "--mimic4-dir",
        type=Path,
        default=DEFAULT_MIMIC4,
        help="Directory containing labevents.csv and d_labitems.csv",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert labevents.csv to Parquet first (run once; uses DuckDB streaming)",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to labevents.parquet (input for images; created by --convert)",
    )
    parser.add_argument(
        "--with-admissions",
        action="store_true",
        help="Use admission-aware labs: build admissions.parquet if needed, run lab_with_admissions, then use lab_events_with_adm.parquet for images",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_LAB_IMAGES,
        help="Output directory for specimen_<id>.npy grid images",
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
        "--limit-specimens",
        type=int,
        default=None,
        help="Process only this many specimens (for testing)",
    )
    parser.add_argument(
        "--no-top-k",
        action="store_true",
        help="Use first H*W itemids from d_labitems instead of top-K by frequency",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only run conversion (--convert); do not generate grid images",
    )
    parser.add_argument(
        "--limit-lab-rows",
        type=int,
        default=None,
        metavar="N",
        help="When using --with-admissions, only load/sort first N lab rows (fast testing; smaller output)",
    )
    parser.add_argument(
        "--force-admissions",
        action="store_true",
        help="Re-run lab_with_admissions even if lab_events_with_adm.parquet already exists and is up to date",
    )
    parser.add_argument(
        "--no-sort-by-category",
        action="store_true",
        help="Order grid by frequency (or itemid) instead of by d_labitems category",
    )
    args = parser.parse_args()

    csv_path = args.mimic4_dir / "labevents.csv"
    d_labitems_path = args.mimic4_dir / "d_labitems.csv"
    grid_shape = tuple(args.grid)

    if args.convert:
        from csv_to_parquet import labevents_csv_to_parquet

        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        labevents_csv_to_parquet(csv_path, args.parquet)
        print(f"Parquet written: {args.parquet}")

    if args.skip_images:
        return

    # Optionally use admission-aware lab Parquet
    parquet_for_images = args.parquet
    if args.with_admissions:
        from config import DEFAULT_ADMISSIONS_CSV

        admissions_parquet = DEFAULT_ADMISSIONS_PARQUET
        if not admissions_parquet.exists() and DEFAULT_ADMISSIONS_CSV.exists():
            from csv_to_parquet import admissions_csv_to_parquet

            admissions_csv_to_parquet(DEFAULT_ADMISSIONS_CSV, admissions_parquet)
            print(f"Admissions Parquet written: {admissions_parquet}")
        if not admissions_parquet.exists():
            raise SystemExit(
                f"Admissions Parquet not found: {admissions_parquet}. "
                "Place admissions.parquet in MIMIC4/ or run after converting admissions.csv."
            )
        out_adm = DEFAULT_LAB_WITH_ADM_PARQUET
        # Skip re-run if output exists and is newer than both inputs (unless --force-admissions)
        skip_build = (
            not args.force_admissions
            and out_adm.exists()
            and args.parquet.exists()
            and admissions_parquet.exists()
            and out_adm.stat().st_mtime >= max(args.parquet.stat().st_mtime, admissions_parquet.stat().st_mtime)
        )
        if skip_build:
            print(f"Using existing {out_adm} (up to date). Use --force-admissions to rebuild.")
        else:
            from lab_with_admissions import run_lab_with_admissions

            run_lab_with_admissions(
                args.parquet,
                admissions_parquet,
                out_adm,
                limit_rows=args.limit_lab_rows,
            )
        parquet_for_images = out_adm

    # Specimen → images (requires Parquet)
    if not parquet_for_images.exists():
        raise SystemExit(
            f"Parquet not found: {parquet_for_images}. Run with --convert first, or point --parquet to existing file."
        )
    if not d_labitems_path.exists():
        raise SystemExit(f"d_labitems not found: {d_labitems_path}")

    from specimen_to_image import run_specimen_to_image

    written = run_specimen_to_image(
        parquet_for_images,
        d_labitems_path,
        args.images_dir,
        grid_shape=grid_shape,
        use_top_k_itemids=not args.no_top_k,
        limit_specimens=args.limit_specimens,
        sort_by_category=not args.no_sort_by_category,
    )
    print(f"Wrote {len(written)} grid images to {args.images_dir}")


if __name__ == "__main__":
    main()
