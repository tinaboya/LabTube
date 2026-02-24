"""
Inspect lab_images/*.npy: print shape/stats and optionally show a heatmap.
Usage:
  uv run python scripts/inspect_lab_images.py
  uv run python scripts/inspect_lab_images.py --specimen 32
  uv run python scripts/inspect_lab_images.py --specimen 32 --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = ROOT / "lab_images"


def inspect_one(path: Path) -> None:
    arr = np.load(path)
    print(f"  {path.name}")
    print(f"    shape: {arr.shape}  dtype: {arr.dtype}")
    valid = arr[~(np.isnan(arr))] if arr.size else np.array([])
    non_missing = np.sum(~np.isnan(arr))
    print(f"    min/max (all): {np.nanmin(arr):.4f} / {np.nanmax(arr):.4f}")
    print(f"    non-missing cells: {non_missing} / {arr.size}")
    if valid.size:
        print(f"    min/max (non-missing): {float(np.min(valid)):.4f} / {float(np.max(valid)):.4f}")
    print()


def plot_heatmap(arr: np.ndarray, title: str = "Lab grid") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to use --plot: uv add matplotlib")
        return
    fig, ax = plt.subplots()
    # For raw values, use nanmin/nanmax; nan (missing) → white
    vmin = np.nanmin(arr) if arr.size else 0
    vmax = np.nanmax(arr) if arr.size else 1
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="value (white=missing)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect lab_images/*.npy")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--specimen", type=int, default=None, help="Inspect only this specimen id")
    parser.add_argument("--plot", action="store_true", help="Show heatmap (requires matplotlib)")
    parser.add_argument("--list", action="store_true", help="Only list .npy files")
    args = parser.parse_args()

    images_dir = args.images_dir
    if not images_dir.exists():
        print(f"Directory not found: {images_dir}")
        return

    npy_files = sorted(images_dir.glob("specimen_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if not npy_files:
        print(f"No specimen_*.npy files in {images_dir}")
        return

    if args.list:
        for p in npy_files:
            print(p.name)
        return

    if args.specimen is not None:
        path = images_dir / f"specimen_{args.specimen}.npy"
        if not path.exists():
            print(f"Not found: {path}")
            return
        inspect_one(path)
        if args.plot:
            plot_heatmap(np.load(path), title=f"Specimen {args.specimen}")
        return

    print(f"Inspecting {len(npy_files)} files in {images_dir}\n")
    for path in npy_files:
        inspect_one(path)
    if args.plot and npy_files:
        plot_heatmap(np.load(npy_files[0]), title=f"First: {npy_files[0].name}")


if __name__ == "__main__":
    main()
