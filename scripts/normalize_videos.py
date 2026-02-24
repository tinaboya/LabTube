"""
Normalize raw admission videos: log1p (skewed labs) → per-lab z-score → clip.

Implements the hybrid normalization strategy from Section 8 of
VIDEO_CONSTRUCTION_REPORT.md:

  1.  Selective log1p transform for 12 right-skewed labs
  2.  Per-lab z-score using training-set statistics
  3.  Clip to [-10, +10]

Pipeline position:
    admission_to_video.py  →  normalize_videos.py  →  VideoMAE DataLoader
    (raw .npy)                (normalized .npy)

Usage:
    # Compute stats from training set, then normalize all splits
    python scripts/normalize_videos.py

    # Specify directories explicitly
    python scripts/normalize_videos.py \\
        --video-dir lab_videos \\
        --out-dir lab_videos_normalized \\
        --grid-layout lab_videos/grid_layout.json

    # Compute stats only (no normalization)
    python scripts/normalize_videos.py --stats-only

    # Normalize using pre-computed stats
    python scripts/normalize_videos.py --stats-file lab_videos_normalized/norm_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Labs that receive log1p transform (right-skewed distributions) ──
# These are the 12+ labs identified in Section 8.3 of VIDEO_CONSTRUCTION_REPORT.md.
# Key: itemid → short label (for reference only; the grid position is what matters).
LOG_TRANSFORM_ITEMIDS: set[int] = {
    50861,  # ALT  (Alanine Aminotransferase)
    50863,  # ALP  (Alkaline Phosphatase)
    50878,  # AST  (Aspartate Aminotransferase)
    50885,  # Bilirubin, Total
    50910,  # Creatine Kinase (CK)
    50911,  # CK-MB Isoenzyme
    50954,  # LDH  (Lactate Dehydrogenase)
    50956,  # Lipase
    51003,  # Troponin T
    51265,  # Platelet Count
    51301,  # WBC  (White Blood Cells)
    52069,  # Absolute Basophil Count
    52073,  # Absolute Eosinophil Count
    52074,  # Absolute Monocyte Count
    52075,  # Absolute Neutrophil Count
}

CLIP_RANGE = (-10.0, 10.0)


def load_grid_layout(path: Path) -> dict[tuple[int, int], int]:
    """Load grid_layout.json → {(row, col): itemid}."""
    with open(path) as f:
        grid_info = json.load(f)
    return {(g["row"], g["col"]): g["itemid"] for g in grid_info}


def build_log_mask(
    rc_to_itemid: dict[tuple[int, int], int],
    grid_shape: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Build a boolean mask of shape (H, W) where True = this cell gets log1p.

    The mask is determined by LOG_TRANSFORM_ITEMIDS — labs whose distribution
    is heavily right-skewed and benefit from variance-stabilizing log transform.
    """
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    for (r, c), itemid in rc_to_itemid.items():
        if itemid in LOG_TRANSFORM_ITEMIDS:
            mask[r, c] = True
    return mask


def compute_norm_stats(
    npy_files: list[Path],
    log_mask: np.ndarray,
    grid_shape: tuple[int, int] = (8, 8),
) -> dict:
    """
    Compute per-lab (per grid cell) mean and std from a set of raw .npy videos.

    For cells in log_mask, statistics are computed on log1p(x).
    For other cells, statistics are computed on raw x.

    Returns dict with:
        "mean": (H, W) array
        "std":  (H, W) array
        "count": (H, W) array  (number of non-NaN values per cell)
        "log_mask": (H, W) bool array
    """
    H, W = grid_shape
    # Online Welford's algorithm accumulators
    n = np.zeros((H, W), dtype=np.float64)
    mean = np.zeros((H, W), dtype=np.float64)
    m2 = np.zeros((H, W), dtype=np.float64)

    for path in npy_files:
        video = np.load(path)  # (T, H, W)
        T = video.shape[0]

        for t in range(T):
            frame = video[t].astype(np.float64)

            # Apply log1p to designated cells
            log_frame = frame.copy()
            log_frame[log_mask] = np.where(
                np.isnan(frame[log_mask]),
                np.nan,
                np.log1p(np.where(np.isnan(frame[log_mask]), 0, frame[log_mask])),
            )

            # Welford's update for each cell with valid data
            valid = ~np.isnan(log_frame)
            n[valid] += 1
            delta = np.where(valid, log_frame - mean, 0)
            mean = np.where(valid, mean + delta / np.maximum(n, 1), mean)
            delta2 = np.where(valid, log_frame - mean, 0)
            m2 = np.where(valid, m2 + delta * delta2, m2)

    # Compute std (use ddof=1 for sample std)
    std = np.sqrt(np.where(n > 1, m2 / (n - 1), 0))
    # Avoid division by zero: if std == 0, set to 1 (no scaling)
    std = np.where(std < 1e-8, 1.0, std)

    return {
        "mean": mean,
        "std": std,
        "count": n,
        "log_mask": log_mask,
    }


def normalize_video(
    video: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    log_mask: np.ndarray,
    clip_range: tuple[float, float] = CLIP_RANGE,
) -> np.ndarray:
    """
    Normalize a single (T, H, W) raw video tensor.

    Steps:
        1. log1p transform for designated cells
        2. z-score: (x - mean) / std  (per grid cell)
        3. clip to [clip_low, clip_high]
        NaN remains NaN at every step.
    """
    T, H, W = video.shape
    out = np.full_like(video, np.nan, dtype=np.float32)

    for t in range(T):
        frame = video[t].astype(np.float64)

        # Step 1: log1p for skewed labs
        transformed = frame.copy()
        valid_log = ~np.isnan(frame) & log_mask
        transformed[valid_log] = np.log1p(frame[valid_log])

        # Step 2: z-score
        valid = ~np.isnan(transformed)
        z = np.where(valid, (transformed - mean) / std, np.nan)

        # Step 3: clip
        z = np.where(~np.isnan(z), np.clip(z, clip_range[0], clip_range[1]), np.nan)

        out[t] = z.astype(np.float32)

    return out


def stats_to_json(
    stats: dict,
    rc_to_itemid: dict[tuple[int, int], int],
    grid_info_path: Path,
) -> dict:
    """Convert numpy stats to JSON-serializable dict with lab labels."""
    # Load labels
    with open(grid_info_path) as f:
        grid_info = json.load(f)
    rc_to_label = {(g["row"], g["col"]): g["label"] for g in grid_info}

    H, W = stats["mean"].shape
    labs = []
    for r in range(H):
        for c in range(W):
            itemid = rc_to_itemid.get((r, c), None)
            lab = {
                "row": r,
                "col": c,
                "itemid": itemid,
                "label": rc_to_label.get((r, c), "?"),
                "log_transform": bool(stats["log_mask"][r, c]),
                "mean": round(float(stats["mean"][r, c]), 6),
                "std": round(float(stats["std"][r, c]), 6),
                "count": int(stats["count"][r, c]),
            }
            labs.append(lab)

    return {
        "clip_range": list(CLIP_RANGE),
        "n_labs": len(labs),
        "n_log_transformed": int(stats["log_mask"].sum()),
        "labs": labs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw admission videos (log1p + z-score + clip).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video-dir", type=Path, default=None,
                        help="Directory containing raw .npy videos (default: lab_videos/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for normalized videos (default: lab_videos_normalized/)")
    parser.add_argument("--grid-layout", type=Path, default=None,
                        help="Path to grid_layout.json")
    parser.add_argument("--stats-file", type=Path, default=None,
                        help="Pre-computed norm_stats.json (skip stats computation)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute and save stats, don't normalize")
    parser.add_argument("--train-glob", type=str, default="admission_*.npy",
                        help="Glob pattern for training-set videos (default: all)")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    video_dir = args.video_dir or (root / "lab_videos")
    out_dir = args.out_dir or (root / "lab_videos_normalized")
    grid_layout_path = args.grid_layout or (video_dir / "grid_layout.json")

    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return
    if not grid_layout_path.exists():
        print(f"Grid layout not found: {grid_layout_path}")
        return

    # ── Load grid layout ─────────────────────────────────────────────
    rc_to_itemid = load_grid_layout(grid_layout_path)
    H = max(r for r, c in rc_to_itemid) + 1
    W = max(c for r, c in rc_to_itemid) + 1
    grid_shape = (H, W)
    log_mask = build_log_mask(rc_to_itemid, grid_shape)

    print(f"Grid: {H}×{W} = {H*W} labs")
    print(f"Log-transform labs: {int(log_mask.sum())} / {H*W}")
    print(f"Clip range: [{CLIP_RANGE[0]}, {CLIP_RANGE[1]}]")

    # ── Compute or load normalization stats ───────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "norm_stats.json"

    if args.stats_file and args.stats_file.exists():
        print(f"\nLoading pre-computed stats from {args.stats_file}")
        with open(args.stats_file) as f:
            stats_json = json.load(f)
        # Reconstruct numpy arrays
        mean = np.zeros(grid_shape, dtype=np.float64)
        std = np.ones(grid_shape, dtype=np.float64)
        for lab in stats_json["labs"]:
            r, c = lab["row"], lab["col"]
            mean[r, c] = lab["mean"]
            std[r, c] = lab["std"]
        stats = {"mean": mean, "std": std, "count": None, "log_mask": log_mask}
    else:
        # Find training .npy files
        train_files = sorted(video_dir.glob(args.train_glob))
        if not train_files:
            print(f"No training files matching '{args.train_glob}' in {video_dir}")
            return
        print(f"\nComputing normalization stats from {len(train_files)} videos...")

        stats = compute_norm_stats(train_files, log_mask, grid_shape)

        # Save stats as JSON
        stats_json = stats_to_json(stats, rc_to_itemid, grid_layout_path)
        with open(stats_path, "w") as f:
            json.dump(stats_json, f, indent=2)
        print(f"Stats saved to {stats_path}")

        # Print summary
        print(f"\nPer-lab normalization stats:")
        print(f"  {'pos':>5}  {'lab':<30}  {'log?':>4}  {'mean':>10}  {'std':>10}  {'count':>8}")
        print(f"  {'─'*5}  {'─'*30}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}")
        for lab in stats_json["labs"]:
            logstr = "yes" if lab["log_transform"] else "no"
            print(f"  ({lab['row']},{lab['col']})  {lab['label']:<30}  {logstr:>4}  "
                  f"{lab['mean']:>10.4f}  {lab['std']:>10.4f}  {lab['count']:>8,}")

    if args.stats_only:
        print("\n--stats-only: done.")
        return

    # ── Normalize all videos ──────────────────────────────────────────
    all_npy = sorted(video_dir.glob("admission_*.npy"))
    if not all_npy:
        print(f"\nNo admission_*.npy files found in {video_dir}")
        return

    print(f"\nNormalizing {len(all_npy)} videos → {out_dir}/")

    from tqdm import tqdm

    n_clipped_low = 0
    n_clipped_high = 0
    n_total_valid = 0

    for npy_path in tqdm(all_npy, desc="Normalizing", unit="video"):
        video = np.load(npy_path)
        normed = normalize_video(
            video, stats["mean"], stats["std"], stats["log_mask"], CLIP_RANGE,
        )

        # Track clipping stats
        valid = ~np.isnan(normed)
        n_total_valid += valid.sum()
        n_clipped_low += (normed[valid] == CLIP_RANGE[0]).sum()
        n_clipped_high += (normed[valid] == CLIP_RANGE[1]).sum()

        out_path = out_dir / npy_path.name
        np.save(out_path, normed)

    # Copy grid layout and metadata to output dir for convenience
    import shutil
    shutil.copy2(grid_layout_path, out_dir / "grid_layout.json")
    meta_path = video_dir / "video_metadata.json"
    if meta_path.exists():
        shutil.copy2(meta_path, out_dir / "video_metadata.json")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Normalization complete.")
    print(f"  Input:  {video_dir}/ ({len(all_npy)} videos)")
    print(f"  Output: {out_dir}/ ({len(all_npy)} normalized videos)")
    print(f"  Stats:  {stats_path}")
    print(f"\n  Total valid values:  {n_total_valid:,}")
    if n_total_valid > 0:
        print(f"  Clipped to {CLIP_RANGE[0]}:  {n_clipped_low:,}  "
              f"({100*n_clipped_low/n_total_valid:.3f}%)")
        print(f"  Clipped to {CLIP_RANGE[1]}:  {n_clipped_high:,}  "
              f"({100*n_clipped_high/n_total_valid:.3f}%)")
        print(f"  Unclipped:           {n_total_valid - n_clipped_low - n_clipped_high:,}  "
              f"({100*(n_total_valid - n_clipped_low - n_clipped_high)/n_total_valid:.3f}%)")


if __name__ == "__main__":
    main()
