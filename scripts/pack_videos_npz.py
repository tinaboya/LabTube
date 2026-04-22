"""Pack normalized per-admission .npy videos into a single .npz dataset.

This is the final step before training:
    admission_to_video.py  →  normalize_videos.py  →  pack_videos_npz.py  →  labtube_train.py
    (raw .npy)                 (normalized .npy)       (.npz)

The output NPZ contains:
    vid_<hadm_id>          : float32 (T, 8, 8)  — z-scored, clipped video
    meta_labels_csv        : bytes   — CSV with outcomes and per-admission stats
    meta_split_json        : bytes   — {"train": [...], "val": [...], "test": [...], "meta": [...]}
    meta_norm_stats_json   : bytes   — per-lab normalization stats (from normalize_videos.py)
    meta_grid_layout_json  : bytes   — grid cell → itemid/label mapping

Splits are inherited from a reference NPZ (default: data_embedding/labtube_dataset.npz)
so the same patients are in the same folds. Any patient absent from the new video dir
is dropped from that fold automatically.

Usage
-----
    # Build 2h-bin NPZ reusing original splits and labels
    python scripts/pack_videos_npz.py \\
        --video-dir lab_videos_2h_normalized \\
        --out     data_embedding/labtube_dataset_2h.npz

    # Use a different reference NPZ for splits
    python scripts/pack_videos_npz.py \\
        --video-dir lab_videos_2h_normalized \\
        --ref-npz  data_embedding/labtube_dataset.npz \\
        --out      data_embedding/labtube_dataset_2h.npz

    # Compute a fresh 70/10/20 split (no reference NPZ needed)
    python scripts/pack_videos_npz.py \\
        --video-dir lab_videos_2h_normalized \\
        --out       data_embedding/labtube_dataset_2h.npz \\
        --no-ref-split
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_fill_stats(video: np.ndarray) -> tuple[float, int]:
    """Return (fill_pct, frames_with_data) for a (T, H, W) video."""
    T = video.shape[0]
    frames_with_data = int(np.any(~np.isnan(video), axis=(1, 2)).sum())
    fill_pct = round(frames_with_data / T * 100, 2)
    return fill_pct, frames_with_data


def _load_ref_npz(ref_npz: Path) -> tuple[dict, pd.DataFrame]:
    """Load split and labels from a reference NPZ."""
    data      = np.load(str(ref_npz), allow_pickle=False)
    split     = json.loads(data["meta_split_json"].tobytes())
    labels_df = pd.read_csv(io.BytesIO(data["meta_labels_csv"].tobytes()))
    return split, labels_df


def _fresh_split(hadm_ids: list[int], seed: int,
                 train_frac: float, val_frac: float) -> dict[str, list[int]]:
    """Random 70/10/20 (or custom) split."""
    rng = random.Random(seed)
    ids = hadm_ids[:]
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return {
        "train": ids[:n_train],
        "val":   ids[n_train : n_train + n_val],
        "test":  ids[n_train + n_val :],
        "meta":  [],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack normalized admission .npy videos into a training .npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video-dir", type=Path, default=Path("lab_videos_2h_normalized"),
                        help="Directory of normalized admission_<hadm_id>.npy files")
    parser.add_argument("--out", type=Path, default=Path("data_embedding/labtube_dataset_2h.npz"),
                        help="Output .npz path")
    parser.add_argument("--ref-npz", type=Path, default=Path("data_embedding/labtube_dataset.npz"),
                        help="Reference NPZ to inherit splits and labels from")
    parser.add_argument("--no-ref-split", action="store_true",
                        help="Ignore reference NPZ: compute a fresh random split")
    parser.add_argument("--train-frac", type=float, default=0.70,
                        help="Training fraction for fresh split (default: 0.70)")
    parser.add_argument("--val-frac",   type=float, default=0.10,
                        help="Validation fraction for fresh split (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video_dir = args.video_dir
    out_path  = args.out

    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")

    # ── Discover normalized videos ────────────────────────────────────────────
    npy_files = sorted(video_dir.glob("admission_*.npy"))
    if not npy_files:
        raise SystemExit(f"No admission_*.npy files found in {video_dir}")

    hadm_id_to_path: dict[int, Path] = {}
    for p in npy_files:
        try:
            hid = int(p.stem.split("_")[1])
            hadm_id_to_path[hid] = p
        except (IndexError, ValueError):
            print(f"  Warning: skipping unexpected filename {p.name}")

    all_hadm_ids = sorted(hadm_id_to_path)
    print(f"Found {len(all_hadm_ids):,} admission videos in {video_dir}/")

    # Peek at first video to get shape
    sample_video = np.load(hadm_id_to_path[all_hadm_ids[0]])
    T, H, W = sample_video.shape
    print(f"Video shape  : ({T}, {H}, {W})  —  {T} frames × {H}×{W} grid")

    # ── Load grid layout ──────────────────────────────────────────────────────
    grid_layout_path = video_dir / "grid_layout.json"
    if not grid_layout_path.exists():
        raise SystemExit(f"grid_layout.json not found in {video_dir}. "
                         "normalize_videos.py should have copied it there.")
    with open(grid_layout_path) as f:
        grid_layout = json.load(f)
    print(f"Grid layout  : {len(grid_layout)} cells")

    # ── Load norm stats ───────────────────────────────────────────────────────
    norm_stats_path = video_dir / "norm_stats.json"
    if not norm_stats_path.exists():
        raise SystemExit(f"norm_stats.json not found in {video_dir}. "
                         "Run normalize_videos.py first.")
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    print(f"Norm stats   : {norm_stats['n_labs']} labs  "
          f"({norm_stats['n_log_transformed']} log-transformed)")

    # ── Build / inherit splits ────────────────────────────────────────────────
    available = set(all_hadm_ids)

    if args.no_ref_split or not args.ref_npz.exists():
        if not args.no_ref_split:
            print(f"Warning: reference NPZ not found at {args.ref_npz}. Computing fresh split.")
        split = _fresh_split(all_hadm_ids, args.seed, args.train_frac, args.val_frac)
        ref_labels_df = None
        print(f"Fresh split  : train={len(split['train']):,}  "
              f"val={len(split['val']):,}  test={len(split['test']):,}")
    else:
        print(f"\nInheriting splits from {args.ref_npz} …")
        ref_split, ref_labels_df = _load_ref_npz(args.ref_npz)
        split = {}
        for fold, ids in ref_split.items():
            # "meta" fold stores a dict of split provenance info — preserve as-is
            if isinstance(ids, dict):
                split[fold] = ids
                print(f"  {fold:<6}: (metadata dict — preserved)")
                continue
            filtered = [h for h in ids if h in available]
            split[fold] = filtered
            orig    = len(ids)
            kept    = len(filtered)
            dropped = orig - kept
            msg = f"  {fold:<6}: {kept:>6,} / {orig:>6,}"
            if dropped:
                msg += f"  ({dropped} dropped — not in video dir)"
            print(msg)

    # ── Build labels DataFrame ────────────────────────────────────────────────
    print("\nBuilding labels …")

    # Determine hadm_ids to include (union of patient folds; skip metadata dicts)
    included_ids: list[int] = []
    for ids in split.values():
        if isinstance(ids, list):
            included_ids.extend(ids)
    included_ids = sorted(set(included_ids))

    if ref_labels_df is not None:
        # Keep only rows present in new video set; recompute fill_pct / frames_with_data
        base_df = ref_labels_df[ref_labels_df["hadm_id"].isin(included_ids)].copy()
        base_df = base_df.set_index("hadm_id")
    else:
        base_df = None

    rows = []
    for hid in included_ids:
        video = np.load(hadm_id_to_path[hid])
        fill_pct, frames_with_data = _compute_fill_stats(video)

        if base_df is not None and hid in base_df.index:
            row = base_df.loc[hid].to_dict()
        else:
            row = {}

        row["hadm_id"]          = hid
        row["fill_pct"]         = fill_pct
        row["frames_with_data"] = frames_with_data
        rows.append(row)

    labels_df = pd.DataFrame(rows)
    # Ensure hadm_id is first column
    cols = ["hadm_id"] + [c for c in labels_df.columns if c != "hadm_id"]
    labels_df = labels_df[cols].reset_index(drop=True)
    print(f"  {len(labels_df):,} admissions  |  columns: {labels_df.columns.tolist()}")

    # ── Pack NPZ ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nPacking NPZ → {out_path} …")

    arrays: dict[str, np.ndarray] = {}

    # Meta arrays (stored as byte arrays)
    arrays["meta_labels_csv"]      = np.frombuffer(
        labels_df.to_csv(index=False).encode(), dtype=np.uint8)
    arrays["meta_split_json"]      = np.frombuffer(
        json.dumps(split).encode(), dtype=np.uint8)
    arrays["meta_norm_stats_json"] = np.frombuffer(
        json.dumps(norm_stats).encode(), dtype=np.uint8)
    arrays["meta_grid_layout_json"] = np.frombuffer(
        json.dumps(grid_layout).encode(), dtype=np.uint8)

    # Video arrays
    n_written = 0
    for hid in included_ids:
        video = np.load(hadm_id_to_path[hid]).astype(np.float32)
        arrays[f"vid_{hid}"] = video
        n_written += 1
        if n_written % 5000 == 0:
            print(f"  {n_written:,} / {len(included_ids):,} videos packed …")

    np.savez_compressed(str(out_path), **arrays)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nDone.")
    print(f"  Videos written : {n_written:,}")
    print(f"  Output size    : {size_mb:.1f} MB")
    print(f"  Output path    : {out_path}")
    print(f"\nSplit summary:")
    for fold, ids in split.items():
        if isinstance(ids, dict):
            print(f"  {fold:<6}: (metadata dict)")
        else:
            print(f"  {fold:<6}: {len(ids):,}")


if __name__ == "__main__":
    main()
