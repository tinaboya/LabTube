"""
Generate visualizations for constructed admission videos.

Supports both raw (unnormalized) and normalized video directories.
When --normalized is passed, uses a diverging colormap centered at z=0
where blue = below mean, white = normal, red = above mean.

Produces:
  1. Per-video heatmap strips: T frames side by side, showing fill pattern
  2. Temporal alignment timeline: when data exists across the 24h window
  3. Aggregate density comparison chart

Usage:
    # Raw videos (viridis colormap, percentile-based scale)
    python scripts/visualize_videos.py

    # Normalized videos (diverging RdBu colormap, z-score scale)
    python scripts/visualize_videos.py --video-dir lab_videos_normalized --normalized

    # Explicit paths
    python scripts/visualize_videos.py --video-dir lab_videos --outdir docs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    parser = argparse.ArgumentParser(description="Visualize constructed videos.")
    parser.add_argument("--video-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "lab_videos")
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "docs" / "figures")
    parser.add_argument("--normalized", action="store_true",
                        help="Videos are z-score normalized; use diverging colormap")
    args = parser.parse_args()

    video_dir = args.video_dir
    outdir = args.outdir
    is_normalized = args.normalized
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = video_dir / "video_metadata.json"
    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        return
    with open(meta_path) as f:
        all_meta = json.load(f)

    # Load grid layout
    layout_path = video_dir / "grid_layout.json"
    grid_labels = {}
    if layout_path.exists():
        with open(layout_path) as f:
            grid_info = json.load(f)
        for g in grid_info:
            grid_labels[(g["row"], g["col"])] = g["label"]

    npy_files = sorted(video_dir.glob("admission_*.npy"))
    if not npy_files:
        print("No .npy video files found.")
        return

    # Load all videos
    videos = {}
    for p in npy_files:
        hid = int(p.stem.split("_")[1])
        videos[hid] = np.load(p)

    n_videos = len(videos)
    mode_str = "normalized (z-score)" if is_normalized else "raw"
    print(f"Loaded {n_videos} videos from {video_dir} [{mode_str}]")

    # ── Color scale setup ─────────────────────────────────────────────
    if is_normalized:
        # Diverging colormap: blue (low z) → white (z=0) → red (high z)
        cmap_frame = plt.cm.RdBu_r.copy()
        cmap_frame.set_bad(color="#f0f0f0")
        # Symmetric z-score range
        z_bound = 5.0  # show z in [-5, +5]; extremes beyond still colored
    else:
        cmap_frame = plt.cm.viridis.copy()
        cmap_frame.set_bad(color="#f0f0f0")

    # ================================================================
    # Figure 1: Temporal Alignment Timeline (all videos)
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * n_videos + 1.5)))

    cmap_density = LinearSegmentedColormap.from_list("fill",
        ["#f7f7f7", "#4393c3", "#2166ac", "#053061"])

    hadm_ids = [m["hadm_id"] for m in all_meta]
    T = all_meta[0]["n_bins"] if all_meta else 24

    timeline_matrix = np.zeros((n_videos, T))
    for i, meta in enumerate(all_meta):
        for fm in meta["frames"]:
            timeline_matrix[i, fm["bin"]] = fm["fill_pct"]

    im = ax.imshow(timeline_matrix, aspect="auto", cmap=cmap_density,
                   vmin=0, vmax=80, interpolation="nearest")
    ax.set_yticks(range(n_videos))
    ax.set_yticklabels([f"adm_{h}" for h in hadm_ids], fontsize=8)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"{t}" for t in range(T)], fontsize=7)
    ax.set_xlabel("Hour (time bin)", fontsize=10)
    title_suffix = " [normalized]" if is_normalized else " [raw]"
    ax.set_title(f"Temporal Alignment: Per-Frame Fill % Across 24h Window{title_suffix}",
                 fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Grid fill %", fontsize=9)

    # Annotate filled frames with specimen count
    for i, meta in enumerate(all_meta):
        for fm in meta["frames"]:
            if fm["n_cells_filled"] > 0:
                ax.text(fm["bin"], i,
                        f"{fm['n_specimens_merged']}s",
                        ha="center", va="center", fontsize=6,
                        color="white" if fm["fill_pct"] > 30 else "black")

    plt.tight_layout()
    fig.savefig(outdir / "temporal_alignment_timeline.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'temporal_alignment_timeline.png'}")

    # ================================================================
    # Figure 2: Per-video heatmap strips (one figure per video)
    # ================================================================
    for hid, video in videos.items():
        meta = next((m for m in all_meta if m["hadm_id"] == hid), None)
        T, H, W = video.shape

        # Count filled frames
        frame_fills = [np.count_nonzero(~np.isnan(video[t])) for t in range(T)]
        active_bins = [t for t in range(T) if frame_fills[t] > 0]
        n_active = len(active_bins)

        if n_active == 0:
            continue

        # Create figure: top row = full 24h bar, middle = info, bottom = heatmaps
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.3, 4], hspace=0.3)

        # ── Top: fill % bar chart ──
        ax_bar = fig.add_subplot(gs[0])
        colors_bar = ["#2166ac" if frame_fills[t] > 0 else "#f0f0f0" for t in range(T)]
        ax_bar.bar(range(T), [frame_fills[t] / 64 * 100 for t in range(T)],
                   color=colors_bar, edgecolor="white", linewidth=0.5)
        ax_bar.set_xlim(-0.5, T - 0.5)
        ax_bar.set_ylabel("Fill %", fontsize=9)
        norm_label = " [z-score normalized]" if is_normalized else " [raw values]"
        ax_bar.set_title(f"Admission {hid} — 24h Temporal Alignment{norm_label}  "
                         f"(fill={meta['video_summary']['fill_pct']:.1f}%, "
                         f"frames_with_data={n_active}/{T}, "
                         f"specimens={meta['video_summary']['total_specimens_merged']})",
                         fontsize=11, fontweight="bold")
        ax_bar.set_xticks(range(T))
        ax_bar.set_xticklabels([f"{t}h" for t in range(T)], fontsize=7)
        for t in range(T):
            if frame_fills[t] > 0:
                fm = meta["frames"][t] if meta else None
                n_sp = fm["n_specimens_merged"] if fm else "?"
                ax_bar.text(t, frame_fills[t] / 64 * 100 + 2,
                            f"{n_sp}s", ha="center", fontsize=6, color="#2166ac")
        ax_bar.set_ylim(0, 85)

        # ── Middle: specimen merge info ──
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis("off")
        if meta:
            info_lines = []
            for fm in meta["frames"]:
                if fm["n_cells_filled"] > 0:
                    times_str = ", ".join(f"{h:.1f}h" for h in fm["charttimes_hours"][:4])
                    info_lines.append(
                        f"Bin {fm['bin']} ({fm['hour_range']}): "
                        f"{fm['n_specimens_merged']} specimens merged, "
                        f"{fm['n_cells_filled']}/64 cells, "
                        f"charttimes=[{times_str}]"
                    )
            text = "\n".join(info_lines[:8])
            if len(info_lines) > 8:
                text += f"\n... (+{len(info_lines)-8} more frames)"
            ax_info.text(0.02, 0.95, text, transform=ax_info.transAxes,
                         fontsize=7, va="top", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                                   edgecolor="#cccccc"))

        # ── Bottom: frame heatmaps (only active frames) ──
        ax_frames = fig.add_subplot(gs[2])
        ax_frames.axis("off")

        # Show up to 12 active frames
        show_bins = active_bins[:12]
        n_show = len(show_bins)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, n_show, subplot_spec=gs[2], wspace=0.15)

        # Build color scale
        all_vals = video[~np.isnan(video)]
        if is_normalized:
            # Diverging scale centered at 0 for z-scores
            if len(all_vals) > 0:
                abs_max = min(np.percentile(np.abs(all_vals), 99), z_bound)
                abs_max = max(abs_max, 1.0)  # at least [-1, +1]
            else:
                abs_max = z_bound
            norm_scale = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        else:
            # Raw: use 1st/99th percentile
            if len(all_vals) > 0:
                vmin_val, vmax_val = np.percentile(all_vals, [1, 99])
            else:
                vmin_val, vmax_val = 0, 1
            norm_scale = None  # use vmin/vmax directly

        for idx, t in enumerate(show_bins):
            ax_f = fig.add_subplot(gs_inner[idx])
            frame = video[t].copy()
            mask = np.isnan(frame)
            display = np.ma.masked_where(mask, frame)

            if is_normalized:
                ax_f.imshow(display, cmap=cmap_frame, norm=norm_scale,
                            aspect="equal", interpolation="nearest")
            else:
                ax_f.imshow(display, cmap=cmap_frame, vmin=vmin_val, vmax=vmax_val,
                            aspect="equal", interpolation="nearest")

            ax_f.set_title(f"h{t}", fontsize=8, fontweight="bold")
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            # Grid lines
            for r in range(H + 1):
                ax_f.axhline(r - 0.5, color="white", linewidth=0.3)
            for c in range(W + 1):
                ax_f.axvline(c - 0.5, color="white", linewidth=0.3)

            # Fill count below
            n_fill = frame_fills[t]
            ax_f.set_xlabel(f"{n_fill}/64", fontsize=7)

        # Add a colorbar for the heatmaps
        # Use the last axis for reference
        if n_show > 0:
            if is_normalized:
                sm = plt.cm.ScalarMappable(cmap=cmap_frame, norm=norm_scale)
            else:
                sm = plt.cm.ScalarMappable(
                    cmap=cmap_frame,
                    norm=plt.Normalize(vmin=vmin_val, vmax=vmax_val),
                )
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.25])
            cb = fig.colorbar(sm, cax=cbar_ax)
            if is_normalized:
                cb.set_label("z-score", fontsize=8)
            else:
                cb.set_label("raw value", fontsize=8)
            cb.ax.tick_params(labelsize=7)

        plt.savefig(outdir / f"video_admission_{hid}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outdir / f'video_admission_{hid}.png'}")

    # ================================================================
    # Figure 3: Aggregate density comparison
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 3a: Fill % distribution
    fills = [m["video_summary"]["fill_pct"] for m in all_meta]
    axes[0].bar(range(len(fills)), fills, color="#4C72B0", edgecolor="white")
    axes[0].set_xticks(range(len(fills)))
    axes[0].set_xticklabels([f"adm_{m['hadm_id']}" for m in all_meta],
                             fontsize=6, rotation=45, ha="right")
    axes[0].set_ylabel("Overall fill %", fontsize=9)
    axes[0].set_title("Video Fill % (cells with data / total cells)",
                      fontsize=10, fontweight="bold")
    axes[0].axhline(np.mean(fills), color="#D65F5F", linestyle="--",
                    linewidth=1, label=f"mean={np.mean(fills):.1f}%")
    axes[0].legend(fontsize=8)

    # 3b: Temporal density (frames with data / 24)
    td = [m["video_summary"]["temporal_density_pct"] for m in all_meta]
    axes[1].bar(range(len(td)), td, color="#55A868", edgecolor="white")
    axes[1].set_xticks(range(len(td)))
    axes[1].set_xticklabels([f"adm_{m['hadm_id']}" for m in all_meta],
                             fontsize=6, rotation=45, ha="right")
    axes[1].set_ylabel("Temporal density %", fontsize=9)
    axes[1].set_title("Temporal Density (frames with data / 24)",
                      fontsize=10, fontweight="bold")
    axes[1].axhline(np.mean(td), color="#D65F5F", linestyle="--",
                    linewidth=1, label=f"mean={np.mean(td):.1f}%")
    axes[1].legend(fontsize=8)

    # 3c: Per-frame fill % (all frames with data, across all videos)
    per_frame_fills = []
    for meta in all_meta:
        for fm in meta["frames"]:
            if fm["n_cells_filled"] > 0:
                per_frame_fills.append(fm["fill_pct"])

    if per_frame_fills:
        axes[2].hist(per_frame_fills, bins=20, color="#8172B2",
                     edgecolor="white", alpha=0.85)
        axes[2].axvline(np.median(per_frame_fills), color="#D65F5F",
                        linestyle="--", linewidth=1.2,
                        label=f"median={np.median(per_frame_fills):.1f}%")
        axes[2].set_xlabel("Fill % per frame", fontsize=9)
        axes[2].set_ylabel("Count", fontsize=9)
        axes[2].set_title("Per-Frame Fill % (non-empty frames only)",
                          fontsize=10, fontweight="bold")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "video_density_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'video_density_summary.png'}")

    # ================================================================
    # Figure 4 (normalized only): Z-score distribution across all videos
    # ================================================================
    if is_normalized:
        all_values = []
        for video in videos.values():
            vals = video[~np.isnan(video)]
            if len(vals) > 0:
                all_values.append(vals)

        if all_values:
            all_z = np.concatenate(all_values)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            # 4a: Z-score histogram
            axes[0].hist(all_z, bins=100, color="#4C72B0", edgecolor="none",
                         alpha=0.85, density=True)
            axes[0].axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
            axes[0].axvline(-3, color="#D65F5F", linewidth=1, linestyle="--", alpha=0.7,
                            label="z = ±3")
            axes[0].axvline(3, color="#D65F5F", linewidth=1, linestyle="--", alpha=0.7)
            axes[0].set_xlabel("Z-score", fontsize=10)
            axes[0].set_ylabel("Density", fontsize=10)
            axes[0].set_title("Z-Score Distribution (all videos, all labs)",
                              fontsize=11, fontweight="bold")
            axes[0].legend(fontsize=9)
            axes[0].set_xlim(-8, 8)

            # 4b: Fraction of extreme values per video
            extreme_fracs = []
            labels_vid = []
            for hid, video in videos.items():
                vals = video[~np.isnan(video)]
                if len(vals) > 0:
                    n_extreme = np.sum(np.abs(vals) > 3)
                    extreme_fracs.append(100.0 * n_extreme / len(vals))
                    labels_vid.append(f"adm_{hid}")

            axes[1].bar(range(len(extreme_fracs)), extreme_fracs,
                        color="#C44E52", edgecolor="white")
            axes[1].set_xticks(range(len(extreme_fracs)))
            axes[1].set_xticklabels(labels_vid, fontsize=7, rotation=45, ha="right")
            axes[1].set_ylabel("% values with |z| > 3", fontsize=9)
            axes[1].set_title("Extreme Value Fraction per Video (|z| > 3)",
                              fontsize=11, fontweight="bold")
            axes[1].axhline(np.mean(extreme_fracs), color="#D65F5F", linestyle="--",
                            linewidth=1, label=f"mean={np.mean(extreme_fracs):.1f}%")
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            fig.savefig(outdir / "zscore_distribution.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: {outdir / 'zscore_distribution.png'}")

    print(f"\nAll visualizations saved to {outdir}/")


if __name__ == "__main__":
    main()
