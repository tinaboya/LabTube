"""
Generate distribution plots for the MIMIC-IV lab events dataset analysis.

Produces figures in docs/figures/ for the data analysis report.

Usage:
    python scripts/admission_plots.py
    python scripts/admission_plots.py --parquet MIMIC4/lab_events_with_adm.parquet --outdir docs/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_LAB_WITH_ADM_PARQUET


def _hist(ax, data, title, xlabel, ylabel="Count", bins=50, color="#4C72B0",
          log_y=False, vlines=None, clip_upper=None, xlim=None):
    """Draw a styled histogram on a given axes."""
    d = np.array(data, dtype=float)
    if clip_upper is not None:
        d = d[d <= clip_upper]
    ax.hist(d, bins=bins, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if log_y:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if vlines:
        for v, lbl, clr in vlines:
            ax.axvline(v, color=clr, linestyle="--", linewidth=1.2, label=f"{lbl}={v:.1f}")
        ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate distribution plots.")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "figures")
    args = parser.parse_args()

    pq_path = args.parquet or DEFAULT_LAB_WITH_ADM_PARQUET
    if not pq_path.exists():
        print(f"Parquet not found: {pq_path}")
        return

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    import duckdb
    pq = pq_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()

    # ============================================================
    # 1. Per-specimen: n_labs and grid fill %
    # ============================================================
    print("Querying per-specimen stats...")
    rows = con.execute(f"""
        SELECT COUNT(DISTINCT itemid) AS n_labs,
               COUNT(*) AS n_events
        FROM read_parquet('{pq}')
        WHERE specimen_id IS NOT NULL AND hadm_id IS NOT NULL
        GROUP BY specimen_id
    """).fetchall()
    labs_per_spec = [float(r[0]) for r in rows]
    events_per_spec = [float(r[1]) for r in rows]
    fill_pct = [100.0 * v / 256.0 for v in labs_per_spec]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    med_labs = float(np.median(labs_per_spec))
    _hist(axes[0], labs_per_spec, "Labs per Specimen (per frame)", "# distinct lab types",
          log_y=True, vlines=[(med_labs, "median", "#D65F5F")])
    med_events = float(np.median(events_per_spec))
    _hist(axes[1], events_per_spec, "Events per Specimen", "# lab events",
          log_y=True, vlines=[(med_events, "median", "#D65F5F")])
    med_fill = float(np.median(fill_pct))
    _hist(axes[2], fill_pct, "Grid Fill % per Specimen (of 256)", "fill %",
          log_y=True, vlines=[(med_fill, "median", "#D65F5F")], clip_upper=50)
    plt.tight_layout()
    fig.savefig(outdir / "per_specimen_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'per_specimen_distributions.png'}")

    # ============================================================
    # 2. Per-admission: n_events, n_labs, n_specimens, hours, freq
    # ============================================================
    print("Querying per-admission stats...")
    rows = con.execute(f"""
        SELECT COUNT(*) AS n_events,
               COUNT(DISTINCT itemid) AS n_labs,
               COUNT(DISTINCT specimen_id) AS n_specimens,
               EXTRACT(EPOCH FROM (MAX(charttime) - MIN(charttime))) / 3600.0 AS hours
        FROM read_parquet('{pq}')
        WHERE hadm_id IS NOT NULL AND charttime IS NOT NULL
        GROUP BY hadm_id
    """).fetchall()
    events_per_adm = [float(r[0]) for r in rows]
    labs_per_adm = [float(r[1]) for r in rows]
    specs_per_adm = [float(r[2]) for r in rows]
    hours_per_adm = [float(r[3]) if r[3] is not None else 0.0 for r in rows]
    specs_per_hour = [s / h if h > 0 else 0.0 for s, h in zip(specs_per_adm, hours_per_adm)]

    # --- Figure 2a: n_events / admission ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    med = float(np.median(events_per_adm))
    _hist(axes[0], events_per_adm, "Lab Events per Admission", "# events",
          log_y=True, vlines=[(med, "median", "#D65F5F")], clip_upper=2000)
    med = float(np.median(labs_per_adm))
    _hist(axes[1], labs_per_adm, "Distinct Lab Types per Admission", "# distinct labs",
          log_y=True, vlines=[(med, "median", "#D65F5F")])
    med = float(np.median(specs_per_adm))
    _hist(axes[2], specs_per_adm, "Specimens (Frames) per Admission", "# specimens",
          log_y=True, vlines=[(med, "median", "#D65F5F")], clip_upper=200)
    plt.tight_layout()
    fig.savefig(outdir / "per_admission_counts.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'per_admission_counts.png'}")

    # --- Figure 2b: hours and frequency ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    med = float(np.median(hours_per_adm))
    _hist(axes[0], hours_per_adm, "Admission Duration (hours)", "hours (first→last charttime)",
          log_y=True, vlines=[(med, "median", "#D65F5F"), (24, "24h", "#5FA55A"), (168, "7d", "#B07AA1")],
          clip_upper=500, bins=60)
    valid_freq = [v for v in specs_per_hour if v > 0]
    med = float(np.median(valid_freq))
    _hist(axes[1], valid_freq, "Specimens per Hour", "specimens/hour",
          log_y=True, vlines=[(med, "median", "#D65F5F")], clip_upper=5, bins=60)
    plt.tight_layout()
    fig.savefig(outdir / "per_admission_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'per_admission_time.png'}")

    # ============================================================
    # 3. Time gaps between consecutive specimens
    # ============================================================
    print("Querying inter-specimen time gaps...")
    rows = con.execute(f"""
        WITH spec_times AS (
            SELECT hadm_id, specimen_id, MIN(charttime) AS spec_time
            FROM read_parquet('{pq}')
            WHERE hadm_id IS NOT NULL AND specimen_id IS NOT NULL AND charttime IS NOT NULL
            GROUP BY hadm_id, specimen_id
        ),
        with_lag AS (
            SELECT *,
                LAG(spec_time) OVER (PARTITION BY hadm_id ORDER BY spec_time) AS prev_time
            FROM spec_times
        )
        SELECT EXTRACT(EPOCH FROM (spec_time - prev_time)) / 3600.0 AS gap_hours
        FROM with_lag
        WHERE prev_time IS NOT NULL
    """).fetchall()
    gaps = [float(r[0]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    med = float(np.median(gaps))
    _hist(axes[0], gaps, "Time Between Specimens (all)", "hours",
          log_y=True, vlines=[(med, "median", "#D65F5F")], clip_upper=48, bins=80)
    short_gaps = [g for g in gaps if g <= 2.0]
    med_short = float(np.median(short_gaps)) if short_gaps else 0
    _hist(axes[1], short_gaps, "Time Between Specimens (≤2h zoom)", "hours",
          log_y=True, vlines=[(med_short, "median", "#D65F5F")], bins=80)
    plt.tight_layout()
    fig.savefig(outdir / "specimen_time_gaps.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'specimen_time_gaps.png'}")

    # ============================================================
    # 4. Top lab types by frequency + coverage
    # ============================================================
    print("Querying lab frequency...")
    rows = con.execute(f"""
        SELECT itemid,
               COUNT(*) AS n_events,
               COUNT(DISTINCT hadm_id) AS n_admissions
        FROM read_parquet('{pq}')
        WHERE hadm_id IS NOT NULL
        GROUP BY itemid
        ORDER BY n_events DESC
    """).fetchall()
    lab_freq = [float(r[1]) for r in rows]
    lab_adm_coverage = [float(r[2]) for r in rows]
    total_adm = 447689  # from earlier

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Top 50 labs bar chart
    top_n = 50
    top_itemids = [str(r[0]) for r in rows[:top_n]]
    top_counts = [float(r[1]) / 1e6 for r in rows[:top_n]]
    axes[0].barh(range(top_n), top_counts[::-1], color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_itemids[::-1], fontsize=5)
    axes[0].set_xlabel("Events (millions)", fontsize=9)
    axes[0].set_title(f"Top {top_n} Lab Types by Event Count", fontsize=11, fontweight="bold")
    axes[0].tick_params(labelsize=7)

    # Cumulative coverage curve (what % of admissions include lab rank k)
    coverage_pct = [100 * c / total_adm for c in lab_adm_coverage]
    axes[1].plot(range(len(coverage_pct)), coverage_pct, color="#4C72B0", linewidth=1.2)
    axes[1].axhline(50, color="#D65F5F", linestyle="--", linewidth=0.8, label="50% admissions")
    axes[1].axvline(256, color="#5FA55A", linestyle="--", linewidth=0.8, label="Top 256 (grid)")
    axes[1].set_xlabel("Lab type (ranked by frequency)", fontsize=9)
    axes[1].set_ylabel("% of admissions containing this lab", fontsize=9)
    axes[1].set_title("Lab Coverage Across Admissions", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, min(800, len(coverage_pct)))
    axes[1].tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "lab_frequency_coverage.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'lab_frequency_coverage.png'}")

    # ============================================================
    # 5. Admission subsets by duration bucket
    # ============================================================
    print("Computing duration-bucket breakdown...")
    buckets = [
        ("<24h", 0, 24),
        ("24h–72h", 24, 72),
        ("72h–7d", 72, 168),
        ("7d–14d", 168, 336),
        (">14d", 336, 1e9),
    ]
    bucket_data = {b[0]: {"n_adm": 0, "specs": [], "labs": [], "events": []} for b in buckets}
    for ev, lb, sp, h in zip(events_per_adm, labs_per_adm, specs_per_adm, hours_per_adm):
        for name, lo, hi in buckets:
            if lo <= h < hi:
                bucket_data[name]["n_adm"] += 1
                bucket_data[name]["specs"].append(sp)
                bucket_data[name]["labs"].append(lb)
                bucket_data[name]["events"].append(ev)
                break

    bucket_names = [b[0] for b in buckets]
    bucket_counts = [bucket_data[b]["n_adm"] for b in bucket_names]
    bucket_med_specs = [float(np.median(bucket_data[b]["specs"])) if bucket_data[b]["specs"] else 0 for b in bucket_names]
    bucket_med_labs = [float(np.median(bucket_data[b]["labs"])) if bucket_data[b]["labs"] else 0 for b in bucket_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    axes[0].bar(bucket_names, bucket_counts, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Admissions by Duration Bucket", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("# admissions", fontsize=9)
    for i, v in enumerate(bucket_counts):
        axes[0].text(i, v + max(bucket_counts)*0.01, f"{v:,}", ha="center", fontsize=7)
    axes[0].tick_params(labelsize=8)

    axes[1].bar(bucket_names, bucket_med_specs, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Median Frames per Bucket", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("median # specimens", fontsize=9)
    for i, v in enumerate(bucket_med_specs):
        axes[1].text(i, v + max(bucket_med_specs)*0.01, f"{v:.0f}", ha="center", fontsize=7)
    axes[1].tick_params(labelsize=8)

    axes[2].bar(bucket_names, bucket_med_labs, color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Median Lab Types per Bucket", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("median # distinct labs", fontsize=9)
    for i, v in enumerate(bucket_med_labs):
        axes[2].text(i, v + max(bucket_med_labs)*0.01, f"{v:.0f}", ha="center", fontsize=7)
    axes[2].tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "duration_bucket_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'duration_bucket_breakdown.png'}")

    # ============================================================
    # 6. Subset feasibility: how many admissions meet various thresholds
    # ============================================================
    print("Computing feasibility thresholds...")
    specs_arr = np.array(specs_per_adm)
    labs_arr = np.array(labs_per_adm)
    hours_arr = np.array(hours_per_adm)
    total = len(specs_arr)

    thresholds_specs = [4, 8, 16, 24, 32, 48, 64]
    thresholds_labs = [20, 30, 50, 80, 100]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts_specs = [int(np.sum(specs_arr >= t)) for t in thresholds_specs]
    axes[0].bar([str(t) for t in thresholds_specs], counts_specs, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Admissions with ≥ N Specimens (Frames)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("minimum # specimens", fontsize=9)
    axes[0].set_ylabel("# admissions meeting threshold", fontsize=9)
    for i, (t, c) in enumerate(zip(thresholds_specs, counts_specs)):
        axes[0].text(i, c + total * 0.005, f"{c:,}\n({100*c/total:.0f}%)", ha="center", fontsize=7)
    axes[0].tick_params(labelsize=8)

    counts_labs = [int(np.sum(labs_arr >= t)) for t in thresholds_labs]
    axes[1].bar([str(t) for t in thresholds_labs], counts_labs, color="#55A868", edgecolor="white")
    axes[1].set_title("Admissions with ≥ N Lab Types", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("minimum # distinct lab types", fontsize=9)
    axes[1].set_ylabel("# admissions meeting threshold", fontsize=9)
    for i, (t, c) in enumerate(zip(thresholds_labs, counts_labs)):
        axes[1].text(i, c + total * 0.005, f"{c:,}\n({100*c/total:.0f}%)", ha="center", fontsize=7)
    axes[1].tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "feasibility_thresholds.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'feasibility_thresholds.png'}")

    con.close()
    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
