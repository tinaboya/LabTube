"""
Analyse the ≤23-specimen admission subset to determine:
  1. Subset size and basic stats
  2. Which lab types (itemids) are most frequent in this subset
  3. Per-frame fill rate at various grid sizes (8×8, 10×10, 12×12, 16×16)
  4. Recommended grid + lab selection

Usage:
    python scripts/subset_analysis.py
    python scripts/subset_analysis.py --max-specimens 23
    python scripts/subset_analysis.py --max-specimens 16 --parquet MIMIC4/lab_events_with_adm.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_LAB_WITH_ADM_PARQUET, DEFAULT_D_LABITEMS


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset analysis for grid/lab selection.")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--max-specimens", type=int, default=23,
                        help="Maximum specimens per admission to include (default: 23, i.e. ≤p75)")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "figures")
    args = parser.parse_args()

    pq_path = args.parquet or DEFAULT_LAB_WITH_ADM_PARQUET
    if not pq_path.exists():
        print(f"Parquet not found: {pq_path}")
        return

    max_spec = args.max_specimens
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    import duckdb
    import numpy as np

    pq = pq_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()

    # ================================================================
    # 0. Identify admissions in the subset
    # ================================================================
    print(f"Filtering to admissions with ≤{max_spec} specimens...")
    con.execute(f"""
        CREATE TEMPORARY TABLE adm_subset AS
        SELECT hadm_id, COUNT(DISTINCT specimen_id) AS n_specimens
        FROM read_parquet('{pq}')
        WHERE hadm_id IS NOT NULL AND specimen_id IS NOT NULL
        GROUP BY hadm_id
        HAVING COUNT(DISTINCT specimen_id) <= {max_spec}
    """)
    subset_info = con.execute("""
        SELECT COUNT(*) AS n_adm,
               AVG(n_specimens) AS mean_spec,
               MEDIAN(n_specimens) AS med_spec,
               MIN(n_specimens) AS min_spec,
               MAX(n_specimens) AS max_spec
        FROM adm_subset
    """).fetchone()
    n_adm = int(subset_info[0])
    print(f"\n{'='*65}")
    print(f"Subset: admissions with ≤{max_spec} specimens")
    print(f"{'='*65}")
    print(f"  # admissions:   {n_adm:,}")
    print(f"  specimens/adm:  mean={subset_info[1]:.1f}  median={subset_info[2]:.0f}  "
          f"min={subset_info[3]:.0f}  max={subset_info[4]:.0f}")

    # Total admissions for context
    total_adm = con.execute(f"""
        SELECT COUNT(DISTINCT hadm_id) FROM read_parquet('{pq}') WHERE hadm_id IS NOT NULL
    """).fetchone()[0]
    print(f"  % of all admissions: {100*n_adm/total_adm:.1f}%  ({n_adm:,} / {total_adm:,})")

    # ================================================================
    # 1. Per-admission lab counts in subset
    # ================================================================
    print(f"\nPer-admission stats (subset)...")
    rows = con.execute(f"""
        SELECT s.hadm_id,
               COUNT(*) AS n_events,
               COUNT(DISTINCT e.itemid) AS n_labs,
               s.n_specimens,
               EXTRACT(EPOCH FROM (MAX(e.charttime) - MIN(e.charttime))) / 3600.0 AS hours
        FROM read_parquet('{pq}') e
        JOIN adm_subset s ON e.hadm_id = s.hadm_id
        WHERE e.hadm_id IS NOT NULL AND e.charttime IS NOT NULL
        GROUP BY s.hadm_id, s.n_specimens
    """).fetchall()

    events_arr = np.array([r[1] for r in rows], dtype=float)
    labs_arr = np.array([r[2] for r in rows], dtype=float)
    specs_arr = np.array([r[3] for r in rows], dtype=float)
    hours_arr = np.array([r[4] if r[4] is not None else 0.0 for r in rows], dtype=float)

    def _pct_str(arr):
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
        return f"mean={np.mean(arr):.1f}  median={p50:.1f}  p5={p5:.1f}  p25={p25:.1f}  p75={p75:.1f}  p95={p95:.1f}"

    print(f"  n_events/adm:    {_pct_str(events_arr)}")
    print(f"  n_labs/adm:      {_pct_str(labs_arr)}")
    print(f"  n_specimens/adm: {_pct_str(specs_arr)}")
    print(f"  hours/adm:       {_pct_str(hours_arr)}")

    # ================================================================
    # 2. Lab frequency ranking in subset
    # ================================================================
    print(f"\nRanking lab types by frequency in subset...")

    # Load lab labels if available
    lab_labels = {}
    d_labitems_path = DEFAULT_D_LABITEMS
    if d_labitems_path.exists():
        label_rows = con.execute(f"""
            SELECT itemid, label, category
            FROM read_csv_auto('{d_labitems_path.resolve().as_posix()}')
        """).fetchall()
        lab_labels = {int(r[0]): (r[1], r[2]) for r in label_rows}

    lab_rows = con.execute(f"""
        SELECT e.itemid,
               COUNT(*) AS n_events,
               COUNT(DISTINCT e.hadm_id) AS n_admissions,
               COUNT(DISTINCT e.specimen_id) AS n_specimens
        FROM read_parquet('{pq}') e
        JOIN adm_subset s ON e.hadm_id = s.hadm_id
        WHERE e.hadm_id IS NOT NULL
        GROUP BY e.itemid
        ORDER BY n_events DESC
    """).fetchall()

    itemids = [int(r[0]) for r in lab_rows]
    lab_events = [int(r[1]) for r in lab_rows]
    lab_adms = [int(r[2]) for r in lab_rows]
    lab_specs = [int(r[3]) for r in lab_rows]

    print(f"  Total distinct lab types in subset: {len(itemids)}")

    # ================================================================
    # 3. Grid size analysis — per-frame fill rate for top-K
    # ================================================================
    print(f"\n{'='*65}")
    print("Grid size analysis: per-frame fill rate with top-K labs")
    print(f"{'='*65}")

    grid_options = [
        ("8×8",   64),
        ("10×10", 100),
        ("12×12", 144),
        ("16×16", 256),
    ]

    # For each grid size, compute per-specimen fill rate using only the top-K labs
    for label, k in grid_options:
        top_k_set = set(itemids[:k])
        top_k_csv = ",".join(str(x) for x in itemids[:k])

        # Per-specimen: count distinct itemids in top-K set
        fill_rows = con.execute(f"""
            SELECT COUNT(DISTINCT e.itemid) AS n_labs_in_grid
            FROM read_parquet('{pq}') e
            JOIN adm_subset s ON e.hadm_id = s.hadm_id
            WHERE e.hadm_id IS NOT NULL
              AND e.specimen_id IS NOT NULL
              AND e.itemid IN ({top_k_csv})
            GROUP BY e.specimen_id
        """).fetchall()

        fills = np.array([r[0] for r in fill_rows], dtype=float)
        fill_pcts = 100.0 * fills / k

        p5, p25, p50, p75, p95 = np.percentile(fill_pcts, [5, 25, 50, 75, 95])
        print(f"\n  Grid {label} (K={k})")
        print(f"    labs/frame:  mean={np.mean(fills):.1f}  median={np.median(fills):.0f}  p25={np.percentile(fills,25):.0f}  p75={np.percentile(fills,75):.0f}")
        print(f"    fill %:      mean={np.mean(fill_pcts):.1f}%  median={p50:.1f}%  p25={p25:.1f}%  p75={p75:.1f}%  p95={p95:.1f}%")

        # Per-admission: how many of the K labs ever appear during the whole stay
        adm_fill_rows = con.execute(f"""
            SELECT COUNT(DISTINCT e.itemid) AS n_labs_in_grid
            FROM read_parquet('{pq}') e
            JOIN adm_subset s ON e.hadm_id = s.hadm_id
            WHERE e.hadm_id IS NOT NULL
              AND e.itemid IN ({top_k_csv})
            GROUP BY e.hadm_id
        """).fetchall()
        adm_fills = np.array([r[0] for r in adm_fill_rows], dtype=float)
        adm_fill_pcts = 100.0 * adm_fills / k
        ap50 = np.median(adm_fill_pcts)
        ap25 = np.percentile(adm_fill_pcts, 25)
        ap75 = np.percentile(adm_fill_pcts, 75)
        print(f"    adm coverage: mean={np.mean(adm_fill_pcts):.1f}%  median={ap50:.1f}%  p25={ap25:.1f}%  p75={ap75:.1f}%")

    # ================================================================
    # 4. Print top labs
    # ================================================================
    print(f"\n{'='*65}")
    print("Top lab types in the ≤{}-specimen subset".format(max_spec))
    print(f"{'='*65}")

    print(f"\n  {'rank':>4}  {'itemid':>7}  {'events':>10}  {'adm':>8}  {'adm%':>6}  {'category':<20}  {'label'}")
    print(f"  {'----':>4}  {'------':>7}  {'------':>10}  {'---':>8}  {'----':>6}  {'--------':<20}  {'-----'}")
    for i, (iid, ev, ad, sp) in enumerate(zip(itemids, lab_events, lab_adms, lab_specs)):
        if i >= 100:
            break
        lbl, cat = lab_labels.get(iid, ("?", "?"))
        pct = 100 * ad / n_adm
        print(f"  {i+1:>4}  {iid:>7}  {ev:>10,}  {ad:>8,}  {pct:>5.1f}%  {cat:<20}  {lbl}")

    # ================================================================
    # 5. Generate comparison plot
    # ================================================================
    print(f"\nGenerating grid comparison plot...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: per-frame fill % by grid size
    grid_labels = []
    medians = []
    p25s = []
    p75s = []
    for label, k in grid_options:
        top_k_csv = ",".join(str(x) for x in itemids[:k])
        fill_rows = con.execute(f"""
            SELECT COUNT(DISTINCT e.itemid) AS n
            FROM read_parquet('{pq}') e
            JOIN adm_subset s ON e.hadm_id = s.hadm_id
            WHERE e.hadm_id IS NOT NULL AND e.specimen_id IS NOT NULL
              AND e.itemid IN ({top_k_csv})
            GROUP BY e.specimen_id
        """).fetchall()
        fills = 100.0 * np.array([r[0] for r in fill_rows], dtype=float) / k
        grid_labels.append(f"{label}\n(K={k})")
        medians.append(np.median(fills))
        p25s.append(np.percentile(fills, 25))
        p75s.append(np.percentile(fills, 75))

    x = range(len(grid_labels))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = axes[0].bar(x, medians, color=colors, edgecolor="white", linewidth=0.5)
    # Error bars for IQR
    for i in range(len(grid_labels)):
        axes[0].plot([i, i], [p25s[i], p75s[i]], color="black", linewidth=1.5)
        axes[0].plot([i-0.1, i+0.1], [p25s[i], p25s[i]], color="black", linewidth=1.5)
        axes[0].plot([i-0.1, i+0.1], [p75s[i], p75s[i]], color="black", linewidth=1.5)
    for i, m in enumerate(medians):
        axes[0].text(i, m + 1, f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grid_labels, fontsize=9)
    axes[0].set_ylabel("Fill % per frame (median ± IQR)", fontsize=10)
    axes[0].set_title(f"Per-Frame Grid Fill %\n(admissions ≤{max_spec} specimens)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, max(p75s) * 1.4)

    # Right: per-admission coverage % by grid size
    adm_medians = []
    adm_p25s = []
    adm_p75s = []
    for label, k in grid_options:
        top_k_csv = ",".join(str(x) for x in itemids[:k])
        adm_rows = con.execute(f"""
            SELECT COUNT(DISTINCT e.itemid) AS n
            FROM read_parquet('{pq}') e
            JOIN adm_subset s ON e.hadm_id = s.hadm_id
            WHERE e.hadm_id IS NOT NULL AND e.itemid IN ({top_k_csv})
            GROUP BY e.hadm_id
        """).fetchall()
        adm_f = 100.0 * np.array([r[0] for r in adm_rows], dtype=float) / k
        adm_medians.append(np.median(adm_f))
        adm_p25s.append(np.percentile(adm_f, 25))
        adm_p75s.append(np.percentile(adm_f, 75))

    bars2 = axes[1].bar(x, adm_medians, color=colors, edgecolor="white", linewidth=0.5)
    for i in range(len(grid_labels)):
        axes[1].plot([i, i], [adm_p25s[i], adm_p75s[i]], color="black", linewidth=1.5)
        axes[1].plot([i-0.1, i+0.1], [adm_p25s[i], adm_p25s[i]], color="black", linewidth=1.5)
        axes[1].plot([i-0.1, i+0.1], [adm_p75s[i], adm_p75s[i]], color="black", linewidth=1.5)
    for i, m in enumerate(adm_medians):
        axes[1].text(i, m + 1.5, f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grid_labels, fontsize=9)
    axes[1].set_ylabel("Coverage % per admission (median ± IQR)", fontsize=10)
    axes[1].set_title(f"Per-Admission Grid Coverage %\n(admissions ≤{max_spec} specimens)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, min(100, max(adm_p75s) * 1.3))

    plt.tight_layout()
    fig.savefig(outdir / "grid_size_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'grid_size_comparison.png'}")

    con.close()
    print(f"\n{'='*65}")
    print("Done.")


if __name__ == "__main__":
    main()
