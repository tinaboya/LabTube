"""
Generate distribution plots for the eICU lab events dataset analysis.

Produces figures in docs/figures_eicu/ for the eICU data analysis report.

Usage:
    uv run python scripts/eicu/admission_plots.py
    uv run python scripts/eicu/admission_plots.py --parquet eICU/lab_events_with_adm.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_LAB_WITH_ADM_PARQUET
from schema_utils import _col_expr, _specimen_expr, resolve_schema_columns


def _hist(
    ax,
    data,
    title,
    xlabel,
    ylabel="Count",
    bins=50,
    color="#4C72B0",
    log_y=False,
    vlines=None,
    clip_upper=None,
    xlim=None,
):
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


def _collection_time_expr(cols: dict[str, str | None]) -> str:
    """Per-collection time coordinate in hours, preferring eICU minute offsets."""
    if cols.get("labresultoffset"):
        offset = _col_expr(cols, "labresultoffset")
        return f"MIN({offset}) / 60.0"
    chart = _col_expr(cols, "charttime")
    return f"EXTRACT(EPOCH FROM MIN({chart})) / 3600.0"


def _duration_hours_expr(cols: dict[str, str | None]) -> str:
    """Stay span in hours, preferring the native offset column."""
    if cols.get("labresultoffset"):
        offset = _col_expr(cols, "labresultoffset")
        return f"(MAX({offset}) - MIN({offset})) / 60.0"
    chart = _col_expr(cols, "charttime")
    return f"EXTRACT(EPOCH FROM (MAX({chart}) - MIN({chart}))) / 3600.0"


def _short_label(text: str | None, limit: int = 22) -> str:
    """Keep bar-chart labels readable."""
    if not text:
        return "NA"
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import duckdb

    parser = argparse.ArgumentParser(description="Generate eICU distribution plots.")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "docs" / "figures_eicu",
    )
    args = parser.parse_args()

    pq_path = args.parquet or DEFAULT_LAB_WITH_ADM_PARQUET
    if not pq_path.exists():
        print(f"Parquet not found: {pq_path}")
        return

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    pq = pq_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    cols = resolve_schema_columns(con, pq)

    hadm = _col_expr(cols, "hadm_id")
    item = _col_expr(cols, "itemid")
    value = _col_expr(cols, "valuenum")
    collection_key = _specimen_expr(cols, prefer_offset=True)
    duration_expr = _duration_hours_expr(cols)
    collection_time_expr = _collection_time_expr(cols)
    label_col = _col_expr(cols, "label") if cols.get("label") else f"CAST({item} AS VARCHAR)"

    n_lab_codes = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT {item})
            FROM read_parquet('{pq}')
            WHERE {item} IS NOT NULL
            """
        ).fetchone()[0]
    )
    total_stays = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT {hadm})
            FROM read_parquet('{pq}')
            WHERE {hadm} IS NOT NULL
            """
        ).fetchone()[0]
    )

    # ============================================================
    # 1. Per-collection: n_labs and fill %
    # ============================================================
    print("Querying per-collection stats...")
    rows = con.execute(
        f"""
        SELECT COUNT(DISTINCT {item}) AS n_labs,
               COUNT(*) AS n_events
        FROM read_parquet('{pq}')
        WHERE {hadm} IS NOT NULL AND {collection_key} IS NOT NULL
        GROUP BY {hadm}, {collection_key}
        """
    ).fetchall()
    labs_per_collection = [float(r[0]) for r in rows]
    events_per_collection = [float(r[1]) for r in rows]
    fill_pct = [100.0 * v / max(n_lab_codes, 1) for v in labs_per_collection]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    med_labs = float(np.median(labs_per_collection))
    _hist(
        axes[0],
        labs_per_collection,
        "Labs per Collection (per frame surrogate)",
        "# distinct lab codes",
        log_y=True,
        vlines=[(med_labs, "median", "#D65F5F")],
    )
    med_events = float(np.median(events_per_collection))
    _hist(
        axes[1],
        events_per_collection,
        "Events per Collection",
        "# lab events",
        log_y=True,
        vlines=[(med_events, "median", "#D65F5F")],
    )
    med_fill = float(np.median(fill_pct))
    _hist(
        axes[2],
        fill_pct,
        f"Vocabulary Fill % per Collection (of {n_lab_codes})",
        "fill %",
        log_y=True,
        vlines=[(med_fill, "median", "#D65F5F")],
        clip_upper=50,
    )
    plt.tight_layout()
    fig.savefig(outdir / "per_collection_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'per_collection_distributions.png'}")

    # ============================================================
    # 2. Per-stay: n_events, n_labs, n_collections, hours, freq
    # ============================================================
    print("Querying per-stay stats...")
    rows = con.execute(
        f"""
        SELECT COUNT(*) AS n_events,
               COUNT(DISTINCT {item}) AS n_labs,
               COUNT(DISTINCT {collection_key}) AS n_collections,
               {duration_expr} AS hours
        FROM read_parquet('{pq}')
        WHERE {hadm} IS NOT NULL
          AND {item} IS NOT NULL
          AND {collection_key} IS NOT NULL
        GROUP BY {hadm}
        """
    ).fetchall()
    events_per_stay = [float(r[0]) for r in rows]
    labs_per_stay = [float(r[1]) for r in rows]
    collections_per_stay = [float(r[2]) for r in rows]
    hours_per_stay = [float(r[3]) if r[3] is not None else 0.0 for r in rows]
    collections_per_hour = [c / h for c, h in zip(collections_per_stay, hours_per_stay) if h > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    med = float(np.median(events_per_stay))
    _hist(
        axes[0],
        events_per_stay,
        "Lab Events per ICU Stay",
        "# events",
        log_y=True,
        vlines=[(med, "median", "#D65F5F")],
        clip_upper=2000,
    )
    med = float(np.median(labs_per_stay))
    _hist(
        axes[1],
        labs_per_stay,
        "Distinct Lab Codes per ICU Stay",
        "# distinct labs",
        log_y=True,
        vlines=[(med, "median", "#D65F5F")],
    )
    med = float(np.median(collections_per_stay))
    _hist(
        axes[2],
        collections_per_stay,
        "Collections (Frames) per ICU Stay",
        "# merged draws",
        log_y=True,
        vlines=[(med, "median", "#D65F5F")],
        clip_upper=250,
    )
    plt.tight_layout()
    fig.savefig(outdir / "per_stay_counts.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'per_stay_counts.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    med = float(np.median(hours_per_stay))
    _hist(
        axes[0],
        hours_per_stay,
        "Stay Duration (hours)",
        "hours (first→last collection)",
        log_y=True,
        vlines=[
            (med, "median", "#D65F5F"),
            (24, "24h", "#5FA55A"),
            (168, "7d", "#B07AA1"),
        ],
        clip_upper=500,
        bins=60,
    )
    valid_freq = [v for v in collections_per_hour if v > 0]
    med = float(np.median(valid_freq))
    _hist(
        axes[1],
        valid_freq,
        "Collections per Hour",
        "collections/hour",
        log_y=True,
        vlines=[(med, "median", "#D65F5F")],
        clip_upper=5,
        bins=60,
    )
    plt.tight_layout()
    fig.savefig(outdir / "per_stay_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'per_stay_time.png'}")

    # ============================================================
    # 3. Time gaps between consecutive collections
    # ============================================================
    print("Querying inter-collection time gaps...")
    rows = con.execute(
        f"""
        WITH collections AS (
            SELECT {hadm} AS hadm_id_key,
                   {collection_key} AS collection_key,
                   {collection_time_expr} AS collection_time_hours
            FROM read_parquet('{pq}')
            WHERE {hadm} IS NOT NULL AND {collection_key} IS NOT NULL
            GROUP BY {hadm}, {collection_key}
        ),
        with_lag AS (
            SELECT *,
                   LAG(collection_time_hours) OVER (
                       PARTITION BY hadm_id_key
                       ORDER BY collection_time_hours
                   ) AS prev_collection_time_hours
            FROM collections
        )
        SELECT collection_time_hours - prev_collection_time_hours AS gap_hours
        FROM with_lag
        WHERE prev_collection_time_hours IS NOT NULL
        """
    ).fetchall()
    gaps = [float(r[0]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    med = float(np.median(gaps))
    _hist(
        axes[0],
        gaps,
        "Time Between Collections (all)",
        "hours",
        log_y=True,
        vlines=[(med, "median", "#D65F5F")],
        clip_upper=48,
        bins=80,
    )
    short_gaps = [g for g in gaps if g <= 24.0]
    med_short = float(np.median(short_gaps)) if short_gaps else 0.0
    _hist(
        axes[1],
        short_gaps,
        "Time Between Collections (≤24h zoom)",
        "hours",
        log_y=True,
        vlines=[(med_short, "median", "#D65F5F")],
        bins=80,
    )
    plt.tight_layout()
    fig.savefig(outdir / "collection_time_gaps.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'collection_time_gaps.png'}")

    # ============================================================
    # 4. Top lab types by frequency + coverage
    # ============================================================
    print("Querying lab frequency...")
    rows = con.execute(
        f"""
        SELECT {item} AS itemid,
               ANY_VALUE({label_col}) AS label,
               COUNT(*) AS n_events,
               COUNT(DISTINCT {hadm}) AS n_stays
        FROM read_parquet('{pq}')
        WHERE {hadm} IS NOT NULL
          AND {item} IS NOT NULL
        GROUP BY {item}
        ORDER BY n_events DESC
        """
    ).fetchall()
    lab_stay_coverage = [100.0 * float(r[3]) / max(total_stays, 1) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    top_n = min(30, len(rows))
    top_labels = [_short_label(str(r[1])) for r in rows[:top_n]]
    top_counts = [float(r[2]) / 1e6 for r in rows[:top_n]]
    axes[0].barh(
        range(top_n),
        top_counts[::-1],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_labels[::-1], fontsize=7)
    axes[0].set_xlabel("Events (millions)", fontsize=9)
    axes[0].set_title(f"Top {top_n} Lab Types by Event Count", fontsize=11, fontweight="bold")
    axes[0].tick_params(labelsize=7)

    axes[1].plot(range(1, len(lab_stay_coverage) + 1), lab_stay_coverage, color="#4C72B0", linewidth=1.2)
    axes[1].axhline(50, color="#D65F5F", linestyle="--", linewidth=0.8, label="50% stays")
    axes[1].axvline(64, color="#5FA55A", linestyle="--", linewidth=0.8, label="Top 64 grid")
    axes[1].set_xlabel("Lab type (ranked by frequency)", fontsize=9)
    axes[1].set_ylabel("% of stays containing this lab", fontsize=9)
    axes[1].set_title("Lab Coverage Across ICU Stays", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(1, len(lab_stay_coverage))
    axes[1].tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "lab_frequency_coverage.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'lab_frequency_coverage.png'}")

    # ============================================================
    # 5. Stay subsets by duration bucket
    # ============================================================
    print("Computing duration-bucket breakdown...")
    buckets = [
        ("<24h", 0, 24),
        ("24h–72h", 24, 72),
        ("72h–7d", 72, 168),
        ("7d–14d", 168, 336),
        (">14d", 336, 1e9),
    ]
    bucket_data = {b[0]: {"n_stay": 0, "collections": [], "labs": [], "events": []} for b in buckets}
    for ev, lb, col, hrs in zip(events_per_stay, labs_per_stay, collections_per_stay, hours_per_stay):
        for name, lo, hi in buckets:
            if lo <= hrs < hi:
                bucket_data[name]["n_stay"] += 1
                bucket_data[name]["collections"].append(col)
                bucket_data[name]["labs"].append(lb)
                bucket_data[name]["events"].append(ev)
                break

    bucket_names = [b[0] for b in buckets]
    bucket_counts = [bucket_data[b]["n_stay"] for b in bucket_names]
    bucket_med_collections = [
        float(np.median(bucket_data[b]["collections"])) if bucket_data[b]["collections"] else 0.0
        for b in bucket_names
    ]
    bucket_med_labs = [
        float(np.median(bucket_data[b]["labs"])) if bucket_data[b]["labs"] else 0.0
        for b in bucket_names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    axes[0].bar(bucket_names, bucket_counts, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("ICU Stays by Duration Bucket", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("# stays", fontsize=9)
    for i, v in enumerate(bucket_counts):
        axes[0].text(i, v + max(bucket_counts) * 0.01, f"{v:,}", ha="center", fontsize=7)
    axes[0].tick_params(labelsize=8)

    axes[1].bar(bucket_names, bucket_med_collections, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Median Frames per Bucket", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("median # collections", fontsize=9)
    for i, v in enumerate(bucket_med_collections):
        axes[1].text(i, v + max(bucket_med_collections) * 0.01, f"{v:.0f}", ha="center", fontsize=7)
    axes[1].tick_params(labelsize=8)

    axes[2].bar(bucket_names, bucket_med_labs, color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Median Lab Codes per Bucket", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("median # distinct labs", fontsize=9)
    for i, v in enumerate(bucket_med_labs):
        axes[2].text(i, v + max(bucket_med_labs) * 0.01, f"{v:.0f}", ha="center", fontsize=7)
    axes[2].tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "duration_bucket_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'duration_bucket_breakdown.png'}")

    # ============================================================
    # 6. Subset feasibility thresholds
    # ============================================================
    print("Computing feasibility thresholds...")
    collections_arr = np.array(collections_per_stay)
    labs_arr = np.array(labs_per_stay)
    total = len(collections_arr)

    thresholds_collections = [4, 8, 16, 24, 32, 48, 64]
    thresholds_labs = [20, 30, 50, 80, 100]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts_collections = [int(np.sum(collections_arr >= t)) for t in thresholds_collections]
    axes[0].bar(
        [str(t) for t in thresholds_collections],
        counts_collections,
        color="#4C72B0",
        edgecolor="white",
    )
    axes[0].set_title("ICU Stays with ≥ N Collections", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("minimum # collections", fontsize=9)
    axes[0].set_ylabel("# stays meeting threshold", fontsize=9)
    for i, c in enumerate(counts_collections):
        axes[0].text(i, c + total * 0.005, f"{c:,}\n({100 * c / total:.0f}%)", ha="center", fontsize=7)
    axes[0].tick_params(labelsize=8)

    counts_labs = [int(np.sum(labs_arr >= t)) for t in thresholds_labs]
    axes[1].bar([str(t) for t in thresholds_labs], counts_labs, color="#55A868", edgecolor="white")
    axes[1].set_title("ICU Stays with ≥ N Lab Codes", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("minimum # distinct lab codes", fontsize=9)
    axes[1].set_ylabel("# stays meeting threshold", fontsize=9)
    for i, c in enumerate(counts_labs):
        axes[1].text(i, c + total * 0.005, f"{c:,}\n({100 * c / total:.0f}%)", ha="center", fontsize=7)
    axes[1].tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "feasibility_thresholds.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'feasibility_thresholds.png'}")

    # ============================================================
    # 7. Offset timing relative to admission (eICU-specific)
    # ============================================================
    if cols.get("labresultoffset"):
        print("Computing offset-relative timing...")
        offset = _col_expr(cols, "labresultoffset")
        rows = con.execute(
            f"""
            SELECT MIN({offset}) / 60.0 AS first_hours,
                   MAX({offset}) / 60.0 AS last_hours
            FROM read_parquet('{pq}')
            WHERE {hadm} IS NOT NULL AND {offset} IS NOT NULL
            GROUP BY {hadm}
            """
        ).fetchall()
        first_offsets = [float(r[0]) for r in rows if r[0] is not None]
        last_offsets = [float(r[1]) for r in rows if r[1] is not None]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        med_first = float(np.median(first_offsets))
        _hist(
            axes[0],
            first_offsets,
            "First Collection Offset",
            "hours from admission",
            log_y=True,
            vlines=[(med_first, "median", "#D65F5F"), (0, "admit", "#5FA55A")],
            clip_upper=24,
            xlim=(-120, 24),
            bins=80,
        )
        med_last = float(np.median(last_offsets))
        _hist(
            axes[1],
            last_offsets,
            "Last Collection Offset",
            "hours from admission",
            log_y=True,
            vlines=[(med_last, "median", "#D65F5F"), (24, "24h", "#5FA55A")],
            clip_upper=500,
            bins=80,
        )
        plt.tight_layout()
        fig.savefig(outdir / "offset_relative_to_admission.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {outdir / 'offset_relative_to_admission.png'}")

    con.close()
    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
