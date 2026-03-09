"""
Convert admissions to VideoMAE-style videos: T×H×W tensors (RAW values).

Each admission becomes one .npy file of shape (T, H, W):
  - T   = number of time bins  (e.g. 24 for first 24 h with 1 h bins)
  - H×W = 8×8 = 64 lab types   (top-64 by frequency, artifact codes excluded)
  - value = last observed RAW lab value in each (time_bin, lab) cell; NaN = missing

This script outputs unnormalized videos. To normalize (log1p + z-score + clip),
run normalize_videos.py afterwards:

    python scripts/admission_to_video.py --top 100 --max-specimens 23
    python scripts/normalize_videos.py                     # ← new step
    python scripts/visualize_videos.py --video-dir lab_videos_normalized --normalized

Pipeline:
  1. Select the top-64 lab itemids from the admission subset (exclude QC/hold codes).
  2. Build an 8×8 grid mapping: itemid → (row, col), grouped by clinical category.
  3. For each admission:
     a. Fetch lab events from Parquet (only itemids in grid).
     b. Compute time_bin = floor((charttime − admittime) / bin_hours).
     c. Keep bins in [0, T).
     d. Within each (time_bin, itemid), take the LAST value by charttime.
     e. Fill a (T, 8, 8) tensor; save as admission_<hadm_id>.npy.
  4. Run normalize_videos.py to produce normalized versions (separate step).

Usage:
    # Build videos for top 100 richest admissions (≤23 specimens), 24h/1h bins
    python scripts/admission_to_video.py --top 100 --max-specimens 23

    # Build videos for specific admissions
    python scripts/admission_to_video.py --hadm-ids 22329603 23365149

    # All qualifying admissions (≤23 specimens, ≥4 frames), 24h window, 1h bins
    python scripts/admission_to_video.py --max-specimens 23 --min-specimens 4

    # Custom window: 48h, 2h bins → T=24
    python scripts/admission_to_video.py --window-hours 48 --bin-hours 2 --top 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_LAB_WITH_ADM_PARQUET, DEFAULT_D_LABITEMS


# ── Artifact / hold codes to exclude from grid ───────────────────────
EXCLUDE_ITEMIDS = {
    50934,  # H  (hemolyzed flag)
    51678,  # L  (lipemic flag)
    50947,  # I  (icteric flag)
    50933,  # Green Top Hold (plasma)
    50955,  # Light Green Top Hold
    50887,  # Blue Top Hold
    50979,  # Red Top Hold
    51103,  # Uhold
    51107,  # Urine tube, held
    52033,  # Specimen Type (blood gas metadata)
    50919,  # EDTA Hold
    51087,  # Length of Urine Collection (duration, not a lab value)
}


def _quote_ident(col: str) -> str:
    """Safely quote a DuckDB identifier."""
    return '"' + col.replace('"', '""') + '"'


def _col_expr(cols: dict[str, str | None], key: str, alias: str | None = None) -> str:
    """Return a quoted column expression, optionally with table alias."""
    col = cols.get(key)
    if not col:
        raise ValueError(f"Missing required mapped column: {key}")
    q = _quote_ident(col)
    return f"{alias}.{q}" if alias else q


def _specimen_expr(cols: dict[str, str | None], alias: str | None = None) -> str:
    """
    Return expression used as specimen key.

    If `specimen_id` is absent (common in eICU extracts), fallback to charttime
    as a deterministic surrogate for counting merged collections.
    """
    prefix = f"{alias}." if alias else ""
    if cols.get("specimen_id"):
        return f"CAST({prefix}{_quote_ident(cols['specimen_id'])} AS VARCHAR)"
    return f"CAST({prefix}{_quote_ident(cols['charttime'])} AS VARCHAR)"


def resolve_schema_columns(con, pq: str) -> dict[str, str | None]:
    """Map canonical names to actual parquet columns (MIMIC-IV vs eICU)."""
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pq}')").fetchall()
    available = {str(r[0]).lower(): str(r[0]) for r in schema_rows}

    def pick(candidates: list[str], required: bool = True) -> str | None:
        for c in candidates:
            if c.lower() in available:
                return available[c.lower()]
        if required:
            raise ValueError(
                f"Could not find any of {candidates} in parquet columns: "
                f"{sorted(available.values())}"
            )
        return None

    return {
        "hadm_id": pick(["hadm_id", "patientunitstayid"]),
        "itemid": pick(["itemid", "lab_code"]),
        "valuenum": pick(["valuenum", "labresult"]),
        "charttime": pick(["charttime"]),
        "admittime": pick(["admittime"]),
        "specimen_id": pick(["specimen_id"], required=False),
    }


def select_top_labs(
    con,
    pq: str,
    cols: dict[str, str | None],
    n_labs: int = 64,
    max_specimens: int | None = None,
    exclude: set[int] | None = None,
) -> list[int]:
    """Return top-N itemids by event count in the qualifying admission subset."""
    exclude = exclude or EXCLUDE_ITEMIDS
    excl_csv = ",".join(str(x) for x in exclude)
    hadm = _col_expr(cols, "hadm_id")
    hadm_e = _col_expr(cols, "hadm_id", alias="e")
    item = _col_expr(cols, "itemid")
    item_e = _col_expr(cols, "itemid", alias="e")
    specimen = _specimen_expr(cols)
    specimen_e = _specimen_expr(cols, alias="e")

    if max_specimens is not None:
        sql = f"""
        WITH adm_sub AS (
            SELECT {hadm} AS hadm_id_key
            FROM read_parquet('{pq}')
            WHERE {hadm} IS NOT NULL AND {specimen} IS NOT NULL
            GROUP BY {hadm}
            HAVING COUNT(DISTINCT {specimen}) <= {max_specimens}
        )
        SELECT {item_e} AS itemid, COUNT(*) AS cnt
        FROM read_parquet('{pq}') e
        JOIN adm_sub s ON {hadm_e} = s.hadm_id_key
        WHERE {hadm_e} IS NOT NULL
          AND {item_e} IS NOT NULL
          AND {specimen_e} IS NOT NULL
          AND {item_e} NOT IN ({excl_csv})
        GROUP BY {item_e}
        ORDER BY cnt DESC
        LIMIT {n_labs}
        """
    else:
        sql = f"""
        SELECT {item} AS itemid, COUNT(*) AS cnt
        FROM read_parquet('{pq}')
        WHERE {hadm} IS NOT NULL
          AND {item} IS NOT NULL
          AND {item} NOT IN ({excl_csv})
        GROUP BY {item}
        ORDER BY cnt DESC
        LIMIT {n_labs}
        """
    rows = con.execute(sql).fetchall()
    return [int(r[0]) for r in rows]


def build_grid_mapping(
    con,
    itemids: list[int],
    d_labitems_path: Path,
    grid_shape: tuple[int, int] = (8, 8),
) -> tuple[dict[int, tuple[int, int]], list[dict]]:
    """
    Build itemid → (row, col) mapping, sorted by category then itemid.

    Returns:
        itemid_to_rc: dict {itemid: (row, col)}
        grid_info: list of dicts with itemid, label, category, row, col
    """
    H, W = grid_shape
    n_cells = H * W

    dl = d_labitems_path.resolve().as_posix().replace("'", "''")
    ids_csv = ",".join(str(i) for i in itemids)

    rows = con.execute(f"""
        SELECT itemid, label, category
        FROM read_csv_auto('{dl}')
        WHERE itemid IN ({ids_csv})
        ORDER BY category, itemid
    """).fetchall()

    # Only keep up to n_cells
    rows = rows[:n_cells]

    itemid_to_rc: dict[int, tuple[int, int]] = {}
    grid_info: list[dict] = []
    for idx, (iid, label, cat) in enumerate(rows):
        r, c = idx // W, idx % W
        itemid_to_rc[int(iid)] = (r, c)
        grid_info.append({
            "index": idx, "row": r, "col": c,
            "itemid": int(iid), "label": label, "category": cat,
        })

    return itemid_to_rc, grid_info


def get_qualifying_hadm_ids(
    con, pq: str, *,
    cols: dict[str, str | None],
    max_specimens: int | None = None,
    min_specimens: int | None = None,
    top: int | None = None,
    hadm_ids: list[int] | None = None,
    window_hours: float = 24,
) -> list[int]:
    """Return list of hadm_ids to process, ranked by in-window event count."""
    if hadm_ids:
        return hadm_ids

    hadm = _col_expr(cols, "hadm_id")
    chart = _col_expr(cols, "charttime")
    admit = _col_expr(cols, "admittime")
    specimen = _specimen_expr(cols)

    # Count events *within the observation window* so ranking reflects
    # how data-rich each admission is for actual video construction.
    conditions = [
        f"{hadm} IS NOT NULL",
        f"{specimen} IS NOT NULL",
        f"{chart} IS NOT NULL",
        f"{admit} IS NOT NULL",
        f"{chart} >= {admit}",
        f"EXTRACT(EPOCH FROM ({chart} - {admit})) / 3600.0 < {window_hours}",
    ]
    having = [
        # Must have at least 1 event in-window
        "COUNT(*) >= 1",
    ]
    if max_specimens is not None:
        having.append(f"COUNT(DISTINCT {specimen}) <= {max_specimens}")
    if min_specimens is not None:
        having.append(f"COUNT(DISTINCT {specimen}) >= {min_specimens}")

    having_sql = " HAVING " + " AND ".join(having)
    order_sql = "ORDER BY COUNT(*) DESC"
    limit_sql = f"LIMIT {top}" if top else ""

    sql = f"""
        SELECT {hadm} AS hadm_id
        FROM read_parquet('{pq}')
        WHERE {" AND ".join(conditions)}
        GROUP BY {hadm}
        {having_sql}
        {order_sql}
        {limit_sql}
    """
    rows = con.execute(sql).fetchall()
    return [int(r[0]) for r in rows]


def build_videos(
    con, pq: str,
    hadm_ids: list[int],
    cols: dict[str, str | None],
    itemid_to_rc: dict[int, tuple[int, int]],
    grid_info: list[dict],
    grid_shape: tuple[int, int],
    n_bins: int,
    bin_hours: float,
    output_dir: Path,
    batch_size: int = 200,
) -> tuple[list[Path], list[dict]]:
    """
    Build T×H×W .npy videos for a list of admissions.

    Processes in batches for efficiency. Within each batch, fetches all
    relevant lab events from Parquet in one query, then splits by hadm_id.

    Returns:
        written: list of .npy file paths
        all_meta: list of metadata dicts (one per admission)
    """
    H, W = grid_shape
    n_cells = H * W
    itemids_csv = ",".join(str(i) for i in itemid_to_rc.keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    all_meta: list[dict] = []

    # Build reverse lookup: (row, col) → itemid/label for metadata
    rc_to_info = {}
    for info in grid_info:
        rc_to_info[(info["row"], info["col"])] = info

    from collections import defaultdict
    from tqdm import tqdm

    total = len(hadm_ids)
    pbar = tqdm(total=total, desc="Admissions → videos", unit="adm")

    for batch_start in range(0, total, batch_size):
        batch_ids = hadm_ids[batch_start : batch_start + batch_size]
        ids_csv = ",".join(str(i) for i in batch_ids)
        hadm_e = _col_expr(cols, "hadm_id", alias="e")
        itemid_e = _col_expr(cols, "itemid", alias="e")
        value_e = _col_expr(cols, "valuenum", alias="e")
        chart_e = _col_expr(cols, "charttime", alias="e")
        admit_e = _col_expr(cols, "admittime", alias="e")
        specimen_e = _specimen_expr(cols, alias="e")

        # Fetch events for this batch: only relevant itemids, compute time_bin
        # Also fetch specimen_id for merge tracking
        sql = f"""
        SELECT {hadm_e} AS hadm_id,
               {itemid_e} AS itemid,
               {value_e} AS valuenum,
               {chart_e} AS charttime,
               {admit_e} AS admittime,
               {specimen_e} AS specimen_id_key,
               CAST(FLOOR(
                   EXTRACT(EPOCH FROM ({chart_e} - {admit_e})) / ({bin_hours} * 3600)
               ) AS INTEGER) AS time_bin,
               EXTRACT(EPOCH FROM ({chart_e} - {admit_e})) / 3600.0 AS hours_since_admit
        FROM read_parquet('{pq}') e
        WHERE {hadm_e} IN ({ids_csv})
          AND {itemid_e} IN ({itemids_csv})
          AND {chart_e} IS NOT NULL
          AND {admit_e} IS NOT NULL
          AND {chart_e} >= {admit_e}
          AND EXTRACT(EPOCH FROM ({chart_e} - {admit_e})) / ({bin_hours} * 3600) < {n_bins}
        ORDER BY {hadm_e}, time_bin, {chart_e}
        """
        rows = con.execute(sql).fetchall()

        # Group by hadm_id
        adm_events: dict[int, list] = defaultdict(list)
        for r in rows:
            adm_events[int(r[0])].append(r)

        for hid in batch_ids:
            video = np.full((n_bins, H, W), np.nan, dtype=np.float32)
            events = adm_events.get(hid, [])

            # ── Collect per-frame tracking ────────────────────────────
            # For each time_bin: which specimens, charttimes, and itemids
            frame_specimens: dict[int, set] = defaultdict(set)     # bin → {specimen_ids}
            frame_charttimes: dict[int, list] = defaultdict(list)  # bin → [hours_since_admit]
            frame_items: dict[int, set] = defaultdict(set)         # bin → {itemids with values}

            # Within each (time_bin, itemid), last row by charttime wins (already sorted)
            last_value: dict[tuple[int, int], float] = {}  # (time_bin, itemid) → value
            for r in events:
                _, itemid, valuenum, charttime, admittime, specimen_key, time_bin, hours = r
                tbin = int(time_bin)
                if specimen_key is not None:
                    frame_specimens[tbin].add(str(specimen_key))
                if hours is not None:
                    frame_charttimes[tbin].append(float(hours))
                if valuenum is not None and not (isinstance(valuenum, float) and np.isnan(valuenum)):
                    last_value[(tbin, int(itemid))] = float(valuenum)
                    frame_items[tbin].add(int(itemid))

            for (tbin, itemid), val in last_value.items():
                if itemid in itemid_to_rc:
                    row, col = itemid_to_rc[itemid]
                    video[tbin, row, col] = val

            out_file = output_dir / f"admission_{hid}.npy"
            np.save(out_file, video)
            written.append(out_file)

            # ── Build per-frame metadata ──────────────────────────────
            frames_meta = []
            frames_with_data = 0
            total_filled_cells = 0

            for t in range(n_bins):
                frame = video[t]
                filled = int(np.count_nonzero(~np.isnan(frame)))
                total_filled_cells += filled
                has_data = filled > 0

                if has_data:
                    frames_with_data += 1

                fm = {
                    "bin": t,
                    "hour_range": f"{t * bin_hours:.0f}–{(t + 1) * bin_hours:.0f}h",
                    "n_cells_filled": filled,
                    "fill_pct": round(100.0 * filled / n_cells, 1),
                    "n_specimens_merged": len(frame_specimens.get(t, set())),
                    "specimen_ids": sorted(frame_specimens.get(t, set())),
                }
                if frame_charttimes.get(t):
                    times = frame_charttimes[t]
                    fm["charttimes_hours"] = [round(x, 2) for x in sorted(set(times))]
                    fm["time_span_minutes"] = round((max(times) - min(times)) * 60, 1)
                else:
                    fm["charttimes_hours"] = []
                    fm["time_span_minutes"] = 0.0

                if has_data:
                    vals = frame[~np.isnan(frame)]
                    fm["value_min"] = round(float(vals.min()), 3)
                    fm["value_max"] = round(float(vals.max()), 3)
                    # Which labs are present
                    fm["labs_present"] = [
                        {"itemid": info["itemid"], "label": info["label"]}
                        for (r, c), info in rc_to_info.items()
                        if not np.isnan(frame[r, c])
                    ]

                frames_meta.append(fm)

            # ── Build video-level metadata ────────────────────────────
            total_cells = n_bins * n_cells
            all_specimens = set()
            for s in frame_specimens.values():
                all_specimens |= s

            meta = {
                "hadm_id": hid,
                "video_shape": [n_bins, H, W],
                "window_hours": float(n_bins * bin_hours),
                "bin_hours": bin_hours,
                "n_bins": n_bins,
                "video_summary": {
                    "total_cells": total_cells,
                    "filled_cells": total_filled_cells,
                    "fill_pct": round(100.0 * total_filled_cells / total_cells, 2),
                    "frames_with_data": frames_with_data,
                    "frames_empty": n_bins - frames_with_data,
                    "temporal_density_pct": round(100.0 * frames_with_data / n_bins, 1),
                    "total_specimens_merged": len(all_specimens),
                    "specimen_ids": sorted(all_specimens),
                },
                "frames": frames_meta,
            }
            all_meta.append(meta)
            pbar.update(1)

    pbar.close()
    return written, all_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert admissions to T×H×W video tensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet", type=Path, default=None,
                        help="Path to lab_events_with_adm.parquet")
    parser.add_argument("--d-labitems", type=Path, default=None,
                        help="Path to d_labitems.csv")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory (default: lab_videos/)")
    # Grid
    parser.add_argument("--grid-h", type=int, default=8, help="Grid height (default 8)")
    parser.add_argument("--grid-w", type=int, default=8, help="Grid width (default 8)")
    parser.add_argument("--n-labs", type=int, default=None,
                        help="Number of labs to select (default: H×W)")
    # Time
    parser.add_argument("--window-hours", type=float, default=24,
                        help="Observation window in hours (default 24)")
    parser.add_argument("--bin-hours", type=float, default=1,
                        help="Time bin size in hours (default 1)")
    # Admission selection
    parser.add_argument("--max-specimens", type=int, default=None,
                        help="Only admissions with ≤ N specimens")
    parser.add_argument("--min-specimens", type=int, default=None,
                        help="Only admissions with ≥ N specimens")
    parser.add_argument("--top", type=int, default=None,
                        help="Process only top N admissions by event count")
    parser.add_argument("--hadm-ids", type=int, nargs="+", default=None,
                        help="Process specific hadm_ids")
    # Processing
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Admissions per DuckDB batch (default 200)")

    args = parser.parse_args()

    pq_path = args.parquet or DEFAULT_LAB_WITH_ADM_PARQUET
    dl_path = args.d_labitems or DEFAULT_D_LABITEMS
    grid_shape = (args.grid_h, args.grid_w)
    n_labs = args.n_labs or (grid_shape[0] * grid_shape[1])
    n_bins = int(args.window_hours / args.bin_hours)
    outdir = args.outdir or (Path(__file__).resolve().parent.parent / "lab_videos")

    if not pq_path.exists():
        print(f"Parquet not found: {pq_path}")
        return
    if not dl_path.exists():
        print(f"d_labitems not found: {dl_path}")
        return

    import duckdb
    pq = pq_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    try:
        cols = resolve_schema_columns(con, pq)
    except ValueError as e:
        print(f"Parquet schema mismatch: {e}")
        con.close()
        return

    print("Resolved parquet columns:")
    print(f"  hadm_id     ← {cols['hadm_id']}")
    print(f"  itemid      ← {cols['itemid']}")
    print(f"  valuenum    ← {cols['valuenum']}")
    print(f"  charttime   ← {cols['charttime']}")
    print(f"  admittime   ← {cols['admittime']}")
    if cols["specimen_id"]:
        print(f"  specimen_id ← {cols['specimen_id']}")
    else:
        print("  specimen_id ← <missing; using charttime as surrogate>")

    # ── Step 1: Select labs ──────────────────────────────────────────
    print(f"Selecting top {n_labs} labs (excluding artifact/hold codes)...")
    top_itemids = select_top_labs(
        con, pq, cols, n_labs=n_labs, max_specimens=args.max_specimens,
    )
    print(f"  Selected {len(top_itemids)} lab types")

    # ── Step 2: Build grid mapping ───────────────────────────────────
    print(f"Building {grid_shape[0]}×{grid_shape[1]} grid mapping (by category)...")
    itemid_to_rc, grid_info = build_grid_mapping(
        con, top_itemids, dl_path, grid_shape=grid_shape,
    )
    print(f"  Grid has {len(itemid_to_rc)} labs mapped to {grid_shape[0]}×{grid_shape[1]} cells")

    # Save grid layout
    outdir.mkdir(parents=True, exist_ok=True)
    layout_path = outdir / "grid_layout.json"
    with open(layout_path, "w") as f:
        json.dump(grid_info, f, indent=2)
    print(f"  Grid layout saved to {layout_path}")

    # Print grid layout summary
    print(f"\n  Grid layout ({grid_shape[0]}×{grid_shape[1]}):")
    for info in grid_info:
        print(f"    ({info['row']},{info['col']})  itemid={info['itemid']}  "
              f"{info['category']:<15} {info['label']}")

    # ── Step 3: Select admissions ────────────────────────────────────
    print(f"\nSelecting admissions...")
    hadm_ids = get_qualifying_hadm_ids(
        con, pq,
        cols=cols,
        max_specimens=args.max_specimens,
        min_specimens=args.min_specimens,
        top=args.top,
        hadm_ids=args.hadm_ids,
        window_hours=args.window_hours,
    )
    print(f"  {len(hadm_ids):,} admissions to process")

    # ── Step 4: Build videos ─────────────────────────────────────────
    print(f"\nBuilding videos: T={n_bins}, H={grid_shape[0]}, W={grid_shape[1]}")
    print(f"  Window: {args.window_hours}h, bin: {args.bin_hours}h")
    print(f"  Output: {outdir}/")

    written, all_meta = build_videos(
        con, pq, hadm_ids,
        cols=cols,
        itemid_to_rc=itemid_to_rc,
        grid_info=grid_info,
        grid_shape=grid_shape,
        n_bins=n_bins,
        bin_hours=args.bin_hours,
        output_dir=outdir,
        batch_size=args.batch_size,
    )

    con.close()

    # ── Save metadata ────────────────────────────────────────────────
    meta_path = outdir / "video_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Done. {len(written):,} videos saved to {outdir}/")
    print(f"  Shape per video: ({n_bins}, {grid_shape[0]}, {grid_shape[1]})")
    print(f"  Format: float32, NaN = missing")

    # Aggregate stats across all videos
    if all_meta:
        fills = [m["video_summary"]["fill_pct"] for m in all_meta]
        td = [m["video_summary"]["temporal_density_pct"] for m in all_meta]
        n_spec = [m["video_summary"]["total_specimens_merged"] for m in all_meta]
        fwd = [m["video_summary"]["frames_with_data"] for m in all_meta]
        print(f"\n  Aggregate stats across {len(all_meta)} videos:")
        print(f"    Overall fill %:       mean={np.mean(fills):.1f}%  median={np.median(fills):.1f}%")
        print(f"    Temporal density:      mean={np.mean(td):.1f}%  median={np.median(td):.1f}%  "
              f"(frames with data / {n_bins})")
        print(f"    Frames with data:      mean={np.mean(fwd):.1f}  median={np.median(fwd):.0f}")
        print(f"    Specimens merged:      mean={np.mean(n_spec):.1f}  median={np.median(n_spec):.0f}")

    # Detailed per-video printout for small runs
    if len(all_meta) <= 20:
        print(f"\n{'='*70}")
        print("Detailed per-video report")
        print(f"{'='*70}")
        for meta in all_meta:
            hid = meta["hadm_id"]
            vs = meta["video_summary"]
            print(f"\n  ┌─ admission_{hid}")
            print(f"  │  fill={vs['fill_pct']:.1f}%  "
                  f"frames_with_data={vs['frames_with_data']}/{n_bins}  "
                  f"specimens_merged={vs['total_specimens_merged']}")
            print(f"  │")
            for fm in meta["frames"]:
                t = fm["bin"]
                filled = fm["n_cells_filled"]
                if filled > 0:
                    n_spec = fm["n_specimens_merged"]
                    times_str = ", ".join(f"{h:.1f}h" for h in fm["charttimes_hours"][:5])
                    if len(fm["charttimes_hours"]) > 5:
                        times_str += f" (+{len(fm['charttimes_hours'])-5} more)"
                    labs_short = [l["label"] for l in fm.get("labs_present", [])[:8]]
                    labs_str = ", ".join(labs_short)
                    if len(fm.get("labs_present", [])) > 8:
                        labs_str += f" (+{len(fm['labs_present'])-8} more)"
                    span_str = f"  span={fm['time_span_minutes']:.0f}min" if fm['time_span_minutes'] > 0 else ""
                    print(f"  │  bin {t:2d} [{fm['hour_range']:>7}]: "
                          f"{filled:2d}/64 cells ({fm['fill_pct']:5.1f}%)  "
                          f"specimens={n_spec}{span_str}")
                    print(f"  │         charttimes: [{times_str}]")
                    print(f"  │         labs: {labs_str}")
                else:
                    print(f"  │  bin {t:2d} [{fm['hour_range']:>7}]: empty")
            print(f"  └─")


if __name__ == "__main__":
    main()
