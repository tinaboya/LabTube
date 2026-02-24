"""
Transform each specimen (or charttime group) into a 2D grid image of (lab_type, lab_value).
Reads from labevents Parquet (or CSV in chunks), fills H×W grid, saves as .npy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_grid import build_itemid_to_grid_index, get_top_itemids_from_parquet


def normalize_value(
    valuenum: float,
    ref_low: Optional[float],
    ref_high: Optional[float],
    *,
    missing_fill: float = 0.0,
    clip: bool = True,
) -> float:
    """
    Normalize a lab value to [0, 1] using reference range when available.
    """
    if np.isnan(valuenum) or pd.isna(valuenum):
        return missing_fill
    if ref_low is not None and ref_high is not None and ref_high > ref_low:
        x = (float(valuenum) - ref_low) / (ref_high - ref_low)
        if clip:
            x = max(0.0, min(1.0, x))
        return float(x)
    # No ref range: use value as-is but clip to [0,1] for safety (or pass through)
    if clip:
        return max(0.0, min(1.0, float(valuenum)))
    return float(valuenum)


def specimen_events_to_grid(
    events: pd.DataFrame,
    itemid_to_rc: dict[int, tuple[int, int]],
    grid_shape: tuple[int, int],
    *,
    value_col: str = "valuenum",
    ref_low_col: str = "ref_range_lower",
    ref_high_col: str = "ref_range_upper",
    missing_fill: float = np.nan,
    use_raw_values: bool = True,
) -> np.ndarray:
    """
    Fill a single H×W grid from a DataFrame of lab events (one specimen).

    Args:
        events: DataFrame with columns itemid, value_col, ref_low_col, ref_high_col
        itemid_to_rc: mapping itemid -> (row, col)
        grid_shape: (H, W)
        value_col, ref_low_col, ref_high_col: column names
        missing_fill: value for cells with no or invalid data (default nan when use_raw_values)
        use_raw_values: if True, store valuenum as-is; if False, normalize to [0,1] by ref range

    Returns:
        img: (H, W) float array; raw lab values or [0,1] when normalized; missing = missing_fill
    """
    H, W = grid_shape
    img = np.full((H, W), missing_fill, dtype=np.float32)
    for _, row in events.iterrows():
        iid = row.get("itemid")
        if iid is None or pd.isna(iid) or int(iid) not in itemid_to_rc:
            continue
        r, c = itemid_to_rc[int(iid)]
        v = row.get(value_col)
        ref_lo = row.get(ref_low_col)
        ref_hi = row.get(ref_high_col)
        if pd.notna(v) and (isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "").replace("-", "").isdigit())):
            try:
                vf = float(v)
            except (ValueError, TypeError):
                vf = np.nan
        else:
            vf = np.nan
        if use_raw_values:
            img[r, c] = vf if not (np.isnan(vf) or pd.isna(vf)) else missing_fill
        else:
            img[r, c] = normalize_value(vf, ref_lo, ref_hi, missing_fill=missing_fill)
    return img


def iter_specimens_from_parquet_chunked(
    parquet_path: str | Path,
    *,
    group_key: str = "specimen_id",
    limit_specimens: Optional[int] = None,
):
    """
    Iterate specimens from Parquet one at a time (per-specimen query) to keep memory low.
    """
    import duckdb

    pq = Path(parquet_path).as_posix()
    con = duckdb.connect()
    try:
        id_sql = f"""
        SELECT DISTINCT specimen_id FROM read_parquet('{pq}')
        WHERE specimen_id IS NOT NULL
        ORDER BY specimen_id
        """
        if limit_specimens is not None:
            id_sql += f" LIMIT {limit_specimens}"
        specimen_ids = [row[0] for row in con.execute(id_sql).fetchall()]
        for spec_id in specimen_ids:
            events = con.execute(
                f"""
                SELECT specimen_id, subject_id, charttime, itemid, valuenum, value,
                       ref_range_lower, ref_range_upper
                FROM read_parquet('{pq}')
                WHERE specimen_id = ?
                """,
                [spec_id],
            ).fetchdf()
            yield spec_id, events
    finally:
        con.close()


def run_specimen_to_image(
    parquet_path: str | Path,
    d_labitems_path: str | Path,
    output_dir: str | Path,
    *,
    grid_shape: tuple[int, int] = (16, 16),
    use_top_k_itemids: bool = True,
    top_k: Optional[int] = None,
    limit_specimens: Optional[int] = None,
    missing_fill: float = np.nan,
    use_raw_values: bool = True,
    sort_by_category: bool = True,
) -> list[Path]:
    """
    For each specimen in labevents Parquet, build one (H×W) image and save as .npy.

    Args:
        parquet_path: labevents.parquet
        d_labitems_path: d_labitems.csv
        output_dir: directory for specimen_<id>.npy files
        grid_shape: (H, W)
        use_top_k_itemids: if True, use top (H*W) itemids by count from Parquet
        top_k: if set, use this many itemids (default H*W)
        limit_specimens: if set, only process this many specimens (for testing)
        missing_fill: fill value for missing cells (default nan when use_raw_values)
        use_raw_values: if True, store raw valuenum in grid; if False, normalize to [0,1]
        sort_by_category: if True, order grid by d_labitems.category so related labs are adjacent

    Returns:
        List of paths to written .npy files.
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cells = grid_shape[0] * grid_shape[1]
    if use_top_k_itemids and parquet_path.exists():
        itemids = get_top_itemids_from_parquet(parquet_path, top_k or n_cells)
    else:
        itemids = None
    itemid_to_rc, _, _ = build_itemid_to_grid_index(
        d_labitems_path, grid_shape=grid_shape, itemids=itemids, sort_by_category=sort_by_category
    )

    written: list[Path] = []
    specimen_iter = iter_specimens_from_parquet_chunked(
        parquet_path, limit_specimens=limit_specimens
    )
    for spec_id, events in tqdm(specimen_iter, desc="Specimens", unit="spec"):
        img = specimen_events_to_grid(
            events, itemid_to_rc, grid_shape, missing_fill=missing_fill, use_raw_values=use_raw_values
        )
        out_file = output_dir / f"specimen_{spec_id}.npy"
        np.save(out_file, img)
        written.append(out_file)
    return written
