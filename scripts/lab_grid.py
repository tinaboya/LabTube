"""
Lab grid: map itemid → (row, col) for a fixed H×W grid of lab types.
Loads d_labitems and builds index; supports top-K by frequency from Parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_d_labitems(path: str | Path) -> pd.DataFrame:
    """Load MIMIC-IV d_labitems.csv."""
    return pd.read_csv(path, dtype={"itemid": "Int64"})


def build_itemid_to_grid_index(
    d_labitems_path: str | Path,
    grid_shape: tuple[int, int] = (16, 16),
    itemids: Optional[list[int]] = None,
    sort_by_category: bool = True,
) -> tuple[dict[int, tuple[int, int]], list[int], pd.DataFrame]:
    """
    Build mapping from itemid to (row, col) for a fixed grid.
    Uses d_labitems.csv (itemid, label, fluid, category). When sort_by_category is True,
    labs are ordered by category then itemid so that related tests sit in contiguous regions.

    Args:
        d_labitems_path: Path to d_labitems.csv
        grid_shape: (H, W) e.g. (16, 16) for 256 lab types
        itemids: If provided, restrict to these itemids (e.g. top-K by frequency); order is then by (category, itemid) when sort_by_category.
        sort_by_category: If True, order grid by d_labitems.category then itemid so same-category labs are adjacent.

    Returns:
        itemid_to_rc: dict itemid -> (row, col)
        index_to_itemid: list of length H*W, index -> itemid
        df_items: subset of d_labitems for the chosen itemids
    """
    H, W = grid_shape
    n_cells = H * W
    df = load_d_labitems(d_labitems_path)
    df = df.dropna(subset=["itemid"])

    if itemids is not None:
        # Restrict to these itemids; then sort by category (and itemid) so related labs are adjacent
        subset = df[df["itemid"].isin(itemids)]
        if sort_by_category and "category" in subset.columns:
            chosen = subset.sort_values(["category", "itemid"], na_position="last").head(n_cells)
        else:
            # Preserve original order (e.g. frequency order from top-K)
            chosen = subset.set_index("itemid").loc[[i for i in itemids if i in subset["itemid"].values]].reset_index()
            chosen = chosen.head(n_cells)
        itemids = chosen["itemid"].tolist()
    else:
        # First H*W from d_labitems, ordered by category then itemid (or itemid only)
        if sort_by_category and "category" in df.columns:
            chosen = df.sort_values(["category", "itemid"], na_position="last").head(n_cells)
        else:
            chosen = df.sort_values("itemid").head(n_cells)
        itemids = chosen["itemid"].tolist()

    index_to_itemid = (itemids + [0] * n_cells)[:n_cells]  # pad if short
    itemid_to_rc = {}
    for idx, iid in enumerate(index_to_itemid):
        if iid != 0:
            row, col = idx // W, idx % W
            itemid_to_rc[int(iid)] = (row, col)

    return itemid_to_rc, index_to_itemid, chosen


def get_top_itemids_from_parquet(
    parquet_path: str | Path,
    top_k: int,
    *,
    duckdb_memory: str = "2GB",
) -> list[int]:
    """
    Get the top-K itemids by event count from labevents.parquet (for building grid from most common labs).
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{duckdb_memory}'")
        r = con.execute(
            f"""
            SELECT itemid, COUNT(*) AS cnt
            FROM read_parquet('{Path(parquet_path).as_posix()}')
            WHERE itemid IS NOT NULL
            GROUP BY itemid
            ORDER BY cnt DESC
            LIMIT {top_k}
            """
        ).fetchall()
        return [int(row[0]) for row in r]
    finally:
        con.close()
