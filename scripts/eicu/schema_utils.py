from __future__ import annotations


def _quote_ident(col: str) -> str:
    """Safely quote a DuckDB identifier."""
    return '"' + col.replace('"', '""') + '"'


def _col_expr(cols: dict[str, str | None], key: str, alias: str | None = None) -> str:
    """Return a quoted column expression, optionally with table alias."""
    col = cols.get(key)
    if not col:
        raise ValueError(f"Missing required mapped column: {key}")
    quoted = _quote_ident(col)
    return f"{alias}.{quoted}" if alias else quoted


def _specimen_expr(
    cols: dict[str, str | None],
    alias: str | None = None,
    *,
    prefer_offset: bool = False,
) -> str:
    """
    Return the collection key expression used for grouping lab rows.

    eICU extracts often do not contain a native `specimen_id`. When that
    happens we fall back to either `labresultoffset` (relative minutes from
    admission) or `charttime`, depending on the caller's preference.
    """
    prefix = f"{alias}." if alias else ""
    if cols.get("specimen_id"):
        return f"CAST({prefix}{_quote_ident(cols['specimen_id'])} AS VARCHAR)"
    if prefer_offset and cols.get("labresultoffset"):
        return f"CAST({prefix}{_quote_ident(cols['labresultoffset'])} AS VARCHAR)"
    return f"CAST({prefix}{_quote_ident(cols['charttime'])} AS VARCHAR)"


def resolve_schema_columns(con, pq: str) -> dict[str, str | None]:
    """Map canonical names to the actual parquet columns."""
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pq}')").fetchall()
    available = {str(row[0]).lower(): str(row[0]) for row in schema_rows}

    def pick(candidates: list[str], required: bool = True) -> str | None:
        for candidate in candidates:
            if candidate.lower() in available:
                return available[candidate.lower()]
        if required:
            raise ValueError(
                f"Could not find any of {candidates} in parquet columns: "
                f"{sorted(available.values())}"
            )
        return None

    return {
        "subject_id": pick(["subject_id", "uniquepid"], required=False),
        "hadm_id": pick(["hadm_id", "patientunitstayid"]),
        "itemid": pick(["itemid", "lab_code"]),
        "valuenum": pick(["valuenum", "labresult"]),
        "charttime": pick(["charttime"]),
        "admittime": pick(["admittime"]),
        "specimen_id": pick(["specimen_id"], required=False),
        "labresultoffset": pick(["labresultoffset"], required=False),
        "label": pick(["label", "labname"], required=False),
    }
