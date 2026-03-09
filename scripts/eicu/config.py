"""Shared paths and defaults for the Latte pipeline. All scripts can import from here."""
from __future__ import annotations

from pathlib import Path

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent.parent

# eICU data directory
DEFAULT_EICU = ROOT / "eICU"

# Lab events
DEFAULT_PARQUET = DEFAULT_EICU / "labevents.parquet"
DEFAULT_LAB_WITH_ADM_PARQUET = DEFAULT_EICU / "lab_events_with_adm.parquet"

# Admissions
DEFAULT_ADMISSIONS_CSV = DEFAULT_EICU / "admissions.csv"
DEFAULT_ADMISSIONS_PARQUET = DEFAULT_EICU / "admissions.parquet"

# Vocab and outputs
DEFAULT_D_LABITEMS = DEFAULT_EICU / "vocab_labs.csv"
DEFAULT_LAB_IMAGES = ROOT / "lab_images"
