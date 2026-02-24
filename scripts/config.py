"""Shared paths and defaults for the Latte pipeline. All scripts can import from here."""
from __future__ import annotations

from pathlib import Path

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent

# MIMIC-IV data directory
DEFAULT_MIMIC4 = ROOT / "MIMIC4"

# Lab events
DEFAULT_PARQUET = DEFAULT_MIMIC4 / "labevents.parquet"
DEFAULT_LAB_WITH_ADM_PARQUET = DEFAULT_MIMIC4 / "lab_events_with_adm.parquet"

# Admissions
DEFAULT_ADMISSIONS_CSV = DEFAULT_MIMIC4 / "admissions.csv"
DEFAULT_ADMISSIONS_PARQUET = DEFAULT_MIMIC4 / "admissions.parquet"

# Vocab and outputs
DEFAULT_D_LABITEMS = DEFAULT_MIMIC4 / "d_labitems.csv"
DEFAULT_LAB_IMAGES = ROOT / "lab_images"
