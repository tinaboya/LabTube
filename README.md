<p align="center">
  <img src="labtube.png" alt="Latte logo" width="180"/>
</p>

<h1 align="center">Latte</h1>

<p align="center">
  <em>VideoMAE-style pre-training on ICU lab time series from MIMIC-IV and eICU</em>
</p>

---

## Overview

**Latte** converts hospital admissions into video tensors for self-supervised pre-training with VideoMAE-style masked autoencoders.

Each admission becomes a video of shape **(T, H, W)**:

| Dimension | Meaning | Value |
|-----------|---------|-------|
| **T** — tube (time) | One frame per 1-hour bin over the first 24 h of admission | 24 frames |
| **H × W** — spatial grid | Each cell = one lab type (top 64 most frequent labs) | 8 × 8 |
| **Pixel value** | Raw lab measurement (float32); NaN = not measured | — |

The model learns temporal and cross-lab patterns purely from the structure of ICU data, without any task-specific supervision.

---

## Pipeline

```
MIMIC-IV / eICU raw CSVs
        │
        ▼
   csv_to_parquet.py        — convert raw CSVs to Parquet
   lab_with_admissions.py   — join lab events with admission records
        │
        ▼
   build_specimen_npy.py    — build per-specimen image arrays
   admission_to_video.py    — stack specimens into (T, H, W) video tensors
        │
        ▼
   normalize_videos.py      — z-score normalization per lab channel
        │
        ▼
   lab_videos_normalized/   — ready for VideoMAE training
```

---

## Project Structure

```
Latte/
├── scripts/                  # All pipeline Python scripts
│   ├── config.py             # Paths and global settings
│   ├── run_pipeline.py       # End-to-end pipeline runner
│   ├── admission_to_video.py # Core video construction
│   ├── normalize_videos.py   # Normalization
│   └── ...
├── docs/                     # Reports and analysis
│   ├── DATA_ANALYSIS_REPORT.md
│   ├── VIDEO_CONSTRUCTION_REPORT.md
│   ├── TRAINING_PLAN.md
│   └── figures/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python ≥ 3.10

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

**Data access:** This project requires access to [MIMIC-IV](https://physionet.org/content/mimiciv/) and [eICU-CRD](https://physionet.org/content/eicu-crd/) via PhysioNet. Data files are **not** included in this repository.

---

## Running the Pipeline

```bash
# Run the full pipeline end-to-end
python scripts/run_pipeline.py

# Or run individual steps
python scripts/csv_to_parquet.py
python scripts/lab_with_admissions.py
python scripts/admission_to_video.py
python scripts/normalize_videos.py
```

---

## Key Design Choices

- **Time = tube:** The temporal axis maps to the video tube dimension, so VideoMAE's spatiotemporal masking operates across both time and lab type simultaneously.
- **Lab type = spatial pixel:** Each position in the 8×8 grid is a fixed lab type, enabling the model to learn cross-lab correlations as spatial structure.
- **First 24 hours:** Each admission is represented by its first 24 h, following the eICU benchmark (1 h bins → 24 frames).
- **Sparse by design:** Grid fill is ~4% on average — the model learns to handle structured clinical missingness.

---

## Datasets

| Dataset | Description |
|---------|-------------|
| [MIMIC-IV](https://physionet.org/content/mimiciv/) | ~450K ICU admissions, Beth Israel Deaconess Medical Center |
| [eICU-CRD](https://physionet.org/content/eicu-crd/) | Multi-center ICU database |

> **Data use:** Access requires completing PhysioNet credentialing and signing the respective data use agreements. Raw data files must never be committed to version control.

---

## Documentation

- [Data Analysis Report](docs/DATA_ANALYSIS_REPORT.md)
- [Video Construction Report](docs/VIDEO_CONSTRUCTION_REPORT.md)
- [Training Plan](docs/TRAINING_PLAN.md)
- [Subset Analysis](docs/SUBSET_ANALYSIS.md)
- [Google Cloud Setup](docs/RUN_ON_GOOGLE_CLOUD.md)
