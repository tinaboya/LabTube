# Training plan: VideoMAE-style ICU lab time series

This document captures the steps from the training plan PDF and maps them to the Latte pipeline and codebase.

---

## 1. Recommendations from the plan (PDF summary)

### 1.1 Filter by lab data completeness

- **Phase 1 (proof of concept):** Filter to admissions with **≥60–70% lab data completeness** to validate that a VideoMAE-style model can learn from ICU time series.
- **Phase 2 (robustness):** Gradually include more sparse data to test handling of missingness.
- **Rationale:** Random masking in VideoMAE (90–95%) is different from structured, informative missingness in ICU data; starting with more complete data isolates whether the architecture works before tackling missingness.

### 1.2 Segment into fixed-length time windows

- **Do not** treat the entire stay as one variable-length sequence.
- **Recommended:** Fixed time windows (e.g. **24 h** or **48 h**) with **regular time bins**.
- **Benchmarks:**
  - MIMIC-III: 48-hour windows, **2-hour bins** → **24 time steps**.
  - eICU: 24-hour windows, **1-hour bins** → **24 time steps**.
- **Why:** VideoMAE typically uses 16–32 frames; clinical tasks often focus on specific windows (e.g. first 24 h for early prediction). Variable-length stays (1–30+ days) would create inconsistent sequence lengths.

### 1.3 Treat each window as one “video” — data layout for VideoMAE

Target layout for each input to the VideoMAE model:

- **Tube (time)** = the frame dimension of the “video”. Each frame index = one time bin (e.g. hour 0, hour 1, …). So **time is the tube** of the video.
- **Each frame** = one 2D grid where:
  - **One dimension (spatial)** = lab type / name of the lab event (e.g. row and column index the set of lab types, so each “pixel” position = one lab).
  - **Pixel value** = the **real lab value** (the actual measurement for that lab at that time). Optionally normalized (e.g. [0,1] by ref range) for training; the important part is that the value encodes the lab result, not a placeholder.
- So one sample = **(T, H, W)**:
  - **T** = number of time steps (tube length).
  - **(H, W)** = grid of lab types (e.g. 16×16); cell (h, w) = one lab; value at (t, h, w) = lab value at time t for that lab (or 0/missing).

Summary: **time = tube**, **lab name/type = spatial dimensions (H×W)**, **pixel = real lab value**.

### 1.3b One video = one admission (start with first 24 hours)

- **One video** = one admission (or one time window of one admission). Start small: e.g. **first 24 hours** of each admission = one video.
- **Specimens** = individual lab draws at **one timestamp**. Many specimens belong to one admission (many draws over time). So specimens do **not** map 1:1 to frames.
- **Frames** = **time bins** within that window. For “first 24 h” with 1 h bins we get **T = 24 frames**. Each frame = one time bin (e.g. hour 0–1, 1–2, …).
- **Building a frame:** For each time bin, take all lab events (specimens) whose `charttime` falls in that bin; fill one (H×W) grid (lab type → pixel, value = lab value). If multiple values per lab in the same bin, aggregate (e.g. last or mean per itemid).
- So: **admission → first 24 h → 24 time bins → 24 frames → stack to (T, H, W)**. One video per admission (for the chosen window).

### 1.4 Handling variable-length stays

Three complementary strategies:

| Strategy | Description |
|----------|-------------|
| **Fixed-window sliding** | Extract multiple 24 h (or 48 h) windows per admission; each window is one training sample. Handles any stay length. |
| **Padding + attention masking** | Pad shorter sequences to max length; mask padded positions in the transformer. |
| **Pooling** | For long stays, use sliding windows and aggregate representations (e.g. mean/max pooling) across windows for patient-level predictions. |

**Recommended start:** **24 time steps** (1-hour bins over **24 hours**), following the eICU benchmark—enough temporal resolution and within typical VideoMAE frame capacity.

### 1.5 Cross-dataset validation (eICU vs MIMIC-IV)

If eICU uses **1 h bins** (24 h → 24 frames) and MIMIC uses **2 h bins** (48 h → 24 frames), “frame 5” means different things: in eICU it’s hours 5–6 from admission; in MIMIC it’s hours 10–12. Training on one and testing on the other then mixes different temporal semantics and makes cross-dataset results hard to interpret.

**Recommendation: use the same temporal schema on both.**

- **Option A (recommended): Single schema for both**  
  Pick **one** window length and **one** bin duration; apply it to **both** eICU and MIMIC-IV:
  - **24 h window, 1 h bins → 24 frames.**  
    For MIMIC-IV: use only the first 24 h from admission and bin into 1 h (same as eICU). Frame index = “hours since admission” in both. Train on MIMIC, test on eICU (or vice versa) with identical frame semantics.
  - **48 h window, 2 h bins → 24 frames.**  
    For eICU: use first 48 h and aggregate into 2 h bins (e.g. mean or last value per 2 h). Then both datasets are 24 frames over 48 h. Downside: eICU may have shorter stays, so 48 h windows may be noisier or require more padding.

- **Option B: Harmonize to the “finer” schema**  
  Use **24 h, 1 h bins** everywhere. For MIMIC-III/IV (often reported as 48 h / 2 h), either:
  - Use only the first 24 h and 1 h bins (so you compare 24 h of care on both), or  
  - Use 48 h of MIMIC but with **1 h bins → 48 frames**, then downsample in the model (e.g. pool or stride) to 24 if you need to match eICU’s 24-frame setup.  

  That way “frame t” = “hour t after admission” in both datasets.

- **Option C: Report both in-domain and cross-domain**  
  Keep dataset-specific binning for **in-domain** benchmarks (e.g. MIMIC 48 h/2 h, eICU 24 h/1 h). For **cross-domain** (train on one, test on the other), always preprocess the **test** set with the **training** set’s schema: e.g. train on MIMIC with 24 h/1 h, then when testing on eICU also use 24 h/1 h so frame indices align. Document that cross-dataset numbers use the training dataset’s window and bin definition.

**Practical takeaway:** For cross-validating eICU and MIMIC-IV, define **one** canonical schema (e.g. **24 h from admission, 1 h bins, 24 frames**) and apply it to both. Then train-on-MIMIC/test-on-eICU and train-on-eICU/test-on-MIMIC are comparable and interpretable.

**Sampling frequency differs anyway.** Even with the same binning, the *frequency of lab draws* is not the same across datasets: MIMIC-IV (single center) and eICU (multi-center) have different ordering practices, so how many values fall in each 1 h bin (and which lab types appear when) will differ. Agreeing on one temporal schema only fixes *what each frame index means* (e.g. “hour 5–6”); it does **not** make the two datasets have the same sampling density or missingness pattern. Cross-validation then tests generalization across both (a) different populations/sites and (b) different real-world sampling—which is a harder, more realistic evaluation, as long as we’re explicit that we’re not controlling for sampling frequency, only for temporal alignment.

---

## 2. Design-level issues and recommendations for cross-dataset validation (eICU vs MIMIC-IV)

Aligning time windows and bin sizes across datasets is **necessary and correct, but not sufficient** for interpretable cross-dataset validation. The following design issues, why they matter, and how to address them should be reflected in the pipeline and in reporting.

### 2.1 Temporal alignment fixes frame semantics, not the data-generating process

**Issue:** Aligning both datasets to the same temporal schema (e.g. 24 h window, 1 h bins) ensures a given frame index has the same meaning (e.g. “hours 5–6 after admission”), but it does **not** align how data are generated within each frame. Models may exploit **sampling density and missingness patterns** (how often labs are ordered, which labs appear early) as shortcut features. These patterns differ substantially between MIMIC-IV (single center) and eICU (multi-center), and across hospitals within eICU.

**Recommendation:**

- **Explicitly model missingness and sampling:** Include per-bin (or per-frame) features such as **observation mask**, **count of observations in bin**, and optionally **time_since_last_observation (Δt)**.
- **When reporting cross-dataset results, distinguish:**
  - Generalization of **lab values**
  - Generalization across **sampling and missingness mechanisms**

This ensures the evaluation acknowledges domain shift rather than silently exploiting it.

### 2.2 Harmonize the definition of “time zero”

**Issue:** The plan assumes a shared reference point (e.g. admission time), but in practice MIMIC-IV and eICU differ in how admission-related timestamps are defined and populated. **Hospital admission time**, **ICU admission time**, and **ED registration time** are not interchangeable. Even with identical bins, frames may be systematically misaligned across datasets.

**Recommendation:**

- **Define a canonical time zero explicitly:**
  - For **ICU-focused tasks:** ICU admission time.
  - For **hospital-level tasks:** hospital admission time, with a documented eICU equivalent.
- **Validate alignment** by plotting and comparing `event_time − time_zero` distributions in both datasets.

### 2.3 Observation and label window alignment

**Issue:** Choosing 24 h vs 48 h is not only an engineering decision; it changes the **task definition**. Cross-dataset training/testing becomes invalid if observation windows or label/prediction horizons differ—this can introduce label shift or unintended information leakage.

**Recommendation:**

- **Define each task using an explicit three-part structure:**
  - **Observation window** (e.g. first 24 h from time zero)
  - **Optional gap** (e.g. 0 h)
  - **Label / prediction horizon** (e.g. outcome at 48 h, or discharge)
- **Enforce identical definitions** for cross-dataset evaluation.
- If definitions differ, treat them as **separate tasks** rather than as cross-validation.

### 2.4 Feature harmonization beyond temporal alignment

**Issue:** Even with aligned bins, labs may not be directly comparable due to **unit differences**, **inconsistent naming or item mappings**, and **center-specific measurement practices**. Poor cross-dataset performance then becomes hard to interpret.

**Recommendation:**

- **Add a feature harmonization step:**
  - Normalize **units** where possible.
  - Map labs to a **shared high-confidence concept set** (conservative intersection across datasets).
  - Apply **robust scaling** (e.g. quantile-based) using **training-set statistics** only.
- **Document** which labs are included and which are excluded (and why).

### 2.5 eICU’s multi-center structure in evaluation

**Issue:** Cross-dataset validation conflates two types of domain shift: **dataset-level shift** (eICU vs MIMIC) and **site-level shift** (between hospitals within eICU).

**Recommendation:**

- **In addition to cross-dataset experiments**, perform:
  - **Hospital-stratified splits** or **leave-one-hospital-out** evaluation within eICU.
- This helps **disentangle site effects from dataset effects**.

### 2.6 Temporal harmonization from raw event timestamps

**Issue:** If one dataset has already been aggregated (e.g. 2 h bins), it is **not** possible to reliably recover finer temporal resolution. “Upsampling” coarse bins introduces artificial data and misleading temporal structure.

**Recommendation:**

- **Enforce that all temporal binning is performed from raw timestamped events**, not from pre-aggregated tensors.
- **State this explicitly** in the training plan and in any preprocessing code.

---

### Summary: what robust cross-dataset validation requires

| Requirement | Purpose |
|-------------|---------|
| Same time bins (temporal alignment) | Frame-level semantic consistency |
| Explicit missingness/sampling features | Avoid shortcut learning; interpret generalization |
| Canonical time zero (ICU vs hospital, documented) | Align “hour 0” across datasets |
| Identical observation + label window definitions | Valid task comparison; no label shift |
| Feature harmonization (units, concept mapping, train-set scaling) | Comparable lab semantics and scale |
| Site-aware evaluation within eICU | Separate site vs dataset effects |
| Binning from raw timestamps only | No artificial upsampling or fake temporal structure |

Addressing these points makes cross-dataset results **interpretable and scientifically defensible**, rather than merely comparable in shape.

**Implementation implications:** Time-window segmentation (Step 2 in §4) must use a **documented time zero** (e.g. ICU admit or hospital admit) and **raw event timestamps** only. The admission-to-video step (Step 3) should output **observation masks** and optionally **count / Δt** per bin. A separate **feature harmonization** step (shared concept set, units, train-set scaling) and **task card** (observation window, gap, label horizon) should be added to the pipeline and documented. Site-aware evaluation (e.g. leave-one-hospital-out for eICU) belongs in the evaluation/training code, not in Latte preprocessing.

---

## 3. Mapping to Latte (what exists vs what to build): what exists vs what to build

| Plan step | Current Latte | To build |
|-----------|----------------|----------|
| Admission-level data | ✅ `lab_events_with_adm.parquet` (with backfilled hadm_id) | — |
| Lab completeness per admission | ❌ | Compute % of (time bins × lab types) with at least one value per admission; filter hadm_id with ≥60–70%. |
| Fixed time windows (24 h / 1 h bins) | ❌ | Segment labs by admission and time bin (e.g. 0–1 h, 1–2 h, …); one “frame” per bin. |
| Lab as channels / grid per bin | ✅ Per-specimen grid (H×W) exists | Either: (1) **per-admission, per–time-bin** grids (T×H×W), or (2) T×C with C = lab itemids. |
| Sliding windows per admission | ❌ | For stays >24 h, emit multiple 24 h windows (e.g. 0–24 h, 12–36 h) as separate samples. |
| Padding + masking | ❌ | For sequences shorter than 24 bins, pad and pass a mask into the model (downstream/training code). |

---

## 4. Proposed implementation steps

### Step 1: Lab completeness per admission (filter for Phase 1)

- **Input:** `lab_events_with_adm.parquet` (or raw labevents + admissions).
- **Definition of completeness (example):** For each `hadm_id`, define a reference grid (e.g. first 24 h from admission, 1 h bins, top-K itemids). Completeness = (number of (bin, itemid) cells with ≥1 value) / (number of cells).
- **Output:** List or table of `hadm_id` with completeness ≥ threshold (e.g. 0.6 or 0.7); optionally a filtered Parquet or admission-level CSV for downstream steps.
- **Where:** New script `scripts/admission_completeness.py` (or a section in `lab_with_admissions.py`). Use Polars/DuckDB for aggregation.

### Step 2: Time-window segmentation (24 h, 1 h bins)

- **Input:** Lab events with `hadm_id` and `charttime` (and admission `admittime` from join).
- **Logic:** For each admission, anchor to `admittime`; assign each lab event to a time bin index: `bin_id = floor((charttime - admittime) / 1h)` for 0–23 (or 0–47 for 48 h with 1 h bins).
- **Output:** Lab events with a new column `time_bin` (0..23 for 24 h), and optionally only keep bins in [0, 24) h from admission.
- **Where:** New script or function `scripts/time_window_segment.py`: read lab_events_with_adm, add `time_bin`, write `lab_events_24h_binned.parquet` or pass directly to the next step.

### Step 3: Build “videos” (T×H×W) per admission/window

- **Input:** Binned lab events (hadm_id, time_bin, itemid, valuenum, ref ranges).
- **Logic:** For each (hadm_id) or (hadm_id, window_start) for sliding:
  - For each time_bin in 0..23, build one “frame” as an **H×W grid**: (row, col) = lab type (same mapping as current grid), **pixel value = real lab value** (raw `valuenum` or normalized by ref range; missing = 0 or mask).
  - Stack frames → tensor of shape **(T, H, W)** with **time = tube**, **lab type = spatial (H×W)**, **value = lab value**.
- **Output:** One file per admission (or per window), e.g. `admission_<hadm_id>_0_24.npy` with shape (24, H, W).
- **Where:** New script `scripts/admission_to_video.py` (or extend `specimen_to_image` concept to admission + time bins). Reuse `lab_grid` itemid→(row,col); use real lab values in pixels (optional normalization for training).

### Step 4: Sliding windows (optional, for long stays)

- For admissions with data spanning >24 h, emit multiple 24 h windows (e.g. 0–24 h, 12–36 h, 24–48 h) as separate samples.
- **Input:** Binned lab events with `time_bin` relative to admission start; or raw charttime.
- **Output:** Multiple (hadm_id, window_start_hour) pairs; then run Step 3 per window.
- **Where:** Same `admission_to_video.py` or a small helper that generates (hadm_id, start_hour) and loops.

### Step 5: Completeness filter in pipeline

- After Step 1, restrict the set of `hadm_id` used in Steps 2–4 to those with completeness ≥ 60–70% for Phase 1 experiments.
- Make the threshold a CLI flag (e.g. `--min-completeness 0.65`) so Phase 2 can lower it.

### Step 6: Padding and masking (training-time)

- When loading sequences shorter than 24 bins (e.g. discharge before 24 h), pad to 24 and pass an attention mask (or valid-length) into the model. This belongs in the **training/ dataloader** code, not necessarily in the Latte preprocessing scripts; document the expected format (e.g. shape, mask convention) in this doc or in a `data_format.md`.

---

## 5. Suggested order of work

1. **Step 2** (time-window segmentation) — so we have a clear definition of “first 24 h” and time bins.
2. **Step 1** (completeness) — defined over the same 24 h grid and top-K itemids; then filter admissions.
3. **Step 3** (build T×H×W or T×C per admission) — core “admission → video” pipeline; reuse grid and normalization.
4. **Step 4** (sliding windows) — extend to multiple windows per admission.
5. **Step 5** — wire completeness filter into `run_pipeline` or a dedicated `prepare_training_data.py`.
6. **Step 6** — in model/dataloader repo.

---

## 6. References (from PDF)

- Nature (s41746-025-02176-y), OpenReview (ZmeAoWQqe0, bFrNPlWchg), NeurIPS 2022/2023 VideoMAE, arXiv 2411.13683, SSRN 4737942, CVPR 2023 VideoMAE V2, Hugging Face VideoMAE, MCG-NJU/VideoMAE (video_transforms.py).

---

## 7. Open choices to decide

- **Grid (H×W) vs flat (C):** Keep current H×W grid per time bin (spatial prior) vs one vector of length C per bin (simpler; “channel” = lab type).
- **Window length:** Start with 24 h (24 frames) or 48 h (24 frames with 2 h bins)?
- **Completeness denominator:** Only top-K itemids × 24 bins, or all itemids that ever appear in the admission?

Once these are fixed, the implementation steps above can be turned into concrete tasks and wired into the repo.
