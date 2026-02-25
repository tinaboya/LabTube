-- eICU -> lab_events_with_adm-compatible extract for notebooks/explore_lab_tensor.ipynb
-- Output keeps the compatible core columns and preserves raw value fields:
-- subject_id, lab_code, charttime, value, labresult_raw, labresulttext_raw,
-- subject_id_right, race, edregtime, admittime, hadm_id
WITH
  lab_vocab AS (
    SELECT
      labname,
      labTypeID,
      DENSE_RANK() OVER (ORDER BY LOWER(TRIM(labname))) AS lab_code
    FROM (
      SELECT DISTINCT labname, labTypeID
      FROM `physionet-data.eicu_crd.lab`
      WHERE labname IS NOT NULL
    )
  ),
  patient_anchor AS (
    SELECT
      p.*,
      -- Synthetic hospital-discharge timestamp from discharge year.
      TIMESTAMP(
        DATETIME(DATE(COALESCE(p.hospitaldischargeyear, 2000), 12, 31), TIME(23, 59, 59))
      ) AS discharge_anchor_ts,
      -- Reconstruct an approximate unit-admit timestamp using minute offsets.
      TIMESTAMP_SUB(
        TIMESTAMP(
          DATETIME(DATE(COALESCE(p.hospitaldischargeyear, 2000), 12, 31), TIME(23, 59, 59))
        ),
        INTERVAL COALESCE(p.hospitaldischargeoffset, p.unitdischargeoffset, 0) MINUTE
      ) AS admittime_anchor_ts
    FROM `physionet-data.eicu_crd.patient` AS p
  )
SELECT
  p.uniquepid as uniquepid,
  v.lab_code AS lab_code,
  v.labname AS labname,
  v.labTypeID AS labtype,
  TIMESTAMP_ADD(p.admittime_anchor_ts, INTERVAL l.labresultoffset MINUTE) AS charttime,
  l.labresultoffset,
  l.labresult AS labresult,
  l.labresulttext AS labresulttext_raw,
  p.ethnicity AS ethnicity,
  CAST(NULL AS TIMESTAMP) AS edregtime,
  p.admittime_anchor_ts AS admittime,
  l.patientunitstayid AS patientunitstayid 
FROM `physionet-data.eicu_crd.lab` AS l
JOIN patient_anchor AS p
  ON l.patientunitstayid = p.patientunitstayid
JOIN lab_vocab AS v
  ON l.labname = v.labname;
