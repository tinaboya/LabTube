SELECT
    l.hadm_id,
    l.subject_id,
    l.specimen_id,
    l.itemid,
    l.charttime,
    l.value,
    SAFE_CAST(l.valuenum AS FLOAT64) AS valuenum,
    SAFE_CAST(l.ref_range_lower AS FLOAT64) AS ref_range_lower,
    SAFE_CAST(l.ref_range_upper AS FLOAT64) AS ref_range_upper
FROM `physionet-data.mimiciv_3_1_hosp.labevents` AS l;
