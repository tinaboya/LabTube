-- eICU vocab_labs-compatible extract for notebooks/explore_lab_tensor.ipynb
-- Output columns mirror expected fields from src/data_preprocess/data/vocab_labs.csv:
-- itemid, label, category
SELECT
  DENSE_RANK() OVER (ORDER BY LOWER(TRIM(labname))) AS itemid,
  labname AS label,
  CAST(MIN(labtypeid) AS STRING) AS category
FROM `physionet-data.eicu_crd.lab`
WHERE labname IS NOT NULL
GROUP BY labname;
