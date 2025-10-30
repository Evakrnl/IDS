====================================================
PREPROCESSING WORKFLOW — CICIDS2017 & CICIDS2018
====================================================

This document describes the full preprocessing workflow,
step order, and Python scripts used for both preprocessing
scenarios: Label-based and Session-based.

----------------------------------------------------
SCENARIO 1 — PER-YEAR-BASED PREPROCESSING
----------------------------------------------------

Step | Description | Script Used
----------------------------------------------------
1️ | Label inspection (counting categories)
    -> attacks_count.py

2️ | Feature/header and duplicate column check
    -> check_duplicate_features.py

3️ | Split by Label (grouping): one CSV per category
    -> group_by_label.py

4️ | Recount Labels after grouping
    -> attacks_count.py

5️ | Cleaning per-label files:
    drop unnecessary columns, replace ±inf→NaN, drop NaN, with log
    -> delete_column.py

6️ | Isolation of “Attempted Category” files in a separate folder
    -> (manual step / no script)

7️ | Label normalization (consistent naming) and filename alignment
    -> normalize_labels.py

8️ | Sampling of 5,000,000 BENIGN rows (only for 2018)
    -> random_benign_pick.py

9️ | Generation of normalized (0–1) feature statistics per label
    -> create_feature_stats.py


----------------------------------------------------
SCENARIO 2 — SESSION-BASED PREPROCESSING
----------------------------------------------------

Step | Description | Script Used
----------------------------------------------------
1️ | Label inspection (counting categories)
    -> attacks_count.py

2️ | Feature/header and duplicate column check
    -> check_duplicate_features.py

3️ | Grouping by original CSV (session):
    creates subfolder per file, saves per-label CSVs, and detailed logs
    -> group_by_label_per_session.py

4️ | (Label recount) — included automatically in logs from Step 3
    -> —

5️ | Isolation of “Attempted Category” files in a separate folder
    (manual/organizational step, no script)
    -> —

6️ | Cleaning per-label files:
    drop unnecessary columns, replace ±inf→NaN, drop NaN, with log
    -> delete_column.py

7️ | Label normalization (consistent naming) and filename alignment
    -> normalize_labels.py

8️ | BENIGN sampling (only for 2018):
    500,000 BENIGN rows per session (total 5,000,000)
    -> random_benign_pick.py

9️ | Enumeration of first appearances (LabelName-<index>):
    renames file and updates the Label column accordingly
    -> label_from_filename.py

10 | Generation of normalized (0–1) feature statistics per label
    -> create_feature_stats.py


----------------------------------------------------
GENERAL NOTES
----------------------------------------------------
• Logs from each preprocessing stage are saved in:
  Documents/ids/datasets directories.

• Shared scripts between both scenarios:
  delete_column.py, normalize_labels.py, create_feature_stats.py

• The same sequence is applied to both CICIDS2017 and CICIDS2018 datasets,
  with the only difference being BENIGN sampling (applied only in 2018).

• The process is fully reproducible and well-documented.

====================================================
END OF FILE
====================================================
