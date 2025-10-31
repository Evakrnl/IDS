# Per-Session Folder Structure

This directory (`per_session/`) contains the **CICIDS2017** and **CICIDS2018** datasets organized **by collection day**. Each day folder includes the corresponding session CSV files labeled and enumerated for easier identification.

---

##  Structure Overview

```
per_session/
├─ 2017/
│  ├─ monday/
│  ├─ tuesday/
│  ├─ wednesday/
│  ├─ thursday/
│  └─ friday/
│     ├─ BENIGN_1.csv
│     ├─ Botnet Ares_1.csv
│     ├─ DDoS LOIC HTTP_1.csv
│     └─ Portscan_1.csv
│
└─ 2018/
   ├─ Friday-16-02-2018/
   ├─ Friday-23-02-2018/
   ├─ Thursday-01-03-2018/
   ├─ Thursday-15-02-2018/
   └─ Friday-02-03-2018/
      ├─ BENIGN_6.csv
      └─ Botnet Ares_2.csv
```

Each **day folder** contains one or more `.csv` files representing traffic sessions captured during that day.

---

##  File Naming Convention

The CSV files follow the pattern:

```
<AttackName>_<Index>.csv
```

Examples:

* `BENIGN_1.csv`
* `Botnet Ares_2.csv`
* `DDoS LOIC HTTP_1.csv`

The index (`_1`, `_2`, etc.) indicates the **order of appearance** within that dataset, ensuring each file name and its internal `Label` column are unique.

---

##  Purpose

This folder structure allows the dataset to be:

* Processed **per day** (session-based analysis).
* Easily merged or selected by date.
* Used consistently across both **CICIDS2017** and **CICIDS2018** datasets.

Each CSV file has a `Label` column corresponding to its filename, ensuring data consistency between file name and label.

---

##  Notes

* The `.gitkeep` files are placeholders to preserve empty folders in version control.
* The naming and labeling were generated automatically using the script `enumerate_csv_labels.py`.
* All datasets in this structure are ready to be loaded by the data processing and prediction pipelines.
