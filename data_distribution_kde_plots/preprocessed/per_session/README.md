# KDE Plots per Session

This script (`kde_plots_per_session.py`) generates **Kernel Density Estimation (KDE)** plots for each numeric feature in the **CICIDS2017** and **CICIDS2018** datasets, separated **per session (day)**.

Each feature’s distribution is visualized per label (e.g., *BENIGN*, *DoS Slowloris*, *Botnet Ares*, etc.), with subplots for each label, and each subplot indicates the corresponding session/day.

---

##  Configuration

Before running the script, make sure to correctly set the folder paths inside the configuration section:

```python
# === Configuration ===
INPUT_FOLDERS = {
    "2017": "/datasets/preprocessed/per_session/2017",
    "2018": "/datasets/preprocessed/per_session/2018"
}

OUTPUT_FOLDER = "/data_distribution_kde_plots/preprocessed/per_session"
BINS = 10  # number of intervals on the X-axis
```

### Folder explanation

* **INPUT_FOLDERS:** Paths to the yearly dataset folders that contain **subfolders per session/day**.
  Example structure:

  ```
  /datasets/preprocessed/per_session/
  ├─ 2017/
  │  ├─ monday/
  │  ├─ tuesday/
  │  ├─ wednesday/
  │  └─ friday/
  └─ 2018/
     ├─ Friday-02-03-2018/
     ├─ Friday-16-02-2018/
     └─ Thursday-22-02-2018/
  ```

* **OUTPUT_FOLDER:** Path where the resulting KDE plots will be saved, grouped by year.
  Example:

  ```
  /data_distribution_kde_plots/preprocessed/per_session/2017/
  ```

---

##  How to Run

Run the script directly:

```bash
python kde_plots_per_session.py
```

It will:

1. Traverse all day/session subfolders for each year.
2. Read and merge every CSV file that contains a `Label` column.
3. Add a new column `_day_` to indicate the session/day of origin.
4. Generate **KDE plots per numeric feature**, grouped by label and session.
5. Save the plots under the defined `OUTPUT_FOLDER`.

---

##  Output Structure

```
/data_distribution_kde_plots/preprocessed/per_session/
├─ 2017/
│  ├─ Flow Duration_KDE.png
│  ├─ Total Fwd Packet_KDE.png
│  └─ ...
└─ 2018/
   ├─ Flow Duration_KDE.png
   ├─ Total Fwd Packet_KDE.png
   └─ ...
```

Each image file corresponds to a single numeric feature, and each subplot within the figure corresponds to a label (attack type) combined with its respective session/day.

---

##  Included Example Images

Inside the output folders, **only two example images** are included for clarity and repository size efficiency.
These images are meant to provide a **visual reference** of the expected KDE plots.
To generate the full set of feature distributions, you must **run the script locally** with your dataset paths configured.

---

##  Notes

* The script automatically skips columns such as `Dst Port` and `Protocol`.
* The number of KDE subplots adapts dynamically to the number of labels.
* Each figure title follows the format:

  ```
  Feature Distribution: <Feature Name> (<Year>)
  ```
* Subplot titles show both the **Label** and the **Day**, e.g.:

  ```
  DDoS LOIC HTTP (Friday)
  ```

---

##  Example Output Preview

Each generated figure includes:

* A **blue shaded density curve** representing the distribution of the feature.
* Multiple **subplots** (one per label-day combination).
* Properly formatted axes and titles for readability.

---


**File:** `kde_plots_per_session.py`
**Purpose:** Generate KDE visualizations for network feature distributions per day/session in IDS datasets.
