# KDE Plots per Year

This script (`kde_plots_per_year.py`) generates **Kernel Density Estimation (KDE)** plots for each numeric feature in the **CICIDS2017** and **CICIDS2018** datasets.

Each feature’s distribution is visualized per label (e.g., *BENIGN*, *Botnet Ares*, *DDoS LOIC HTTP*, etc.), producing one figure per feature with multiple subplots — one for each label.

---

##  Configuration

Before running the script, make sure to correctly set the folder paths inside the configuration section:

```python
# === Configuration ===
INPUT_FOLDERS = {
    "2017": "/datasets/preprocessed/per_year/2017",
    "2018": "/datasets/preprocessed/per_year/2018",
}

OUTPUT_FOLDER = "/data_distribution_kde_plots/preprocessed/per_year"
BINS = 10  # number of intervals on the X-axis
```

* `INPUT_FOLDERS`: Directories containing the yearly preprocessed CSV files with a `Label` column.
* `OUTPUT_FOLDER`: Directory where the generated KDE images will be stored.
* `BINS`: Number of divisions on the X-axis (default: 10).

---

##  How to Run

Run the script using:

```bash
python kde_plots_per_year.py
```

The script will:

1. Load all `.csv` files from the folders defined in `INPUT_FOLDERS`.
2. Merge them and identify all numeric columns.
3. Generate KDE distribution plots for each numeric feature grouped by label.
4. Save all resulting plots as `.png` files under the `OUTPUT_FOLDER`, organized by dataset year.

---

##  Output Structure

```
/data_distribution_kde_plots/preprocessed/per_year/
├─ 2017/
│  ├─ Flow Duration_KDE.png
│  ├─ Total Fwd Packet_KDE.png
│  └─ ...
└─ 2018/
   ├─ Flow Duration_KDE.png
   ├─ Total Fwd Packet_KDE.png
   └─ ...
```

Each generated image includes KDE distributions of the same feature across all labels within that dataset.

---

##  Included Example Images

For clarity and repository size efficiency, **only two sample images** from the results are included within the output folders.
These are meant to provide a **visual reference** of what the generated plots look like.

To view all KDE plots, you must **run the script locally** with your dataset directories correctly configured.

---

##  Notes

* If you only wish to process a single dataset (e.g., 2017), simply define one entry in `INPUT_FOLDERS`.
* The script automatically skips columns like `Dst Port` and `Protocol`.
* Each figure dynamically adjusts its grid layout based on the number of labels.
* The KDE visualization uses **Gaussian kernel smoothing** to show the underlying feature distributions.

---

##  Example Output

Each generated figure has:

* Blue shaded area representing the feature’s density distribution.
* One subplot per label.
* Properly formatted X and Y axes.
* Title: `Feature Distribution: <Feature Name> (Year)`

---


**File:** `kde_plots_per_year.py`
**Purpose:** Visualizing feature distributions per year for IDS dataset analysis.
