# Datasets Overview

## Source
The original datasets were obtained from the Canadian Institute for Cybersecurity:
- CICIDS 2017
- CSE-CIC-IDS 2018

This thesis uses the **improved CICIDS2017 / CSE-CIC-IDS2018 datasets** prepared by DistriNet (KU Leuven) for **CNS 2022**:

- https://intrusion-detection.distrinet-research.be/CNS2022/index.html

> The original CIC datasets are widely known, but in this work we rely **exclusively on the improved releases linked above** (cleaned labels, fixes, and grouped variants).  
> Due to size and licensing, the data files themselves are **not included** in this repository.

## Folder Structure

datasets/
├─ original/
│ ├─ CICIDS2017_improved/ # place improved CSVs here (not tracked)
│ ├─ CSECICIDS2018_improved/
│ ├─ CICIDS2017_improved_grouped/
│ ├─ CSECICIDS2018_improved_grouped/
│ └─ attempted_categories/ # filtered/excluded samples + label stats
│ ├─ 2017/
│ └─ 2018/
│
├─ preprocessed/
│ ├─ per_session/
│ │ ├─ 2017/
│ │ └─ 2018/
│ └─ per_year/
│ ├─ 2017/
│ └─ 2018/
│
└─ synthetics/
├─ oversampling_TVAE/
│ ├─ 2017/
│ ├─ 2018/
│ └─ for_all/
└─ predict_CTGAN/
│ ├─ 2017/
│ ├─ 2018/
│ └─ for_all/


### Explanation
- **`original/`**  
  The improved datasets as downloaded from the CNS2022 page, including **grouped** variants by attack type and an `attempted_categories/` folder for excluded/attempted samples and label statistics.

- **`preprocessed/`**  
  Cleaned/standardized subsets produced by this thesis (ready for training/testing).  
  The exact steps and I/O are documented in `/preprocessing/README.txt`.

- **`synthetics/`**  
  Synthetic samples generated with **TVAE** and **CTGAN**, typically conditioned per attack label (used for balancing and generalization).

## Download & Placement
1. Download the improved datasets from the CNS2022 page above.  
2. Place the CSV files under the corresponding folders, e.g.:


*(This repository tracks only the **folder structure**; large data files are ignored.)*

## Schemas & Notes
- Column schemas and datatypes are summarized in `original/SCHEMA.md`.
- Some numeric columns (e.g., `Flow Bytes/s`, `Flow Packets/s`) may contain `inf`/`NaN`; these are handled during preprocessing.
- Grouped variants exclude incomplete/attempted attacks to stabilize labels.
