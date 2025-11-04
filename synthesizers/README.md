Synthetic Data Generation
=========================

This directory contains all code and outputs related to synthetic data generation using deep generative models,
including CTGAN (GAN-based) and TVAE (VAE-based).

Currently supported models:
---------------------------

• CTGAN → implemented in the `ctgan/` folder  
• TVAE  → implemented in the `tvae/` folder

Each model subfolder includes:
  - The training script (e.g., generate_ctgan.py)
  - Per-label synthetic model outputs (trained models, metrics, feature comparisons)
  - Separate runs for 2017, 2018, and combined data (for_all)

Folder Structure:
-----------------

• ctgan/
    Synthetic data generation using CTGANSynthesizer from SDV.
    Includes per-label models, metrics, and comparison statistics.

• tvae/
    Same structure, using TVAESynthesizer instead of CTGAN.

• Each subfolder (e.g., ctgan/2017/) contains:

    - model_<LABEL>.pkl              → Trained model for a specific label
    
    - <LABEL>_metrics.json           → Overall similarity metrics
    
    - <LABEL>_per_feature_distances.csv → Per-feature statistical comparison

Usage in the IDS Pipeline
--------------------------

The generated synthetic data was used in different phases of the IDS experimentation:

• CTGAN:
  Synthetic attack samples were used as the main input during the **prediction phase** of the XGBoost-based IDS model.
  Instead of real attack data, the model received **CTGAN-generated samples**.
  This approach was chosen in order to test whether the model could **generalize** to artificially generated attacks,
  rather than simply memorizing the real samples seen during training.

  The decision to use CTGAN samples only during prediction (and not for training) was intentional:
  - CTGAN had slightly lower quality compared to TVAE for certain categories.
  - Its internal generative process differs significantly from that of TVAE.
  - I wanted to **avoid training and evaluating on samples that might share structural similarities**, in order to
    prevent misleadingly high performance due to data familiarity.

• TVAE:
  Synthetic samples from TVAE were used during the **training phase** as a form of **oversampling** to improve class balance.
  Because TVAE showed better generation quality overall, and its latent space behaved more smoothly,
  it was more suitable for enriching underrepresented classes without overfitting risk.


Notes:
------
- All models are trained using real, preprocessed data from `datasets/preprocessed/per_year`.
- The `for_all/` folder in each method includes experiments combining categories from both 2017 and 2018 datasets.
- Trained models can be reloaded using SDV’s `.load()` method to generate additional synthetic samples without retraining.

Model Compatibility Notice:
---------------------------
The saved models (.pkl) are only compatible with the **exact versions** of the libraries used
at the time of training. Loading them with a different version of SDV, torch, pandas, or other
dependencies may result in runtime errors or invalid outputs.

To ensure reproducibility:
• Each model folder (e.g., `ctgan/`, `tvae/`) includes a `requirements.txt` with the exact versions used.
• It is strongly recommended to use a virtual environment and run:
    pip install -r requirements.txt
