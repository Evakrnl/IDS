import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_data(base_folders):
    """
    Read and combine all CSV files from the specified folders.

    Parameters
    ----------
    base_folders : list of str
        List of folders containing CSV files.

    Returns
    -------
    dict[str, list[pd.DataFrame]]
        A dictionary where each key is a filename (label)
        and the value is a list of DataFrames for that label.
    """
    print("Reading all CSV files...")
    combined_files = {}

    for folder in base_folders:
        all_csvs = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Folder: {folder} | Found {len(all_csvs)} files")

        for csv_file in all_csvs:
            file_name = os.path.basename(csv_file)
            try:
                df_temp = pd.read_csv(csv_file)
                combined_files.setdefault(file_name, []).append(df_temp)
                print(f"  Loaded: {file_name} ({len(df_temp)} rows)")
            except Exception as e:
                print(f" Error reading {csv_file}: {e}")

    return combined_files


def select_labels(available_labels):
    """
    Prompt the user to select which labels to process.

    Parameters
    ----------
    available_labels : list of str
        List of available labels (CSV filenames).

    Returns
    -------
    list[str]
        List of selected labels.
    """
    print("\nAvailable labels:")
    for i, lbl in enumerate(available_labels, 1):
        print(f"{i}. {lbl.replace('.csv', '')}")

    while True:
        choice = input("\nEnter label numbers (e.g. 1,3,5) or 'all' for all: ").strip().lower()
        if choice in ["all", ""]:
            return available_labels

        indices = []
        valid = True
        for x in choice.split(","):
            x = x.strip()
            if x.isdigit():
                idx = int(x) - 1
                if 0 <= idx < len(available_labels):
                    indices.append(idx)
                else:
                    print(f"'{x}' is not a valid label number.")
                    valid = False
            else:
                print(f"'{x}' is not a valid number.")
                valid = False
        if valid and indices:
            return [available_labels[i] for i in indices]
        print("Please try again with a valid selection.")


def combine_selected_data(combined_files, selected_labels):
    """
    Combine all selected DataFrames into one final dataset.

    Parameters
    ----------
    combined_files : dict[str, list[pd.DataFrame]]
        The loaded CSV files grouped by filename.
    selected_labels : list of str
        The labels selected by the user.

    Returns
    -------
    pd.DataFrame
        The combined and cleaned DataFrame.
    """
    dfs = [pd.concat(combined_files[file_name], ignore_index=True)
           for file_name in selected_labels]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal dataset: {len(df)} rows")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Removed {before - len(df)} rows with NaN/Inf values")

    return df


def load_models(models_dir):
    """
    Load all saved XGBoost models from subfolders.

    Parameters
    ----------
    models_dir : str
        Path to the directory containing model subfolders.

    Returns
    -------
    dict[str, XGBClassifier]
        Dictionary mapping label names to loaded models.
    """
    print("\nLoading saved models...")
    models = {}

    for label_folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                model_path = os.path.join(folder_path, file)
                try:
                    model = XGBClassifier(tree_method="hist", device="cuda")
                    model.load_model(model_path)
                    models[label_folder] = model
                    print(f" Loaded model: {label_folder}")
                except Exception as e:
                    print(f" Error loading {model_path}: {e}")

    return models


def predict_all(models, df, threshold=0.2):
    """
    Perform predictions using all models (One-vs-All approach).

    Parameters
    ----------
    models : dict[str, XGBClassifier]
        Loaded models for each label.
    df : pd.DataFrame
        Data containing features and the 'Label' column.
    threshold : float, optional
        Minimum probability for accepting a label prediction.
        Defaults to 0.2.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The DataFrame with probability and final prediction columns,
        and the true labels.
    """
    X = df.drop(columns=["Label"])
    y_true = df["Label"]

    for label, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        df[f"prob_{label}"] = probs

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    max_proba = df[prob_cols].max(axis=1)
    best_label = df[prob_cols].idxmax(axis=1).str.replace("prob_", "")
    df["final_prediction"] = np.where(max_proba < threshold, "Unknown", best_label)

    return df, y_true


def evaluate(df, y_true, results_path):
    """
    Evaluate model predictions, save classification report and confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predictions.
    y_true : pd.Series
        True labels.
    results_path : str
        Folder path to save the results.
    """
    os.makedirs(results_path, exist_ok=True)

    print("\nClassification Report:")
    labels_for_eval = sorted(set(y_true))
    report = classification_report(
        y_true,
        df["final_prediction"],
        labels=labels_for_eval,
        zero_division=0
    )
    print(report)

    report_path = os.path.join(results_path, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    cm_classes = sorted(set(y_true) | set(df["final_prediction"]))
    cm = confusion_matrix(y_true, df["final_prediction"], labels=cm_classes)
    cm_df = pd.DataFrame(cm, index=cm_classes, columns=cm_classes)
    cm_df.to_csv(os.path.join(results_path, "confusion_matrix.csv"), index=True)
    print("Confusion Matrix saved as CSV.")

    plt.figure(figsize=(20, 14))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=cm_classes, yticklabels=cm_classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(results_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix image saved at: {cm_path}")


def main():
    """Main function for One-vs-All XGBoost prediction pipeline."""
    data_folders = [
        "..."
    ]
    models_dir = "..."
    results_path = "..."

    print(f"Prediction results will be saved at: {results_path}")

    combined_files = load_data(data_folders)
    labels = list(combined_files.keys())
    selected_labels = select_labels(labels)
    df = combine_selected_data(combined_files, selected_labels)
    models = load_models(models_dir)
    df, y_true = predict_all(models, df)
    evaluate(df, y_true, results_path)


if __name__ == "__main__":
    main()
