import os
import re
import pandas as pd


def sanitize_filename(name: str) -> str:
    """
    Keep the label name as-is, only replace illegal filesystem characters so the file can be saved.
    Example:
        "Infiltration / Victim" -> "Infiltration _ Victim"
        "DDoS LOIC HTTP"        -> "DDoS LOIC HTTP"
    """
    return re.sub(r'[\\/:\*\?"<>\|\n\r\t]', "_", name.strip())


# Root input folder
rootFolder = "..."

# Map (old -> new)
labelMap = {
    "Botnet": "Botnet Ares",
    "DDoS": "DDoS LOIC HTTP",
    "FTP-Patator": "FTP-BruteForce-Patator",
    "Infiltration - Portscan": "Infiltration - NMAP Portscan",
    "SSH-Patator": "SSH-BruteForce Patator",
    "Infiltration": "Infiltration - Communication Victim Attacker",
    "DDoS-LOIC-HTTP": "DDoS LOIC HTTP",
    "DDoS-LOIC-UDP": "DDoS LOIC UDP",
    "DDoS-HOIC": "DDoS HOIC",
    "SSH-BruteForce": "SSH-BruteForce-Patator",
    "Web Attack - SQL": "Web Attack - SQL Injection",
}

# Processing
for root, dirs, files in os.walk(rootFolder):
    for filename in files:
        if not filename.endswith(".csv"):
            continue

        input_path = os.path.join(root, filename)

        try:
            # Read as strings to avoid dtype issues
            df = pd.read_csv(input_path, dtype=str, low_memory=False)

            # Clean column names (trim spaces)
            df.columns = [col.strip() for col in df.columns]

            if "Label" in df.columns:
                # Normalize Label column (trim + mapping)
                df["Label"] = df["Label"].astype(str).str.strip().replace(labelMap)

                # Unique non-empty labels
                unique_labels = [v for v in df["Label"].dropna().unique() if str(v).strip() != ""]
                if len(unique_labels) == 1:
                    unique_label = str(unique_labels[0]).strip()

                    # File name must match label exactly (only minimal sanitization for illegal chars)
                    new_name = f"{sanitize_filename(unique_label)}.csv"
                    output_path = os.path.join(root, new_name)

                    # Save to new file (overwrites if exists, same behavior as your original)
                    df.to_csv(output_path, index=False)

                    # Delete old file if different path
                    if output_path != input_path:
                        os.remove(input_path)

                    print(f"{filename} -> {new_name} (original removed: {output_path != input_path})")
                else:
                    print(f"{filename} has multiple labels: {unique_labels}")
            else:
                print(f"{filename} has no 'Label' column")

        except Exception as e:
            print(f"Error in file {filename}: {e}")
