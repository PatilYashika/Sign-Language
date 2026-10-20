import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from labels import ALL_LABELS, SEQUENCE_LENGTH, TOTAL_FEATURES
from feature_extractor import HandFeatureExtractor

DATA_DIR = "data"
PROCESSED_DIR = "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_all():
    print("==================================================")
    print("  PREPROCESSING ISL DATASET")
    print("==================================================")

    # Check which folders actually exist with valid .npy files
    available_labels = []
    if os.path.exists(DATA_DIR):
        for item in sorted(os.listdir(DATA_DIR)):
            folder = os.path.join(DATA_DIR, item)
            if os.path.isdir(folder):
                npy_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
                if len(npy_files) > 0:
                    available_labels.append(item)

    if not available_labels:
        print("ERROR: No recorded sign data found in 'data/' directory!")
        print("Please run 'python record_dataset.py' to record sign sequences first.")
        return

    print(f"\nFound {len(available_labels)} active labels with recorded data:")
    for idx, lbl in enumerate(available_labels, 1):
        num_seq = len([f for f in os.listdir(os.path.join(DATA_DIR, lbl)) if f.endswith('.npy')])
        print(f"  {idx:2d}. {lbl} ({num_seq} sequences)")

    label_to_id = {label: i for i, label in enumerate(available_labels)}
    id_to_label = {i: label for i, label in enumerate(available_labels)}

    # Save active label mapping
    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump({"labels": available_labels, "label_to_id": label_to_id}, f, indent=2)

    X, y = [], []

    for label in available_labels:
        folder = os.path.join(DATA_DIR, label)
        files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

        for file in files:
            path = os.path.join(folder, file)
            raw_seq = np.load(path)

            # Handle 63-feature single-hand data vs 126-feature dual-hand data
            if raw_seq.ndim == 2 and raw_seq.shape[1] == 63:
                # 1. Primary Right-hand slot (matches default single hand)
                padded_r = np.zeros((raw_seq.shape[0], TOTAL_FEATURES), dtype=np.float32)
                padded_r[:, 63:] = raw_seq
                norm_r = HandFeatureExtractor.normalize_sequence(padded_r)
                X.append(norm_r)
                y.append(label_to_id[label])

                # 2. Dual-hand augmentation (Left-hand slot)
                padded_l = np.zeros((raw_seq.shape[0], TOTAL_FEATURES), dtype=np.float32)
                padded_l[:, :63] = raw_seq
                norm_l = HandFeatureExtractor.normalize_sequence(padded_l)
                X.append(norm_l)
                y.append(label_to_id[label])
                continue

            if raw_seq.shape[0] != SEQUENCE_LENGTH or raw_seq.shape[1] != TOTAL_FEATURES:
                print(f"Skipping malformed sequence: {path} with shape {raw_seq.shape}")
                continue

            normalized_seq = HandFeatureExtractor.normalize_sequence(raw_seq)
            X.append(normalized_seq)
            y.append(label_to_id[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nTotal valid sequences: {len(X)}")

    if len(X) < 10:
        print("Not enough sequences to split. Saving all for training.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        # Stratified train-test split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("\nDataset successfully prepared:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set:  {X_test.shape[0]} samples")
    print(f"  Saved to: {PROCESSED_DIR}/")

if __name__ == "__main__":
    preprocess_all()
preprocess = preprocess_all
