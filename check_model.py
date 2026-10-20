import os
import glob
import numpy as np
import torch
from model import SignLSTM
from feature_extractor import HandFeatureExtractor
from labels import TOTAL_FEATURES, ALL_LABELS

MODEL_PATH = "lstm_model.pth"
DATA_DIR = "data"

def check():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    labels = checkpoint.get("labels", [])
    num_classes = checkpoint.get("num_classes", len(labels))
    input_size = checkpoint.get("input_size", TOTAL_FEATURES)

    model = SignLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("==================================================")
    print("  MODEL DIAGNOSTIC & ACCURACY VERIFICATION")
    print("==================================================")

    total_correct = 0
    total_samples = 0

    for label in labels:
        folder = os.path.join(DATA_DIR, label)
        if not os.path.exists(folder):
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        if not files:
            continue

        print(f"\nSign: {label}")
        for file in files:
            raw_seq = np.load(file)
            if raw_seq.ndim == 2 and raw_seq.shape[1] == 63:
                padded = np.zeros((raw_seq.shape[0], TOTAL_FEATURES), dtype=np.float32)
                padded[:, 63:] = raw_seq
                raw_seq = padded

            norm_seq = HandFeatureExtractor.normalize_sequence(raw_seq)
            tensor_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
                pred_idx = int(np.argmax(probs))
                pred_label = labels[pred_idx]
                conf = probs[pred_idx] * 100

            is_correct = (pred_label == label)
            total_samples += 1
            if is_correct:
                total_correct += 1

            status = "CORRECT" if is_correct else f"WRONG (Predicted: {pred_label})"
            print(f"  {os.path.basename(file)} -> {conf:.1f}% {status}")

    if total_samples > 0:
        accuracy = (total_correct / total_samples) * 100
        print("\n==================================================")
        print(f"OVERALL ACCURACY: {total_correct}/{total_samples} ({accuracy:.1f}%)")
        print("==================================================")

if __name__ == "__main__":
    check()