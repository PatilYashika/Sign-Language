import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import SignLSTM
from labels import TOTAL_FEATURES

PROCESSED_DIR = "processed"
MODEL_PATH = "lstm_model.pth"

def train():
    print("==================================================")
    print("  TRAINING ISL RECOGNITION MODEL")
    print("==================================================")

    label_map_file = os.path.join(PROCESSED_DIR, "label_map.json")
    if not os.path.exists(label_map_file):
        print("ERROR: label_map.json not found! Run 'python preprocess_data.py' first.")
        return

    with open(label_map_file, "r") as f:
        meta = json.load(f)
    labels = meta["labels"]
    num_classes = len(labels)
    print(f"Training on {num_classes} active classes: {', '.join(labels[:10])}{'...' if len(labels)>10 else ''}")

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    batch_size = 32 if len(X_train) >= 64 else 16
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLSTM(input_size=TOTAL_FEATURES, hidden_size=128, num_layers=2, num_classes=num_classes, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
    epochs = 60

    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        scheduler.step()
        train_acc = (correct_train / total_train) * 100 if total_train > 0 else 0

        # Evaluation
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == batch_y).sum().item()
                total_test += batch_y.size(0)

        test_acc = (correct_test / total_test) * 100 if total_test > 0 else 0

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            avg_loss = total_loss / total_train
            print(f"Epoch [{epoch:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "labels": labels,
                "input_size": TOTAL_FEATURES,
                "hidden_size": 128,
                "num_classes": num_classes
            }, MODEL_PATH)

    print(f"\nTraining Complete! Best Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
