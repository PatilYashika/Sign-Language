import os
import cv2
import numpy as np
import torch
from collections import deque
from model import SignLSTM
from feature_extractor import HandFeatureExtractor
from labels import TOTAL_FEATURES, SEQUENCE_LENGTH

MODEL_PATH = "lstm_model.pth"

def draw_hud(frame, current_pred, confidence, recent_predictions):
    h, w, _ = frame.shape
    
    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 85), (25, 25, 25), -1)
    cv2.line(frame, (0, 85), (w, 85), (70, 70, 70), 2)

    # Format human readable text (replace underscores with spaces)
    display_text = current_pred.replace("_", " ") if current_pred else "Detecting..."
    
    # Text and confidence
    cv2.putText(frame, f"Sign: {display_text}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    
    conf_pct = int(confidence * 100)
    conf_color = (0, 255, 0) if conf_pct > 75 else ((0, 165, 255) if conf_pct > 50 else (120, 120, 120))
    cv2.putText(frame, f"Confidence: {conf_pct}%", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_x = w - 220
    bar_w = 180
    bar_h = 18
    cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_w, 30 + bar_h), (50, 50, 50), -1)
    fill_w = int((conf_pct / 100.0) * bar_w)
    cv2.rectangle(frame, (bar_x, 30), (bar_x + fill_w, 30 + bar_h), conf_color, -1)

    # Bottom recent predictions history bar
    cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
    history_str = " -> ".join([p.replace("_", " ") for p in recent_predictions][-4:])
    cv2.putText(frame, f"History: {history_str}", (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: lstm_model.pth not found! Please record data and run train_lstm.py first.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    labels = checkpoint.get("labels", [])
    num_classes = checkpoint.get("num_classes", len(labels))
    input_size = checkpoint.get("input_size", TOTAL_FEATURES)

    model = SignLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    extractor = HandFeatureExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        extractor.close()
        return

    print("\n==================================================")
    print("  REAL-TIME ISL SENTENCE RECOGNITION ACTIVE")
    print("==================================================")
    print(f"Loaded {num_classes} classes.")
    print("Press 'q' in the camera window to quit.\n")

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    recent_predictions = deque(maxlen=5)
    
    current_prediction = ""
    current_confidence = 0.0
    prediction_hold_counter = 0

    cv2.namedWindow("ISL Real-Time Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Real-Time Recognition", 1000, 700)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, result = extractor.extract_landmarks(rgb)

        sequence_buffer.append(features)

        # Predict when buffer is full
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence_buffer, dtype=np.float32)
            norm_seq = HandFeatureExtractor.normalize_sequence(seq_array)
            tensor_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])

                # Confidence threshold for sentences (70%)
                if confidence > 0.70:
                    pred_label = labels[best_idx]
                    current_prediction = pred_label
                    current_confidence = confidence

                    if len(recent_predictions) == 0 or recent_predictions[-1] != pred_label:
                        recent_predictions.append(pred_label)
                else:
                    # Decay confidence display gracefully
                    current_confidence = max(0.0, current_confidence - 0.02)

        draw_hud(frame, current_prediction, current_confidence, recent_predictions)

        cv2.imshow("ISL Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()