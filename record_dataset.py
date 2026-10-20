import cv2
import numpy as np
import os
import time
from labels import CATEGORIES, ALL_LABELS, SEQUENCE_LENGTH
from feature_extractor import HandFeatureExtractor

def draw_text_with_background(img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(30, 30, 30), thickness=2, padding=6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - text_h - padding), (x + text_w + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def main():
    print("==================================================")
    print("  ISL 50-PHRASE INTERACTIVE DATA COLLECTOR")
    print("==================================================")
    
    # Display category options
    cat_list = list(CATEGORIES.keys())
    print("\nSelect a category to record:")
    for idx, cat_name in enumerate(cat_list, start=1):
        print(f" [{idx}] {cat_name} ({len(CATEGORIES[cat_name])} signs)")
    print(f" [{len(cat_list)+1}] All 50+ Signs / Custom Sign Selection")
    
    try:
        choice = input("\nEnter category number (or press Enter for [1]): ").strip()
        choice_idx = int(choice) - 1 if choice else 0
        if 0 <= choice_idx < len(cat_list):
            selected_signs = CATEGORIES[cat_list[choice_idx]]
        else:
            selected_signs = ALL_LABELS
    except Exception:
        selected_signs = CATEGORIES[cat_list[0]]

    print("\nSigns in this session:")
    for i, sign in enumerate(selected_signs, 1):
        print(f"  {i:2d}. {sign}")

    num_samples_input = input("\nHow many sequences per sign? (Default 15): ").strip()
    target_sequences = int(num_samples_input) if num_samples_input.isdigit() else 15

    extractor = HandFeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        extractor.close()
        return

    cv2.namedWindow("ISL Sentence Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Sentence Data Collector", 1000, 700)

    for sign_idx, label in enumerate(selected_signs, 1):
        output_dir = os.path.join("data", label)
        os.makedirs(output_dir, exist_ok=True)
        
        # Count existing samples
        existing = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
        current_count = len(existing)

        while current_count < target_sequences:
            seq_num = current_count + 1

            # --- PHASE 1: COUNTDOWN / GET READY ---
            countdown_seconds = 3
            start_time = time.time()
            cancelled = False

            while True:
                elapsed = time.time() - start_time
                remaining = countdown_seconds - int(elapsed)

                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, result = extractor.extract_landmarks(rgb)

                # Draw overlay
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
                
                draw_text_with_background(frame, f"SIGN [{sign_idx}/{len(selected_signs)}]: {label}", (20, 40), font_scale=0.8, text_color=(0, 255, 255), bg_color=(20, 20, 20))
                draw_text_with_background(frame, f"Sample: {seq_num}/{target_sequences}", (20, 80), font_scale=0.6, text_color=(200, 200, 200), bg_color=(20, 20, 20))
                
                # Countdown prompt
                cd_text = f"GET READY: {max(1, remaining)}" if remaining > 0 else "RECORDING NOW!"
                cd_color = (0, 165, 255) if remaining > 0 else (0, 255, 0)
                draw_text_with_background(frame, cd_text, (w//2 - 120, h//2), font_scale=1.2, text_color=cd_color, bg_color=(0, 0, 0), thickness=3)

                cv2.imshow("ISL Sentence Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cancelled = True
                    break
                if key == ord('s'):  # Skip sign
                    current_count = target_sequences
                    break

                if elapsed >= countdown_seconds:
                    break

            if cancelled:
                break

            # --- PHASE 2: RECORD SEQUENCE (30 FRAMES) ---
            sequence = []
            frame_idx = 0

            while frame_idx < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features, result = extractor.extract_landmarks(rgb)
                sequence.append(features)
                frame_idx += 1

                # Visual progress
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 120), -1)
                
                draw_text_with_background(frame, f"RECORDING: {label} (Sample {seq_num}/{target_sequences})", (20, 40), font_scale=0.8, text_color=(0, 0, 255), bg_color=(0, 0, 0))
                
                progress_pct = int((frame_idx / SEQUENCE_LENGTH) * 100)
                draw_text_with_background(frame, f"Frame {frame_idx}/{SEQUENCE_LENGTH} ({progress_pct}%)", (20, 80), font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0))
                
                # Red recording dot
                cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)

                cv2.imshow("ISL Sentence Data Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cancelled = True
                    break

            if cancelled:
                break

            # Save sequence
            if len(sequence) == SEQUENCE_LENGTH and not cancelled:
                save_path = os.path.join(output_dir, f"sequence_{seq_num:03d}.npy")
                np.save(save_path, np.array(sequence, dtype=np.float32))
                current_count += 1
                print(f"Saved: {save_path}  (Press 'r' during countdown to redo if needed)")

        if cancelled:
            print("\nRecording paused by user.")
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print("\nSession complete!")

if __name__ == "__main__":
    main()
