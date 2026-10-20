import os
import sys
import time
import glob
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_DATASET_DIR = os.path.join("datasets", "Video_Dataset", "Video_Dataset")
IMAGE_DATASET_DIR = os.path.join("datasets", "data")
OUTPUT_DATA_DIR = "data"
SAMPLES_PER_CLASS = 20
SEQUENCE_LENGTH = 30
NUM_WORKERS = 10

def normalize_label_name(name):
    """Normalize sign name: uppercase with spaces replaced by underscores."""
    clean = name.strip().upper().replace(" ", "_")
    # Clean up double underscores if any
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean

def extract_single_video(args):
    """Worker function to extract (30, 126) landmarks from a single video."""
    video_path, out_path = args
    if os.path.exists(out_path):
        return True, "cached"

    try:
        from feature_extractor import HandFeatureExtractor
        extractor = HandFeatureExtractor()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 5:
            cap.release()
            return False, "too_short"

        indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
        target_indices = set(indices)

        frames = []
        cur_idx = 0
        while cap.isOpened() and len(frames) < SEQUENCE_LENGTH:
            ret = cap.grab()
            if not ret:
                break
            if cur_idx in target_indices:
                _, frame = cap.retrieve()
                if frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cur_idx += 1
        cap.release()

        if len(frames) == 0:
            return False, "no_frames"

        # Pad if short or trim if needed
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])
        frames = frames[:SEQUENCE_LENGTH]

        sequence = []
        for f in frames:
            feat, _ = extractor.extract_landmarks(f, is_mirrored=False)
            sequence.append(feat)

        sequence = np.array(sequence, dtype=np.float32)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, sequence)
        return True, "success"
    except Exception as e:
        return False, str(e)

def extract_single_image(args):
    """Worker function to extract landmarks from an image and tile to 30 frames."""
    img_path, out_path = args
    if os.path.exists(out_path):
        return True, "cached"

    try:
        from feature_extractor import HandFeatureExtractor
        extractor = HandFeatureExtractor()

        img = cv2.imread(img_path)
        if img is None:
            return False, "read_fail"

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feat, res = extractor.extract_landmarks(rgb, is_mirrored=False)

        # Tile to (30, 126) for sequence model with subtle temporal jitter
        seq = np.tile(feat, (SEQUENCE_LENGTH, 1)).astype(np.float32)
        if np.any(feat):
            jitter = np.random.normal(0, 0.002, seq.shape).astype(np.float32)
            mask = (seq != 0).astype(np.float32)
            seq = seq + (jitter * mask)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, seq)
        return True, "success"
    except Exception as e:
        return False, str(e)

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print("=" * 65)
    print("  ISL DATASET EXTRACTION PIPELINE")
    print(f"  Target: {SAMPLES_PER_CLASS} samples per class | Parallel Workers: {NUM_WORKERS}")
    print("=" * 65)

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------
    # 1. Collect Video Tasks
    # -------------------------------------------------------------
    video_tasks = []
    video_labels = []
    if os.path.isdir(VIDEO_DATASET_DIR):
        folders = sorted([d for d in os.listdir(VIDEO_DATASET_DIR) if os.path.isdir(os.path.join(VIDEO_DATASET_DIR, d))])
        print(f"\n[1/3] Found {len(folders)} video sign folders in '{VIDEO_DATASET_DIR}'.")
        for folder_name in folders:
            norm_label = normalize_label_name(folder_name)
            video_labels.append(norm_label)
            fpath = os.path.join(VIDEO_DATASET_DIR, folder_name)
            vids = sorted([f for f in os.listdir(fpath) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
            if not vids:
                continue

            # Pick SAMPLES_PER_CLASS videos evenly
            if len(vids) <= SAMPLES_PER_CLASS:
                chosen = vids
            else:
                indices = np.linspace(0, len(vids) - 1, SAMPLES_PER_CLASS, dtype=int)
                chosen = [vids[i] for i in indices]

            label_dir = os.path.join(OUTPUT_DATA_DIR, norm_label)
            for idx, vid_name in enumerate(chosen):
                vpath = os.path.join(fpath, vid_name)
                out_path = os.path.join(label_dir, f"vid_seq_{idx:04d}.npy")
                video_tasks.append((vpath, out_path))

        print(f"   Prepared {len(video_tasks)} video extraction tasks across {len(folders)} sign classes.")

    # -------------------------------------------------------------
    # 2. Collect Image Tasks (Alphabet & Numbers)
    # -------------------------------------------------------------
    image_tasks = []
    image_labels = []
    if os.path.isdir(IMAGE_DATASET_DIR):
        folders = sorted([d for d in os.listdir(IMAGE_DATASET_DIR) if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, d))])
        print(f"\n[2/3] Found {len(folders)} image sign folders in '{IMAGE_DATASET_DIR}'.")
        for folder_name in folders:
            norm_label = normalize_label_name(folder_name)
            image_labels.append(norm_label)
            fpath = os.path.join(IMAGE_DATASET_DIR, folder_name)
            imgs = sorted([f for f in os.listdir(fpath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not imgs:
                continue

            # Pick SAMPLES_PER_CLASS images evenly
            if len(imgs) <= SAMPLES_PER_CLASS:
                chosen = imgs
            else:
                indices = np.linspace(0, len(imgs) - 1, SAMPLES_PER_CLASS, dtype=int)
                chosen = [imgs[i] for i in indices]

            label_dir = os.path.join(OUTPUT_DATA_DIR, norm_label)
            for idx, img_name in enumerate(chosen):
                ipath = os.path.join(fpath, img_name)
                out_path = os.path.join(label_dir, f"img_seq_{idx:04d}.npy")
                image_tasks.append((ipath, out_path))

        print(f"   Prepared {len(image_tasks)} image extraction tasks across {len(folders)} sign classes.")

    total_tasks = len(video_tasks) + len(image_tasks)
    total_classes = len(set(video_labels + image_labels))
    print(f"\n[3/3] Total classes: {total_classes} | Total sample extraction jobs: {total_tasks}")
    print(f"   Executing across {NUM_WORKERS} parallel CPU processes...\n")

    # -------------------------------------------------------------
    # 3. Execute Video Extraction
    # -------------------------------------------------------------
    vid_success = 0
    if video_tasks:
        print("--> Extracting Video Landmarks...")
        t0_vid = time.time()
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_task = {executor.submit(extract_single_video, task): task for task in video_tasks}
            done_count = 0
            for future in as_completed(future_to_task):
                done_count += 1
                ok, status = future.result()
                if ok:
                    vid_success += 1
                if done_count % 100 == 0 or done_count == len(video_tasks):
                    pct = (done_count / len(video_tasks)) * 100
                    elapsed = time.time() - t0_vid
                    speed = done_count / max(1e-3, elapsed)
                    print(f"   Progress: {done_count}/{len(video_tasks)} ({pct:.1f}%) | {speed:.1f} vids/sec | Elapsed: {elapsed:.1f}s")

    # -------------------------------------------------------------
    # 4. Execute Image Extraction
    # -------------------------------------------------------------
    img_success = 0
    if image_tasks:
        print("\n--> Extracting Image Landmarks (Alphabet & Numbers)...")
        t0_img = time.time()
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_task = {executor.submit(extract_single_image, task): task for task in image_tasks}
            done_count = 0
            for future in as_completed(future_to_task):
                done_count += 1
                ok, status = future.result()
                if ok:
                    img_success += 1
                if done_count % 100 == 0 or done_count == len(image_tasks):
                    pct = (done_count / len(image_tasks)) * 100
                    elapsed = time.time() - t0_img
                    speed = done_count / max(1e-3, elapsed)
                    print(f"   Progress: {done_count}/{len(image_tasks)} ({pct:.1f}%) | {speed:.1f} imgs/sec | Elapsed: {elapsed:.1f}s")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  EXTRACTION COMPLETE!")
    print(f"  Videos processed: {vid_success}/{len(video_tasks)}")
    print(f"  Images processed: {img_success}/{len(image_tasks)}")
    print(f"  Total valid sequences in 'data/': {vid_success + img_success}")
    print(f"  Total time elapsed: {total_elapsed / 60:.2f} minutes")
    print("=" * 65)

if __name__ == "__main__":
    main()
