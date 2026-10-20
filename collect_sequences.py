import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# SETTINGS
# =========================

# Change this label for each sign:
# HELLO
# THANK_YOU
# YES
# NO
# PLEASE

LABEL = "PLEASE"

SEQUENCE_LENGTH = 30
NUM_SEQUENCES = 10

OUTPUT_DIR = os.path.join("data", LABEL)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# MEDIAPIPE SETUP
# =========================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# =========================
# CAMERA SETUP
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam")
    detector.close()
    exit()


# =========================
# FIND EXISTING SEQUENCES
# =========================

existing_files = [
    f
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".npy")
]

existing_numbers = []

for filename in existing_files:

    try:
        number = int(
            filename.replace("sequence_", "")
                    .replace(".npy", "")
        )

        existing_numbers.append(number)

    except ValueError:
        pass


if existing_numbers:
    sequence_number = max(existing_numbers) + 1
else:
    sequence_number = 1


print("===================================")
print(f"{LABEL} SEQUENCE DATA COLLECTION")
print("===================================")

print(f"Existing sequences found: {len(existing_files)}")
print(f"Starting from sequence: {sequence_number}")
print(f"Target total sequences: {NUM_SEQUENCES}")
print(f"Frames per sequence: {SEQUENCE_LENGTH}")

print()
print("Press S to start recording.")
print("Press Q to quit.")
print("===================================")


# =========================
# MAIN COLLECTION LOOP
# =========================

while sequence_number <= NUM_SEQUENCES:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read webcam frame")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Display information
    cv2.putText(
        frame,
        f"{LABEL} - Sequence {sequence_number}/{NUM_SEQUENCES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press S to record",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "ISL Sequence Collection",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q") or key == ord("Q"):

        print("Collection stopped by user.")
        break


    # Start recording
    if key == ord("s") or key == ord("S"):

        print()
        print(
            f"Recording {LABEL} "
            f"sequence {sequence_number}..."
        )

        sequence = []


        # =========================
        # RECORD 30 HAND FRAMES
        # =========================

        while len(sequence) < SEQUENCE_LENGTH:

            success, frame = camera.read()

            if not success:

                print("ERROR: Could not read frame")
                break


            frame = cv2.flip(frame, 1)


            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # Detect hand
            result = detector.detect(mp_image)


            # =========================
            # HAND FOUND
            # =========================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                landmarks = []


                # 21 landmarks × 3 coordinates = 63 values
                for landmark in hand:

                    landmarks.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])


                sequence.append(landmarks)


                cv2.putText(
                    frame,
                    f"Recording: "
                    f"{len(sequence)}/{SEQUENCE_LENGTH}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            # =========================
            # NO HAND
            # =========================

            else:

                cv2.putText(
                    frame,
                    "Show your hand!",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Frames: "
                    f"{len(sequence)}/{SEQUENCE_LENGTH}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


            cv2.imshow(
                "ISL Sequence Collection",
                frame
            )


            key = cv2.waitKey(1) & 0xFF


            # Allow quitting during recording
            if key == ord("q") or key == ord("Q"):

                print("Collection stopped by user.")

                camera.release()
                cv2.destroyAllWindows()
                detector.close()

                exit()


        # =========================
        # SAVE SEQUENCE
        # =========================

        if len(sequence) == SEQUENCE_LENGTH:

            sequence = np.array(
                sequence,
                dtype=np.float32
            )


            filename = os.path.join(
                OUTPUT_DIR,
                f"sequence_{sequence_number:03d}.npy"
            )


            # Safety check: never overwrite
            if os.path.exists(filename):

                print(
                    f"WARNING: {filename} already exists."
                )

                print(
                    "Skipping this sequence."
                )

            else:

                np.save(
                    filename,
                    sequence
                )

                print(
                    f"Saved: {filename}"
                )

                print(
                    f"Shape: {sequence.shape}"
                )

                sequence_number += 1


        else:

            print(
                "Sequence incomplete."
                " Please try again."
            )


# =========================
# CLEANUP
# =========================

camera.release()

cv2.destroyAllWindows()

detector.close()


print()
print("===================================")
print("COLLECTION FINISHED")
print("===================================")
print(f"Label: {LABEL}")
print(f"Total target sequences: {NUM_SEQUENCES}")
