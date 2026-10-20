import cv2
import mediapipe as mp
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

LABEL = "HELLO"

OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{LABEL}.csv"
)


# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------

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


# --------------------------------------------------
# CSV FILE
# --------------------------------------------------

file_exists = os.path.exists(OUTPUT_FILE)

file = open(
    OUTPUT_FILE,
    "a",
    newline=""
)

writer = csv.writer(file)


# --------------------------------------------------
# WEBCAM
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()

print("Landmark collection started.")
print("Perform the sign:", LABEL)
print("Press Q to stop.")


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)


    # --------------------------------------------------
    # EXTRACT LANDMARKS
    # --------------------------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        row = []

        for landmark in hand:

            row.append(landmark.x)
            row.append(landmark.y)
            row.append(landmark.z)

        # Save label + landmarks
        writer.writerow(
            [LABEL] + row
        )

        file.flush()

        print(
            f"Saved frame for {LABEL}",
            end="\r"
        )


    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"Collecting: {LABEL}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Landmark Collection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()

file.close()
detector.close()

print("\nCollection finished.")