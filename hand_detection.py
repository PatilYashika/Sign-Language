import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. Create the MediaPipe hand landmarker
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# --------------------------------------------------
# 2. Start webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()

print("Hand detection started.")
print("Press Q to quit.")


# --------------------------------------------------
# 3. Process webcam frames
# --------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = detector.detect(mp_image)


    # --------------------------------------------------
    # 4. Draw landmarks
    # --------------------------------------------------

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            height, width, _ = frame.shape

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )


    # --------------------------------------------------
    # 5. Display webcam
    # --------------------------------------------------

    cv2.imshow(
        "ISL Hand Landmark Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 6. Cleanup
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()

detector.close()

print("Program ended.")