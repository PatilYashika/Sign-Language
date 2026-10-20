import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from labels import NUM_HANDS, LANDMARKS_PER_HAND, COORDS, TOTAL_FEATURES

class HandFeatureExtractor:
    """
    Extracts and normalizes 2-hand landmarks (126 features) from camera frames.
    """
    def __init__(self, model_asset_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, rgb_frame, is_mirrored=True):
        """
        Extracts raw landmarks for Left and Right hands.
        Returns:
            features: np.ndarray of shape (126,)
            detection_result: MediaPipe detection result object
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        result = self.detector.detect(mp_image)
        
        # 2 hands: Index 0 = Left Hand (63 floats), Index 1 = Right Hand (63 floats)
        left_hand = np.zeros((LANDMARKS_PER_HAND, COORDS), dtype=np.float32)
        right_hand = np.zeros((LANDMARKS_PER_HAND, COORDS), dtype=np.float32)

        if result.hand_landmarks:
            num_detected = len(result.hand_landmarks)
            if num_detected == 1:
                single_hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)
                raw_handedness = "Right"
                if result.handedness and len(result.handedness) > 0:
                    raw_handedness = result.handedness[0][0].category_name

                # On horizontally flipped (mirrored) webcam frames:
                # MediaPipe 'Right' -> User's physical LEFT Hand
                # MediaPipe 'Left'  -> User's physical RIGHT Hand
                is_physical_left = (raw_handedness == "Right") if is_mirrored else (raw_handedness == "Left")
                if is_physical_left:
                    left_hand = single_hand_coords
                else:
                    right_hand = single_hand_coords
            else:
                # Two hands detected: sort spatially by x-position on mirrored screen
                coords_0 = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)
                coords_1 = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[1]], dtype=np.float32)
                if coords_0[0, 0] <= coords_1[0, 0]:
                    left_hand = coords_0
                    right_hand = coords_1
                else:
                    left_hand = coords_1
                    right_hand = coords_0

        combined = np.concatenate([left_hand.flatten(), right_hand.flatten()])
        return combined, result

    @staticmethod
    def normalize_sequence(sequence):
        """
        Normalizes a (T, 126) sequence.
        Separately centers and scales left hand and right hand if present.
        """
        seq_len = sequence.shape[0]
        normalized = np.zeros_like(sequence, dtype=np.float32)

        for t in range(seq_len):
            frame = sequence[t]
            # Split into left (63) and right (63)
            left_hand = frame[:63].reshape(21, 3).copy()
            right_hand = frame[63:].reshape(21, 3).copy()

            # Normalize Left Hand if detected
            if np.any(left_hand):
                wrist_l = left_hand[0].copy()
                left_hand = left_hand - wrist_l
                size_l = np.max(np.linalg.norm(left_hand, axis=1))
                if size_l > 1e-6:
                    left_hand = left_hand / size_l

            # Normalize Right Hand if detected
            if np.any(right_hand):
                wrist_r = right_hand[0].copy()
                right_hand = right_hand - wrist_r
                size_r = np.max(np.linalg.norm(right_hand, axis=1))
                if size_r > 1e-6:
                    right_hand = right_hand / size_r

            normalized[t] = np.concatenate([left_hand.flatten(), right_hand.flatten()])

        return normalized

    def close(self):
        self.detector.close()
