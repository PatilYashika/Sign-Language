import cv2
import numpy as np

# Cyber HUD Color Palette (BGR)
CYAN = (255, 230, 0)
CYAN_GLOW = (255, 160, 0)
MAGENTA = (230, 0, 255)
MAGENTA_GLOW = (180, 0, 200)
EMERALD = (80, 230, 80)
AMBER = (0, 180, 255)
DARK_BG = (15, 18, 26)
PANEL_BORDER = (55, 65, 81)
WHITE = (250, 250, 250)
GRAY = (140, 140, 150)

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

def draw_corner_brackets(img, pt1, pt2, color, length=15, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_cyber_landmarks(frame, detection_result, show_skeleton=True, is_mirrored=True):
    """
    Renders high-tech glowing cyber landmarks and bounding boxes on the frame.
    Accurately accounts for mirrored camera inversion so physical Left Hand
    is labeled 'LEFT HAND' and physical Right Hand is labeled 'RIGHT HAND'.
    """
    if not detection_result or not detection_result.hand_landmarks:
        return frame

    h, w, _ = frame.shape

    for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
        raw_handedness = "Right"
        if detection_result.handedness and len(detection_result.handedness) > hand_idx:
            raw_handedness = detection_result.handedness[hand_idx][0].category_name

        # On horizontally flipped (mirrored) webcam frames, MediaPipe's classification is inverted:
        # MediaPipe 'Right' -> User's physical LEFT Hand
        # MediaPipe 'Left'  -> User's physical RIGHT Hand
        if is_mirrored:
            is_left = (raw_handedness == "Right")
        else:
            is_left = (raw_handedness == "Left")

        main_color = CYAN if is_left else MAGENTA
        glow_color = CYAN_GLOW if is_left else MAGENTA_GLOW
        label_text = "LEFT HAND" if is_left else "RIGHT HAND"

        pts = []
        min_x, min_y = w, h
        max_x, max_y = 0, 0

        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((px, py))
            min_x = min(min_x, px)
            min_y = min(min_y, py)
            max_x = max(max_x, px)
            max_y = max(max_y, py)

        if show_skeleton:
            # Draw connection lines with smooth alpha glow
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = pts[start_idx]
                p2 = pts[end_idx]
                # Outer glow
                cv2.line(frame, p1, p2, glow_color, 3, cv2.LINE_AA)
                # Core line
                cv2.line(frame, p1, p2, main_color, 1, cv2.LINE_AA)

            # Draw glowing landmark joints
            for idx, pt in enumerate(pts):
                # Larger radius for wrist and fingertips
                is_tip = idx in (4, 8, 12, 16, 20)
                is_wrist = (idx == 0)
                radius = 6 if (is_tip or is_wrist) else 3

                # Outer glow
                cv2.circle(frame, pt, radius + 2, glow_color, -1, cv2.LINE_AA)
                # Core circle
                cv2.circle(frame, pt, radius, WHITE if is_tip else main_color, -1, cv2.LINE_AA)

            # Sci-Fi Bounding Box with Padding
            pad = 18
            bx1 = max(0, min_x - pad)
            by1 = max(0, min_y - pad)
            bx2 = min(w, max_x + pad)
            by2 = min(h, max_y + pad)

            # Draw subtle semi-transparent box background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (30, 30, 40), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Tech corner brackets
            draw_corner_brackets(frame, (bx1, by1), (bx2, by2), main_color, length=14, thickness=2)

            # Hand Tag Label
            tag_w = 95
            tag_h = 20
            tag_y1 = max(0, by1 - tag_h - 4)
            cv2.rectangle(frame, (bx1, tag_y1), (bx1 + tag_w, tag_y1 + tag_h), (20, 24, 33), -1)
            cv2.rectangle(frame, (bx1, tag_y1), (bx1 + tag_w, tag_y1 + tag_h), main_color, 1)
            cv2.putText(frame, label_text, (bx1 + 6, tag_y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, main_color, 1, cv2.LINE_AA)

    return frame

def draw_hud(frame, current_pred, confidence, recent_predictions, buffer_len=30, max_buffer=30, fps=0.0):
    """
    Renders the modern glassy sci-fi HUD on top of the webcam frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top HUD Header Panel (Semi-transparent dark glass)
    top_h = 92
    cv2.rectangle(overlay, (0, 0), (w, top_h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, top_h), (w, top_h), PANEL_BORDER, 2)

    # Glowing Accent Line at very top
    conf_pct = int(confidence * 100)
    status_color = EMERALD if conf_pct >= 75 else (AMBER if conf_pct >= 45 else GRAY)
    cv2.line(frame, (0, 0), (w, 0), status_color, 3)

    # Title & Detected Sign
    cv2.putText(frame, "AI ISL RECOGNITION", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)

    display_pred = current_pred.replace("_", " ") if current_pred else "AWAITING SIGN..."
    pred_color = status_color if current_pred else (120, 130, 150)
    cv2.putText(frame, display_pred, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.1, pred_color, 2, cv2.LINE_AA)

    # Right side: Confidence & Performance Metrics
    bar_w = 170
    bar_h = 16
    bar_x = w - bar_w - 24
    bar_y = 48

    # Confidence Label & Score
    cv2.putText(frame, f"CONFIDENCE: {conf_pct}%", (bar_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)
    if fps > 0:
        cv2.putText(frame, f"{fps:.1f} FPS", (w - 75, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 160), 1, cv2.LINE_AA)

    # Confidence Progress Bar Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 42, 55), -1)
    fill_w = int((conf_pct / 100.0) * bar_w)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), PANEL_BORDER, 1)

    # 2. Bottom HUD Footer Panel (History Stream & Buffer meter)
    bot_h = 44
    bot_y = h - bot_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bot_y), (w, h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, bot_y), (w, bot_y), PANEL_BORDER, 1)

    # History Stream
    hist_clean = [p.replace("_", " ") for p in recent_predictions]
    history_str = "  ->  ".join(hist_clean[-4:]) if hist_clean else "None yet"
    cv2.putText(frame, "HISTORY:", (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, history_str, (95, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 240), 1, cv2.LINE_AA)

    # Buffer indicator (right side)
    buf_pct = int((buffer_len / float(max_buffer)) * 100)
    buf_color = CYAN if buffer_len == max_buffer else (100, 100, 120)
    cv2.putText(frame, f"BUFFER: {buffer_len}/{max_buffer}", (w - 150, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, buf_color, 1, cv2.LINE_AA)

    return frame
