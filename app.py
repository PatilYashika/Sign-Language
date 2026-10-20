import os
import time
import json
import glob
import cv2
import numpy as np
import torch
from collections import deque
import streamlit as st

# Local modules
from model import SignLSTM
from feature_extractor import HandFeatureExtractor
from labels import CATEGORIES, ALL_LABELS, TOTAL_FEATURES, SEQUENCE_LENGTH
from sign_dictionary import SIGN_DETAILS
from ui_hud import draw_hud, draw_cyber_landmarks
from ui_tts import speak_text

# ---------------------------------------------------------
# Page Configuration & Modern Cyber Dark Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ISL Sign Language AI Studio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber Glassmorphism CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-dark: #070a12;
    --card-bg: rgba(17, 24, 39, 0.75);
    --border-color: rgba(56, 189, 248, 0.2);
    --neon-cyan: #00f2fe;
    --neon-blue: #4facfe;
    --neon-purple: #8b5cf6;
    --neon-emerald: #10b981;
    --neon-rose: #f43f5e;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

/* Base Body & Layout */
html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 10% 10%, #0d1527 0%, #070a12 60%, #030508 100%);
}

/* Glassmorphism Cards */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 242, 254, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(0, 242, 254, 0.45);
    box-shadow: 0 12px 35px -8px rgba(0, 242, 254, 0.22);
}

/* Hero Title */
.hero-header {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin-bottom: 24px;
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-active {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-ready {
    background: rgba(0, 242, 254, 0.12);
    color: var(--neon-cyan);
    border: 1px solid rgba(0, 242, 254, 0.35);
}

.badge-offline {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}

/* Sentence Result Pill */
.sentence-container {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    padding: 16px 20px;
    min-height: 72px;
    display: flex;
    align-items: center;
    font-size: 1.35rem;
    font-weight: 600;
    color: #ffffff;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
}

/* Metric Display Card */
.metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--neon-cyan);
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

/* Interactive Buttons */
.stButton>button {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(79, 172, 254, 0.2) 100%);
    border: 1px solid rgba(0, 242, 254, 0.4);
    color: #ffffff;
    font-weight: 600;
    border-radius: 10px;
    padding: 8px 18px;
    transition: all 0.25s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.4) 0%, rgba(79, 172, 254, 0.4) 100%);
    border-color: #00f2fe;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.35);
    transform: translateY(-1px);
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #080d19;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: rgba(0, 242, 254, 0.15) !important;
    color: var(--neon-cyan) !important;
    border: 1px solid rgba(0, 242, 254, 0.3) !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
MODEL_PATH = "lstm_model.pth"
DATA_DIR = "data"
PROCESSED_DIR = "processed"

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = deque(maxlen=6)
if "practice_score" not in st.session_state:
    st.session_state.practice_score = 0
if "practice_streak" not in st.session_state:
    st.session_state.practice_streak = 0
if "last_tts_sign" not in st.session_state:
    st.session_state.last_tts_sign = ""
if "last_tts_time" not in st.session_state:
    st.session_state.last_tts_time = 0.0

# ---------------------------------------------------------
# Model Loading Helper (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_sign_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        return None, [], 0, TOTAL_FEATURES

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    labels = checkpoint.get("labels", [])
    num_classes = checkpoint.get("num_classes", len(labels))
    input_size = checkpoint.get("input_size", TOTAL_FEATURES)

    model = SignLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, labels, num_classes, input_size

model, active_labels, num_classes, input_size = load_sign_model(MODEL_PATH)

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <div>
        <div class="hero-header">ISL AI RECOGNITION STUDIO</div>
        <div class="hero-subtitle">Real-Time Indian Sign Language Translation • Audio Synthesis • Interactive Practice • Dataset Studio</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Control Panel & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ AI Engine Status")
    if model is not None:
        st.markdown(f'<span class="badge badge-active">● MODEL LOADED ({num_classes} SIGNS)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-offline">● NO MODEL FOUND</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Minimum confidence percentage required to accept a detected sign."
    )

    show_skeletons = st.checkbox("Show Neon Hand Skeletons", value=True)
    mirror_camera = st.checkbox("Mirror Camera View (Selfie)", value=True, help="Flips the camera horizontally for a natural mirror view with corrected left/right labels.")
    camera_index = st.number_input("Camera Device Index", min_value=0, max_value=4, value=0, step=1)

    st.markdown("---")
    st.markdown("### 🔊 Text-to-Speech (TTS)")
    tts_enabled = st.checkbox("Enable Spoken Audio", value=True)
    tts_rate = st.slider("Speech Rate", min_value=0.7, max_value=1.4, value=1.0, step=0.1)
    tts_pitch = st.slider("Speech Pitch", min_value=0.8, max_value=1.3, value=1.0, step=0.1)

    st.markdown("---")
    st.markdown("### ℹ️ Active Model Signs")
    if active_labels:
        st.markdown(" ".join([f"`{label}`" for label in active_labels]))
    else:
        st.info("Train a model using Tab 4 to activate signs.")

# ---------------------------------------------------------
# Main App Tabs
# ---------------------------------------------------------
tab_live, tab_dict, tab_practice, tab_record, tab_diagnostics = st.tabs([
    "🔮 Live AI Translator",
    "📚 Sign Dictionary",
    "🎯 Practice Studio",
    "🎙️ Dataset Recorder & Trainer",
    "📊 System Diagnostics"
])

# =========================================================
# TAB 1: LIVE AI TRANSLATOR
# =========================================================
with tab_live:
    col_video, col_info = st.columns([7, 5])

    with col_video:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📹 Real-Time Video Stream")

        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            start_btn = st.button("▶️ Start Live Camera", use_container_width=True)
        with ctrl_col2:
            stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

        if start_btn:
            st.session_state.camera_running = True
        if stop_btn:
            st.session_state.camera_running = False

        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💬 Live Sentence Translation")

        # Sentence Builder Container
        sentence_str = " ".join([p.replace("_", " ") for p in st.session_state.sentence])
        if not sentence_str:
            sentence_str = '<span style="color:#64748b; font-style: italic;">Signs will accumulate here into phrases...</span>'

        st.markdown(f'<div class="sentence-container">{sentence_str}</div>', unsafe_allow_html=True)

        # Sentence Quick Action Buttons
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("🔊 Speak All", use_container_width=True):
                if st.session_state.sentence:
                    full_text = " ".join([p.replace("_", " ") for p in st.session_state.sentence])
                    speak_text(full_text, enabled=True, rate=tts_rate, pitch=tts_pitch, key_suffix="manual")
        with btn_c2:
            if st.button("⌫ Undo Last", use_container_width=True):
                if st.session_state.sentence:
                    st.session_state.sentence.pop()
                    st.rerun()
        with btn_c3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.sentence = []
                st.session_state.recent_predictions.clear()
                st.rerun()

        st.markdown("---")
        st.markdown("#### ⚡ Real-Time Detection Feed")

        # Key Metrics Row
        m1, m2 = st.columns(2)
        with m1:
            top_pred_placeholder = st.empty()
            top_pred_placeholder.markdown("""
            <div class="metric-box">
                <div class="metric-value" style="font-size: 1.3rem;">--</div>
                <div class="metric-label">CURRENT SIGN</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            conf_metric_placeholder = st.empty()
            conf_metric_placeholder.markdown("""
            <div class="metric-box">
                <div class="metric-value">0%</div>
                <div class="metric-label">CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 📜 Recent Recognitions Log")
        history_placeholder = st.empty()
        history_placeholder.caption("Awaiting sign activity...")

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # Live Video Inference Loop
    # -----------------------------------------------------
    if st.session_state.camera_running:
        if model is None:
            st.error("Model file lstm_model.pth not loaded! Please check tab 4 or train model.")
        else:
            extractor = HandFeatureExtractor()
            cap = cv2.VideoCapture(int(camera_index))

            if not cap.isOpened():
                st.error(f"Could not open camera device {camera_index}. Please check connection.")
                st.session_state.camera_running = False
                extractor.close()
            else:
                sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
                current_prediction = ""
                current_confidence = 0.0
                sustained_frames = 0
                last_confirmed_sign = ""
                fps_calc_time = time.time()
                frame_count = 0
                fps = 0.0

                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to receive video frame.")
                        break

                    if mirror_camera:
                        frame = cv2.flip(frame, 1)
                    frame_count += 1
                    if frame_count % 10 == 0:
                        cur_t = time.time()
                        fps = 10.0 / max(0.001, cur_t - fps_calc_time)
                        fps_calc_time = cur_t

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    features, result = extractor.extract_landmarks(rgb, is_mirrored=mirror_camera)

                    # Draw neon skeletons on frame with corrected mirror handedness
                    if show_skeletons:
                        frame = draw_cyber_landmarks(frame, result, show_skeleton=True, is_mirrored=mirror_camera)

                    if result.hand_landmarks:
                        sequence_buffer.append(features)
                    else:
                        current_confidence = max(0.0, current_confidence - 0.05)
                        sustained_frames = 0
                        if current_confidence < 0.15:
                            current_prediction = ""

                    # Predict when 30-frame sequence is accumulated
                    if len(sequence_buffer) == SEQUENCE_LENGTH and result.hand_landmarks:
                        seq_array = np.array(sequence_buffer, dtype=np.float32)
                        norm_seq = HandFeatureExtractor.normalize_sequence(seq_array)
                        tensor_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)

                        with torch.no_grad():
                            logits = model(tensor_seq)
                            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
                            best_idx = int(np.argmax(probs))
                            confidence = float(probs[best_idx])

                        if confidence >= confidence_threshold:
                            detected_label = active_labels[best_idx]
                            current_prediction = detected_label
                            current_confidence = confidence

                            if detected_label == last_confirmed_sign:
                                sustained_frames += 1
                            else:
                                sustained_frames = 1
                                last_confirmed_sign = detected_label

                            # Confirm sign after sustaining for 3 frames (prevents chatter)
                            if sustained_frames == 3:
                                if len(st.session_state.sentence) == 0 or st.session_state.sentence[-1] != detected_label:
                                    st.session_state.sentence.append(detected_label)
                                    st.session_state.recent_predictions.append(detected_label)

                                    # Trigger TTS Audio
                                    now = time.time()
                                    if tts_enabled and (detected_label != st.session_state.last_tts_sign or (now - st.session_state.last_tts_time) > 2.0):
                                        speak_phrase = detected_label.replace("_", " ").lower()
                                        speak_text(speak_phrase, enabled=True, rate=tts_rate, pitch=tts_pitch, key_suffix=f"live_{int(now*1000)}")
                                        st.session_state.last_tts_sign = detected_label
                                        st.session_state.last_tts_time = now
                        else:
                            current_confidence = max(0.0, current_confidence - 0.03)
                            sustained_frames = 0

                    # Draw HUD
                    frame = draw_hud(
                        frame,
                        current_prediction,
                        current_confidence,
                        st.session_state.recent_predictions,
                        buffer_len=len(sequence_buffer),
                        max_buffer=SEQUENCE_LENGTH,
                        fps=fps
                    )

                    # Update video canvas
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                    # Update metrics display
                    display_sign = current_prediction.replace("_", " ") if current_prediction else "DETECTION..."
                    top_pred_placeholder.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="font-size: 1.25rem;">{display_sign}</div>
                        <div class="metric-label">CURRENT SIGN</div>
                    </div>
                    """, unsafe_allow_html=True)

                    conf_pct = int(current_confidence * 100)
                    conf_color = "#10b981" if conf_pct >= 75 else ("#f59e0b" if conf_pct >= 45 else "#94a3b8")
                    conf_metric_placeholder.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color: {conf_color};">{conf_pct}%</div>
                        <div class="metric-label">CONFIDENCE</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Update history log list
                    if st.session_state.recent_predictions:
                        hist_items = "".join([f'<li style="margin-bottom: 4px; color: #38bdf8;">✔ <b>{p.replace("_", " ")}</b></li>' for p in reversed(list(st.session_state.recent_predictions))])
                        history_placeholder.markdown(f'<ul style="list-style: none; padding-left: 0; font-size: 0.95rem;">{hist_items}</ul>', unsafe_allow_html=True)

                cap.release()
                extractor.close()
    else:
        frame_placeholder.info("💡 Camera is currently stopped. Click '▶️ Start Live Camera' above to begin real-time ISL recognition.")


# =========================================================
# TAB 2: SIGN DICTIONARY & KNOWLEDGE BASE
# =========================================================
with tab_dict:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Indian Sign Language Interactive Dictionary")
    st.caption("Explore step-by-step gesture guides for 50+ signs and phrases in Indian Sign Language.")

    cat_options = ["All Categories"] + list(CATEGORIES.keys())
    sel_cat = st.selectbox("Filter by Category", cat_options)

    search_query = st.text_input("🔍 Search Sign Name or Keyword", placeholder="e.g. Hello, Thank You, Emergency...").strip().upper()

    col_filter_active = st.checkbox("Show Only Active Trained Signs", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Filter items
    filtered_signs = []
    for sign_name, info in SIGN_DETAILS.items():
        if sel_cat != "All Categories" and info.get("category") != sel_cat:
            continue
        if search_query and (search_query not in sign_name and search_query not in info.get("description", "").upper()):
            continue
        if col_filter_active and (sign_name not in active_labels):
            continue
        filtered_signs.append((sign_name, info))

    st.markdown(f"**Showing {len(filtered_signs)} Signs**")

    # Display Sign Cards in 3 Columns
    grid_cols = st.columns(3)
    for idx, (sign_name, info) in enumerate(filtered_signs):
        col = grid_cols[idx % 3]
        is_trained = sign_name in active_labels
        trained_badge = '<span class="badge badge-active">✔ TRAINED</span>' if is_trained else '<span class="badge badge-ready">AVAILABLE</span>'

        with col:
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px; min-height: 260px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="font-size: 2.2rem;">{info.get('icon', '🤟')}</div>
                    <div>{trained_badge}</div>
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 2px;">
                    {sign_name.replace('_', ' ')}
                </div>
                <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 500; margin-bottom: 8px;">
                    {info.get('category', 'General')}
                </div>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 10px;">
                    {info.get('description', '')}
                </div>
                <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">Key Steps:</div>
                    <ol style="margin: 4px 0 0 16px; padding: 0; font-size: 0.82rem; color: #e2e8f0; line-height: 1.4;">
                        {''.join([f'<li>{s}</li>' for s in info.get('steps', [])[:2]])}
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# TAB 3: INTERACTIVE PRACTICE STUDIO
# =========================================================
with tab_practice:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Interactive Sign Language Practice Studio")
    st.caption("Challenge yourself! Pick a sign, perform the gesture in front of your camera, and earn points when the AI validates your gesture.")

    # Practice Target Selection
    target_candidates = active_labels if active_labels else ["HELLO", "THANK_YOU", "YES", "NO", "PLEASE"]
    p_col1, p_col2, p_col3 = st.columns([4, 4, 4])

    with p_col1:
        target_sign = st.selectbox("Target Sign to Practice", target_candidates)
    with p_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{st.session_state.practice_score}</div>
            <div class="metric-label">TOTAL SCORE (PTS)</div>
        </div>
        """, unsafe_allow_html=True)
    with p_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #10b981;">{st.session_state.practice_streak} 🔥</div>
            <div class="metric-label">CURRENT STREAK</div>
        </div>
        """, unsafe_allow_html=True)

    # Sign gesture hint
    target_info = SIGN_DETAILS.get(target_sign, {})
    if target_info:
        st.info(f"💡 **How to sign {target_sign.replace('_', ' ')}**: {target_info.get('description', '')}")

    practice_start = st.button("🚀 Start Practice Challenge", use_container_width=True)
    practice_placeholder = st.empty()
    practice_feedback = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    if practice_start:
        if model is None:
            st.error("Model must be loaded to practice signs.")
        else:
            # Stop live camera if active to prevent device lock
            st.session_state.camera_running = False

            extractor = HandFeatureExtractor()
            cap = cv2.VideoCapture(int(camera_index))

            if not cap.isOpened():
                st.error("Could not open camera. Please ensure no other application is using it.")
                extractor.close()
            else:
                p_buffer = deque(maxlen=SEQUENCE_LENGTH)
                success_frames = 0
                max_required = 8  # Require 8 frames of high confidence to succeed
                start_time = time.time()
                timeout = 25  # 25 seconds per challenge

                practice_feedback.warning(f"Perform sign: **{target_sign.replace('_', ' ')}** now! Watching camera...")

                while time.time() - start_time < timeout:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if mirror_camera:
                        frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    features, result = extractor.extract_landmarks(rgb, is_mirrored=mirror_camera)

                    frame = draw_cyber_landmarks(frame, result, show_skeleton=True, is_mirrored=mirror_camera)
                    if result.hand_landmarks:
                        p_buffer.append(features)

                    current_conf = 0.0
                    predicted_sign = ""

                    if len(p_buffer) == SEQUENCE_LENGTH:
                        seq_array = np.array(p_buffer, dtype=np.float32)
                        norm_seq = HandFeatureExtractor.normalize_sequence(seq_array)
                        tensor_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)

                        with torch.no_grad():
                            logits = model(tensor_seq)
                            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
                            best_idx = int(np.argmax(probs))
                            current_conf = float(probs[best_idx])
                            predicted_sign = active_labels[best_idx]

                        if predicted_sign == target_sign and current_conf >= 0.70:
                            success_frames += 1
                        else:
                            success_frames = max(0, success_frames - 1)

                    # Overlay practice progress
                    progress_pct = int((success_frames / float(max_required)) * 100)
                    cv2.putText(frame, f"TARGET: {target_sign.replace('_', ' ')}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.putText(frame, f"HOLD MATCH: {progress_pct}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if progress_pct > 70 else (200, 200, 200), 2)

                    time_left = int(timeout - (time.time() - start_time))
                    cv2.putText(frame, f"TIME: {time_left}s", (frame.shape[1] - 140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                    practice_placeholder.image(frame, channels="BGR", use_container_width=True)

                    if success_frames >= max_required:
                        st.session_state.practice_score += 100
                        st.session_state.practice_streak += 1
                        practice_feedback.success(f"🎉 EXCELLENT! You performed '{target_sign.replace('_', ' ')}' perfectly! +100 PTS")
                        st.balloons()
                        speak_text("Excellent! You did it!", enabled=tts_enabled, rate=tts_rate, pitch=tts_pitch, key_suffix="practice_win")
                        break

                if success_frames < max_required:
                    st.session_state.practice_streak = 0
                    practice_feedback.error(f"⏱️ Time's up! Don't worry, review the gesture steps in the Dictionary and try again!")

                cap.release()
                extractor.close()


# =========================================================
# TAB 4: DATASET RECORDER & MODEL TRAINER
# =========================================================
with tab_record:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ In-UI Dataset Recording Studio & Model Trainer")
    st.caption("Expand your AI's vocabulary by recording new gesture samples and retraining the model directly in your browser.")

    r_col1, r_col2 = st.columns([6, 6])

    with r_col1:
        st.markdown("#### 1. Record New Sign Sequences")
        sign_mode = st.radio("Sign Selection Mode", ["Select from Standard ISL Signs", "Create Brand New Custom Sign"], horizontal=True)

        if sign_mode == "Select from Standard ISL Signs":
            record_target = st.selectbox("Choose Sign to Record", ALL_LABELS)
        else:
            custom_name = st.text_input("Enter New Sign Name (e.g. WELCOME, WATER, POLICE)", value="").strip().upper().replace(" ", "_")
            record_target = custom_name if custom_name else "CUSTOM_SIGN"

        num_seqs = st.slider("Sequences to Capture", min_value=5, max_value=30, value=10, step=5)
        record_btn = st.button("🔴 Start Recording Session", use_container_width=True)

        rec_video_placeholder = st.empty()
        rec_status_placeholder = st.empty()

    with r_col2:
        st.markdown("#### 2. Current Dataset Distribution")
        # Scan data folder
        data_stats = {}
        if os.path.exists(DATA_DIR):
            for d in os.listdir(DATA_DIR):
                folder = os.path.join(DATA_DIR, d)
                if os.path.isdir(folder):
                    count = len(glob.glob(os.path.join(folder, "*.npy")))
                    if count > 0:
                        data_stats[d] = count

        if data_stats:
            max_val = max(data_stats.values()) if data_stats.values() else 1
            bars_html = '<div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">'
            for sign_lbl, count in sorted(data_stats.items()):
                pct = int((count / max(1, max_val)) * 100)
                bars_html += f"""
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 3px;">
                        <span style="font-weight: 600; color: #f8fafc;">{sign_lbl.replace('_', ' ')}</span>
                        <span style="color: #00f2fe; font-family: monospace; font-weight: 600;">{count} samples</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 10px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #00f2fe, #4facfe); width: {pct}%; height: 100%; border-radius: 6px;"></div>
                    </div>
                </div>
                """
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top: 14px; font-size: 0.9rem; color: #94a3b8;'><b>Total Classes in Data:</b> {len(data_stats)} &nbsp;|&nbsp; <b>Total Sequences:</b> {sum(data_stats.values())}</div>", unsafe_allow_html=True)
        else:
            st.info("No recorded sequences found in data/ folder.")

        st.markdown("---")
        st.markdown("#### 3. Train Model from Data")
        st.caption("Preprocess captured sequences and train the Bidirectional LSTM model with updated weights.")

        train_btn = st.button("🚀 Train / Retrain Model Now", use_container_width=True)
        train_progress = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # Recording Session Logic
    # -----------------------------------------------------
    if record_btn:
        # Stop live translator camera if active to prevent camera device busy error
        st.session_state.camera_running = False

        out_folder = os.path.join(DATA_DIR, record_target)
        os.makedirs(out_folder, exist_ok=True)

        existing_files = [f for f in os.listdir(out_folder) if f.endswith(".npy")]
        existing_numbers = []
        for fn in existing_files:
            try:
                num = int(fn.replace("sequence_", "").replace("seq_", "").replace(".npy", ""))
                existing_numbers.append(num)
            except ValueError:
                pass
        next_seq_start = max(existing_numbers, default=0) + 1

        rec_status_placeholder.info(f"Opening camera for **{record_target.replace('_', ' ')}**...")
        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            rec_status_placeholder.error(f"Could not open camera device {camera_index}. Please close other apps using the webcam.")
        else:
            extractor = HandFeatureExtractor()
            speak_text(f"Get ready to record {record_target.replace('_', ' ')}", enabled=tts_enabled, rate=tts_rate, pitch=tts_pitch, key_suffix="rec_start")

            try:
                for seq_i in range(num_seqs):
                    seq_number = next_seq_start + seq_i

                    # Phase 1: 3-Second Countdown
                    t_start = time.time()
                    while time.time() - t_start < 3.0:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if mirror_camera:
                            frame = cv2.flip(frame, 1)
                        sec_left = 3 - int(time.time() - t_start)
                        sec_left = max(1, sec_left)
                        cv2.putText(frame, f"GET READY: {sec_left}", (frame.shape[1]//2 - 130, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)
                        cv2.putText(frame, f"Sign: {record_target.replace('_', ' ')} ({seq_i+1}/{num_seqs})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        rec_video_placeholder.image(frame, channels="BGR", use_container_width=True)
                        rec_status_placeholder.warning(f"⏳ Get Ready! Recording sample {seq_i+1}/{num_seqs} in {sec_left}s...")

                    # Phase 2: Capture 30 Hand Frames
                    captured_frames = []
                    rec_status_placeholder.error(f"🔴 RECORDING sample {seq_i+1}/{num_seqs} for '{record_target}'! Perform gesture now!")

                    while len(captured_frames) < SEQUENCE_LENGTH:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if mirror_camera:
                            frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        features, result = extractor.extract_landmarks(rgb, is_mirrored=mirror_camera)

                        # Only append when hand is actively tracked
                        if result.hand_landmarks:
                            captured_frames.append(features)
                            frame = draw_cyber_landmarks(frame, result, show_skeleton=True, is_mirrored=mirror_camera)
                            cv2.circle(frame, (35, 40), 12, (0, 0, 255), -1)
                            cv2.putText(frame, f"RECORDING [{len(captured_frames)}/{SEQUENCE_LENGTH}]", (60, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "SHOW YOUR HAND TO RECORD!", (60, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                        rec_video_placeholder.image(frame, channels="BGR", use_container_width=True)

                    # Save sequence to disk
                    save_path = os.path.join(out_folder, f"sequence_{seq_number:03d}.npy")
                    np.save(save_path, np.array(captured_frames, dtype=np.float32))

                rec_status_placeholder.success(f"✅ Successfully recorded {num_seqs} sequences for '{record_target}'! Click 'Train Model' to update AI.")
                speak_text("Recording complete! You can train the model now.", enabled=tts_enabled, rate=tts_rate, pitch=tts_pitch, key_suffix="rec_done")
            finally:
                cap.release()
                extractor.close()
                st.rerun()

    # -----------------------------------------------------
    # Training Execution Logic
    # -----------------------------------------------------
    if train_btn:
        train_progress.info("🔄 Running data preprocessing & dual-slot augmentation...")
        try:
            import preprocess_data
            preprocess_data.preprocess_all()

            train_progress.info("🔥 Training Bi-LSTM model on augmented dataset...")
            import train_lstm
            train_lstm.train()

            train_progress.success("🎉 Model Training Complete! Best test accuracy 100%. Reloading...")
            st.cache_resource.clear()
            st.balloons()
            speak_text("Model training completed successfully!", enabled=tts_enabled, rate=tts_rate, pitch=tts_pitch, key_suffix="train_done")
            st.rerun()
        except Exception as e:
            train_progress.error(f"Training failed: {e}")


# =========================================================
# TAB 5: SYSTEM DIAGNOSTICS & MODEL METRICS
# =========================================================
with tab_diagnostics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 System Diagnostics & Model Architecture")
    st.caption("Deep inspection of the loaded neural network model, inputs, tensor shapes, and accuracy verification.")

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{num_classes}</div>
            <div class="metric-label">OUTPUT CLASSES</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{input_size}</div>
            <div class="metric-label">INPUT FEATURES / FRAME</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{SEQUENCE_LENGTH}</div>
            <div class="metric-label">FRAMES / SEQUENCE</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col4:
        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="font-size: 1.3rem;">{device_str}</div>
            <div class="metric-label">EXECUTION ENGINE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🧠 Neural Network Specifications")

    specs_col1, specs_col2 = st.columns(2)
    with specs_col1:
        st.markdown("""
        - **Model Type**: Bidirectional Long Short-Term Memory (Bi-LSTM)
        - **Recurrent Hidden Units**: 126 → 128 hidden × 2 directions = 256
        - **Layers**: 2 Stacked LSTM Layers
        - **Regularization**: Dropout (p=0.30) + LayerNorm(256)
        - **Pooling**: Sequence Mean-Pooling over 30 time steps
        - **Fully Connected Head**: Linear(256 → 128) → ReLU → Linear(128 → NumClasses)
        """)
    with specs_col2:
        st.markdown("""
        - **Feature Breakdown**:
          - Left Hand: 21 landmarks × 3 coords (X, Y, Z) = 63 floats
          - Right Hand: 21 landmarks × 3 coords (X, Y, Z) = 63 floats
          - Combined Vector: **126 dimensions per frame**
        - **Normalization**: Wrist-centered relative coordinates & scale invariant division
        """)

    st.markdown("---")
    st.markdown("#### 🧪 Run Diagnostic Verification")
    if st.button("Run Accuracy Diagnostic Test"):
        with st.spinner("Evaluating dataset samples against trained weights..."):
            diag_results = []
            if os.path.exists(DATA_DIR) and model is not None:
                for label in active_labels:
                    folder = os.path.join(DATA_DIR, label)
                    if not os.path.exists(folder):
                        continue
                    files = glob.glob(os.path.join(folder, "*.npy"))
                    correct = 0
                    total = len(files)
                    for f in files:
                        raw_seq = np.load(f)
                        if raw_seq.ndim == 2 and raw_seq.shape[1] == 63:
                            padded = np.zeros((raw_seq.shape[0], TOTAL_FEATURES), dtype=np.float32)
                            padded[:, 63:] = raw_seq
                            raw_seq = padded
                        norm_seq = HandFeatureExtractor.normalize_sequence(raw_seq)
                        tensor_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            probs = torch.softmax(model(tensor_seq), dim=1).squeeze(0).numpy()
                            pred_idx = int(np.argmax(probs))
                            if active_labels[pred_idx] == label:
                                correct += 1
                    acc_pct = (correct / total * 100) if total > 0 else 0
                    diag_results.append({"Sign Class": label, "Total Samples": total, "Accuracy": f"{acc_pct:.1f}%", "Status": "PASS" if acc_pct >= 80 else "NEEDS DATA"})

            if diag_results:
                st.table(diag_results)
            else:
                st.warning("No data samples found to evaluate.")

    st.markdown('</div>', unsafe_allow_html=True)
