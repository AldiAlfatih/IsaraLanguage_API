import os
import io
import re
import json
import base64
import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
import cv2
from flask import Flask, request, jsonify

import mediapipe as mp
import tensorflow as tf
from model_loader import load_bisindo_model
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("bisindo_api")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

SEQUENCE_LENGTH = 30
POSE_LANDMARKS = 33  
HAND_LANDMARKS = 21  
SKELETON_FEATURES_LEN = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2  
HAND_SHAPE_SINGLE_LEN = 1280  
HAND_SHAPE_BOTH_LEN = HAND_SHAPE_SINGLE_LEN * 2  
HYBRID_FEATURE_LEN = SKELETON_FEATURES_LEN + HAND_SHAPE_BOTH_LEN  
MOBILENET_INPUT_SIZE = (224, 224)
MODEL_PATH = os.environ.get("BISINDO_MODEL_PATH", "bisindo_hybrid_model.h5")

try:
    LOGGER.info("Loading Bi-LSTM Keras model from %s", MODEL_PATH)
    BISINDO_MODEL = load_bisindo_model(MODEL_PATH)
    LOGGER.info("Bi-LSTM model loaded.")
except Exception as e:
    LOGGER.exception("Failed to load model '%s'", MODEL_PATH)
    raise

mp_holistic = mp.solutions.holistic
HOLISTIC = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
HOLISTIC_LOCK = threading.Lock()

FEATURE_EXTRACTOR = MobileNetV2(
    input_shape=(MOBILENET_INPUT_SIZE[1], MOBILENET_INPUT_SIZE[0], 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)

CLASS_NAMES: Optional[List[str]] = None
for fname in ("class_names.json", "labels.json"):
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "classes" in data:
                    CLASS_NAMES = list(data["classes"]) 
                elif isinstance(data, list):
                    CLASS_NAMES = list(data)
                LOGGER.info("Loaded class names from %s", fname)
                break
        except Exception:
            LOGGER.warning("Failed to parse %s for class names; falling back to index labels.", fname)

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False


def _strip_base64_header(b64: str) -> str:
    """Remove data URI prefix if present."""
    if not isinstance(b64, str):
        return ""
    if "base64," in b64:
        return b64.split("base64,", 1)[1]
    return b64


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decode a base64 string to an OpenCV BGR image array.

    Returns None if decoding fails.
    """
    try:
        clean = _strip_base64_header(b64_string.strip())
        img_bytes = base64.b64decode(clean, validate=True)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img  # BGR
    except Exception:
        return None


def landmarks_to_bbox(hand_landmarks, image_shape: Tuple[int, int], margin: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
    """Compute a bounding box around hand landmarks (normalized coords) with margin.

    Returns (x_min, y_min, x_max, y_max) in pixel coordinates or None if invalid.
    """
    if hand_landmarks is None:
        return None

    h, w = image_shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(0.0, min(xs)) * w
    x_max = min(1.0, max(xs)) * w
    y_min = max(0.0, min(ys)) * h
    y_max = min(1.0, max(ys)) * h

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None

    x_min = int(max(0, x_min - margin * bw))
    y_min = int(max(0, y_min - margin * bh))
    x_max = int(min(w - 1, x_max + margin * bw))
    y_max = int(min(h - 1, y_max + margin * bh))

    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def extract_hand_shape_feature_single(frame_bgr: np.ndarray, hand_landmarks) -> np.ndarray:
    """Extract a 1280-D hand-shape feature from a cropped hand ROI using MobileNetV2.

    If hand landmarks are missing or ROI invalid, returns zeros(1280).
    """
    if hand_landmarks is None or frame_bgr is None:
        return np.zeros((HAND_SHAPE_SINGLE_LEN,), dtype=np.float32)

    h, w, _ = frame_bgr.shape
    bbox = landmarks_to_bbox(hand_landmarks, (h, w))
    if bbox is None:
        return np.zeros((HAND_SHAPE_SINGLE_LEN,), dtype=np.float32)

    x_min, y_min, x_max, y_max = bbox
    roi = frame_bgr[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return np.zeros((HAND_SHAPE_SINGLE_LEN,), dtype=np.float32)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, MOBILENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    roi_arr = roi_resized.astype(np.float32)
    roi_arr = mobilenet_preprocess(roi_arr)
    roi_arr = np.expand_dims(roi_arr, axis=0)  

    try:
        feats = FEATURE_EXTRACTOR.predict(roi_arr, verbose=0)  
        feats = feats.reshape(-1).astype(np.float32)
        if feats.size != HAND_SHAPE_SINGLE_LEN:
            # Safety check
            vec = np.zeros((HAND_SHAPE_SINGLE_LEN,), dtype=np.float32)
            vec[: min(HAND_SHAPE_SINGLE_LEN, feats.size)] = feats[:HAND_SHAPE_SINGLE_LEN]
            return vec
        return feats
    except Exception:
        return np.zeros((HAND_SHAPE_SINGLE_LEN,), dtype=np.float32)


def extract_hand_shape_features(frame_bgr: np.ndarray, hand_landmarks) -> np.ndarray:
    """Wrapper with exact training name: returns 1280-D feature for a single hand.

    This simply delegates to `extract_hand_shape_feature_single()` to ensure the
    function name matches the training-time code, preventing feature mismatch.
    """
    return extract_hand_shape_feature_single(frame_bgr, hand_landmarks)


def extract_skeleton_features(results) -> np.ndarray:
    """Extract skeleton feature vector (length 258) from MediaPipe Holistic results.

    Layout:
      - Pose: 33 landmarks x (x, y, z, visibility) = 132
      - Left hand: 21 landmarks x (x, y, z) = 63
      - Right hand: 21 landmarks x (x, y, z) = 63
    Total = 258
    """
    features: List[float] = []

    # Pose (x, y, z, visibility)
    if results.pose_landmarks is not None:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0.0] * (POSE_LANDMARKS * 4))

    # Left hand (x, y, z)
    if results.left_hand_landmarks is not None:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * (HAND_LANDMARKS * 3))

    # Right hand (x, y, z)
    if results.right_hand_landmarks is not None:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * (HAND_LANDMARKS * 3))

    arr = np.asarray(features, dtype=np.float32)
    if arr.size != SKELETON_FEATURES_LEN:
        # Safety pad/trim
        vec = np.zeros((SKELETON_FEATURES_LEN,), dtype=np.float32)
        vec[: min(SKELETON_FEATURES_LEN, arr.size)] = arr[:SKELETON_FEATURES_LEN]
        return vec
    return arr


def build_hybrid_feature_vector(frame_bgr: np.ndarray, results) -> np.ndarray:
    """Combine skeleton (258) and both-hand shape features (2560) into one vector (2818)."""
    skel = extract_skeleton_features(results)  # (258,)

    left_vec = extract_hand_shape_features(frame_bgr, results.left_hand_landmarks)
    right_vec = extract_hand_shape_features(frame_bgr, results.right_hand_landmarks)

    hand_vec = np.concatenate([left_vec, right_vec], axis=0)  # (2560,)
    hybrid = np.concatenate([skel, hand_vec], axis=0).astype(np.float32)

    # Strict length enforcement
    if hybrid.size != HYBRID_FEATURE_LEN:
        vec = np.zeros((HYBRID_FEATURE_LEN,), dtype=np.float32)
        vec[: min(HYBRID_FEATURE_LEN, hybrid.size)] = hybrid[:HYBRID_FEATURE_LEN]
        return vec
    return hybrid


def build_sequence_features(frames_b64: List[str]) -> Optional[np.ndarray]:
    """Convert 30 base64 frames to a (1, 30, 2818) numpy array of hybrid features."""
    if len(frames_b64) != SEQUENCE_LENGTH:
        return None

    sequence_features: List[np.ndarray] = []

    for idx, b64 in enumerate(frames_b64):
        frame_bgr = decode_base64_image(b64)
        if frame_bgr is None:
            LOGGER.warning("Failed to decode frame index %d", idx)
            return None

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        with HOLISTIC_LOCK:
            results = HOLISTIC.process(img_rgb)

        hybrid_vec = build_hybrid_feature_vector(frame_bgr, results)
        sequence_features.append(hybrid_vec)

    seq_arr = np.stack(sequence_features, axis=0).astype(np.float32)  
    seq_arr = np.expand_dims(seq_arr, axis=0)  
    return seq_arr


def index_to_label(idx: int) -> str:
    if CLASS_NAMES is not None and 0 <= idx < len(CLASS_NAMES):
        return str(CLASS_NAMES[idx])
    return f"class_{idx}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None or "frames" not in data:
            return jsonify({"error": "Invalid request. JSON body with key 'frames' is required."}), 400

        frames = data.get("frames")
        if not isinstance(frames, list):
            return jsonify({"error": "'frames' must be a list of 30 base64-encoded images."}), 400
        if len(frames) != SEQUENCE_LENGTH:
            return jsonify({"error": f"'frames' must have exactly {SEQUENCE_LENGTH} elements."}), 400

        seq = build_sequence_features(frames)
        if seq is None:
            return jsonify({"error": "Failed to decode/process one or more frames."}), 400

        preds = BISINDO_MODEL.predict(seq, verbose=0)  
        preds = np.asarray(preds).reshape(-1)
        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = index_to_label(pred_idx)

        return jsonify({"prediction": label, "confidence": round(confidence, 6)})
    except Exception as e:
        LOGGER.exception("/predict failed: %s", str(e))
        return jsonify({"error": "Internal server error."}), 500


@app.route("/", methods=["GET"]) 
def health():
    return jsonify({
        "status": "ok",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_length": HYBRID_FEATURE_LEN,
        "model_loaded": True,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
