import cv2
import numpy as np
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

EMOTION_EMOJIS = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "neutral":  "😐",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "unknown":  "❓",
}

ENGAGEMENT_HINTS = {
    "happy":    "Interested & positive",
    "surprise": "Highly attentive",
    "neutral":  "Listening / focused",
    "sad":      "Disengaged or tired",
    "fear":     "Anxious or overwhelmed",
    "angry":    "Frustrated",
    "disgust":  "Negative reaction",
    "unknown":  "Face not detected",
}

# FER+ model outputs 8 classes; map to 7 display emotions
_FERPLUS_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
_EMOTION_MAP = {
    'neutral':   'neutral',
    'happiness': 'happy',
    'surprise':  'surprise',
    'sadness':   'sad',
    'anger':     'angry',
    'disgust':   'disgust',
    'fear':      'fear',
    'contempt':  'neutral',   # merge contempt → neutral
}

_MODEL_URL  = (
    "https://github.com/onnx/models/raw/main/validated/"
    "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_ferplus.onnx")


class EmotionDetector:
    def __init__(self):
        self.session      = None
        self.face_cascade = None
        self._initialize()

    def _initialize(self):
        try:
            import onnxruntime as ort

            if not os.path.exists(_MODEL_PATH):
                logger.info("Downloading FER+ ONNX emotion model (~2 MB)…")
                urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
                logger.info("Model downloaded.")

            self.session = ort.InferenceSession(
                _MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            logger.info("EmotionDetector ready (ONNX Runtime + Haar cascade)")
        except Exception as e:
            logger.warning(f"EmotionDetector init failed: {e}")

    def detect(self, frame: np.ndarray) -> dict:
        if self.session is None or self.face_cascade is None:
            return self._empty_result()

        try:
            # Upscale small frames so Haar cascade can find faces
            h, w = frame.shape[:2]
            if w < 480:
                scale = 480 / w
                frame = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_LINEAR)

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )

            if len(faces) == 0:
                return self._empty_result()

            # Use the largest detected face
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face = gray[y : y + fh, x : x + fw]
            face = cv2.resize(face, (64, 64)).astype(np.float32)
            face = face.reshape(1, 1, 64, 64)   # NCHW

            input_name = self.session.get_inputs()[0].name
            raw_scores = self.session.run(None, {input_name: face})[0][0]

            # Softmax
            exp_s = np.exp(raw_scores - raw_scores.max())
            probs = exp_s / exp_s.sum()

            dominant_idx = int(np.argmax(probs))
            dominant_raw = _FERPLUS_LABELS[dominant_idx]
            dominant     = _EMOTION_MAP[dominant_raw]
            confidence   = round(float(probs[dominant_idx]) * 100, 1)

            # Merge 8-class scores into 7 display emotions
            scores: dict[str, float] = {}
            for i, lbl in enumerate(_FERPLUS_LABELS):
                key = _EMOTION_MAP[lbl]
                scores[key] = round(scores.get(key, 0.0) + float(probs[i]) * 100, 1)

            return {
                "dominant":     dominant,
                "emoji":        EMOTION_EMOJIS.get(dominant, "❓"),
                "hint":         ENGAGEMENT_HINTS.get(dominant, ""),
                "scores":       scores,
                "face_detected": True,
                "confidence":   confidence,
            }

        except Exception as e:
            logger.error(f"Emotion detection error: {e}")

        return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "dominant":      "unknown",
            "emoji":         "❓",
            "hint":          "Face not detected",
            "scores":        {},
            "face_detected": False,
            "confidence":    0,
        }
