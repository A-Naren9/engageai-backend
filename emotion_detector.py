import cv2
import numpy as np
import logging

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


class EmotionDetector:
    def __init__(self):
        self.detector = None
        self._initialize()

    def _initialize(self):
        # Prefer MTCNN (CNN-based, more accurate) over Haar cascade
        for mtcnn in (True, False):
            try:
                from fer.fer import FER
                self.detector = FER(mtcnn=mtcnn)
                mode = "MTCNN" if mtcnn else "Haar cascade"
                logger.info(f"FER detector initialized ({mode})")
                return
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"FER(mtcnn={mtcnn}) failed: {e}")
        logger.warning("FER unavailable — emotion detection disabled")

    def detect(self, frame: np.ndarray) -> dict:
        if self.detector is None:
            return self._empty_result()

        try:
            # Ensure minimum size — small frames cause Haar/MTCNN to miss faces
            h, w = frame.shape[:2]
            if w < 480:
                scale = 480 / w
                frame = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_LINEAR)

            # FER / MTCNN expect RGB; OpenCV delivers BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.detector.detect_emotions(rgb)
            if results:
                emotions = results[0]["emotions"]
                dominant = max(emotions, key=emotions.get)
                scores = {k: round(v * 100, 1) for k, v in emotions.items()}
                confidence = round(emotions[dominant] * 100, 1)

                return {
                    "dominant": dominant,
                    "emoji": EMOTION_EMOJIS.get(dominant, "❓"),
                    "hint": ENGAGEMENT_HINTS.get(dominant, ""),
                    "scores": scores,
                    "face_detected": True,
                    "confidence": confidence,
                }
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")

        return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "dominant": "unknown",
            "emoji": "❓",
            "hint": "Face not detected",
            "scores": {},
            "face_detected": False,
            "confidence": 0,
        }
