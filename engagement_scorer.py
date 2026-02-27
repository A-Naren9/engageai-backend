from collections import deque
import logging

logger = logging.getLogger(__name__)

# Emotion → base engagement score (0–100)
EMOTION_SCORES = {
    "happy":    88,
    "surprise": 80,
    "neutral":  62,
    "fear":     42,
    "angry":    40,
    "sad":      28,
    "disgust":  22,
    "unknown":  50,
}

# Speech pace → audio engagement score
PACE_SCORES = {
    "moderate": 82,
    "fast":     68,
    "slow":     32,
    "silent":   50,
    "unknown":  50,
}

LEVELS = [
    (80, "Highly Engaged",      "#22c55e", "🔥"),
    (60, "Moderately Engaged",  "#84cc16", "✅"),
    (40, "Low Engagement",      "#f59e0b", "⚠️"),
    (0,  "Disengaged",          "#ef4444", "😴"),
]

HISTORY_LEN = 60   # keep 60 data-points (~30 s at 500 ms/frame)


class EngagementScorer:
    def __init__(self):
        self.history: deque = deque(maxlen=HISTORY_LEN)

    def score(self, data: dict) -> dict:
        emotion_score = 50
        audio_score = 50
        confidence_weight = 1.0

        # ── Video signal ──────────────────────────────────────────────
        emotion_data = data.get("emotion", {})
        if emotion_data.get("face_detected"):
            dominant = emotion_data.get("dominant", "unknown")
            emotion_score = EMOTION_SCORES.get(dominant, 50)
            # Scale by detection confidence (0–100 → 0.5–1.0)
            confidence = emotion_data.get("confidence", 50)
            confidence_weight = 0.5 + confidence / 200

        # ── Audio signal ───────────────────────────────────────────────
        audio_data = data.get("audio", {})
        pace = audio_data.get("pace", "unknown")
        audio_score = PACE_SCORES.get(pace, 50)

        # ── Fuse (60 % video, 40 % audio) ─────────────────────────────
        raw = 0.6 * emotion_score + 0.4 * audio_score
        final = round(raw * confidence_weight)
        final = max(0, min(100, final))

        self.history.append(final)

        level, color, icon = self._classify(final)
        trend = self._trend()

        return {
            "score": final,
            "level": level,
            "color": color,
            "icon": icon,
            "trend": trend,
            "history": list(self.history),
            "alert": final < 40,
        }

    @staticmethod
    def _classify(score: int):
        for threshold, label, color, icon in LEVELS:
            if score >= threshold:
                return label, color, icon
        return "Disengaged", "#ef4444", "😴"

    def _trend(self) -> str:
        if len(self.history) < 5:
            return "stable"
        recent = list(self.history)[-5:]
        delta = recent[-1] - recent[0]
        if delta > 5:
            return "rising"
        elif delta < -5:
            return "falling"
        return "stable"
