import numpy as np
import logging

logger = logging.getLogger(__name__)

SILENCE_THRESHOLD = 0.005   # RMS below this → silent
SLOW_ZCR_THRESHOLD = 0.05
FAST_ZCR_THRESHOLD = 0.15


def _try_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        return None


class AudioAnalyzer:
    def __init__(self):
        self.librosa = _try_librosa()
        if self.librosa:
            logger.info("Librosa audio analyzer initialized")
        else:
            logger.warning("Librosa unavailable — audio analysis will use basic stats")

    def analyze(self, audio_data: list, sample_rate: int = 16000) -> dict:
        try:
            y = np.array(audio_data, dtype=np.float32)
            if len(y) == 0:
                return self._empty_result()

            rms = float(np.sqrt(np.mean(y ** 2)))

            if rms < SILENCE_THRESHOLD:
                return {
                    "rms": round(rms, 4),
                    "zcr": 0.0,
                    "tempo": 0.0,
                    "spectral_centroid": 0.0,
                    "pace": "silent",
                    "pace_label": "Silent / Listening",
                    "pace_color": "#64748b",
                    "is_speaking": False,
                    "energy_pct": 0,
                }

            if self.librosa:
                result = self._librosa_analyze(y, sample_rate, rms)
            else:
                result = self._basic_analyze(y, rms)

            return result

        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return self._empty_result()

    def _librosa_analyze(self, y: np.ndarray, sr: int, rms: float) -> dict:
        lib = self.librosa

        zcr = float(np.mean(lib.feature.zero_crossing_rate(y)))
        spec_centroid = float(np.mean(lib.feature.spectral_centroid(y=y, sr=sr)))

        try:
            tempo, _ = lib.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except Exception:
            tempo = 0.0

        pace, label, color = self._classify_pace(zcr, rms)
        energy_pct = min(100, int(rms * 1000))

        return {
            "rms": round(rms, 4),
            "zcr": round(zcr, 4),
            "tempo": round(tempo, 1),
            "spectral_centroid": round(spec_centroid, 1),
            "pace": pace,
            "pace_label": label,
            "pace_color": color,
            "is_speaking": True,
            "energy_pct": energy_pct,
        }

    def _basic_analyze(self, y: np.ndarray, rms: float) -> dict:
        signs = np.sign(y)
        zcr = float(np.mean(np.abs(np.diff(signs)) / 2))
        pace, label, color = self._classify_pace(zcr, rms)
        energy_pct = min(100, int(rms * 1000))

        return {
            "rms": round(rms, 4),
            "zcr": round(zcr, 4),
            "tempo": 0.0,
            "spectral_centroid": 0.0,
            "pace": pace,
            "pace_label": label,
            "pace_color": color,
            "is_speaking": True,
            "energy_pct": energy_pct,
        }

    @staticmethod
    def _classify_pace(zcr: float, rms: float):
        if zcr > FAST_ZCR_THRESHOLD:
            return "fast", "Fast / Excited", "#f59e0b"
        elif zcr < SLOW_ZCR_THRESHOLD:
            return "slow", "Slow / Low Energy", "#ef4444"
        else:
            return "moderate", "Moderate / Attentive", "#22c55e"

    @staticmethod
    def _empty_result() -> dict:
        return {
            "rms": 0.0,
            "zcr": 0.0,
            "tempo": 0.0,
            "spectral_centroid": 0.0,
            "pace": "unknown",
            "pace_label": "No audio",
            "pace_color": "#64748b",
            "is_speaking": False,
            "energy_pct": 0,
        }
