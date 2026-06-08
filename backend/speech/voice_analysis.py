import librosa
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class VoiceEmotionAnalyzer:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(y) == 0:
                return self._empty_result()
            
            features = {}
            
            features["speaking_speed"] = self._calculate_speaking_speed(y, sr)
            features["pitch_variation"] = self._calculate_pitch_variation(y, sr)
            features["energy_level"] = self._calculate_energy_level(y)
            features["hesitation_count"] = self._count_hesitations(y, sr)
            features["pause_duration"] = self._calculate_pause_duration(y, sr)
            features["voice_confidence"] = self._calculate_voice_confidence(features)
            features["stress_level"] = self._calculate_stress_level(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return self._empty_result()

    def _calculate_speaking_speed(self, y: np.ndarray, sr: int) -> float:
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            duration = len(y) / sr
            if duration > 0:
                return len(onset_frames) / duration
            return 0.0
        except Exception:
            return 0.0

    def _calculate_pitch_variation(self, y: np.ndarray, sr: int) -> float:
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_voiced = f0[voiced_flag]
            if len(f0_voiced) > 0:
                return float(np.std(f0_voiced) / np.mean(f0_voiced))
            return 0.0
        except Exception:
            return 0.0

    def _calculate_energy_level(self, y: np.ndarray) -> float:
        try:
            rms = librosa.feature.rms(y=y)[0]
            return float(np.mean(rms))
        except Exception:
            return 0.0

    def _count_hesitations(self, y: np.ndarray, sr: int) -> int:
        try:
            intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
            hesitation_count = 0
            for i in range(1, len(intervals)):
                gap = (intervals[i][0] - intervals[i-1][1]) / sr
                if 0.3 < gap < 2.0:
                    hesitation_count += 1
            return hesitation_count
        except Exception:
            return 0

    def _calculate_pause_duration(self, y: np.ndarray, sr: int) -> float:
        try:
            intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
            pauses = []
            for i in range(1, len(intervals)):
                gap = (intervals[i][0] - intervals[i-1][1]) / sr
                if gap > 0.3:
                    pauses.append(gap)
            return float(np.mean(pauses)) if pauses else 0.0
        except Exception:
            return 0.0

    def _calculate_voice_confidence(self, features: Dict[str, float]) -> float:
        confidence = 1.0
        
        if features["hesitation_count"] > 3:
            confidence -= 0.15
        if features["pause_duration"] > 1.5:
            confidence -= 0.1
        if features["pitch_variation"] < 0.05:
            confidence -= 0.1
        if features["energy_level"] < 0.01:
            confidence -= 0.15
        if features["speaking_speed"] < 1.0:
            confidence -= 0.1
            
        return max(0.0, min(1.0, confidence))

    def _calculate_stress_level(self, features: Dict[str, float]) -> float:
        stress = 0.0
        
        if features["speaking_speed"] > 4.0:
            stress += 0.2
        elif features["speaking_speed"] < 1.0:
            stress += 0.15
            
        if features["pitch_variation"] > 0.3:
            stress += 0.2
        elif features["pitch_variation"] < 0.05:
            stress += 0.1
            
        if features["hesitation_count"] > 3:
            stress += 0.2
            
        if features["pause_duration"] > 2.0:
            stress += 0.15
            
        if features["energy_level"] > 0.1:
            stress += 0.1
        elif features["energy_level"] < 0.005:
            stress += 0.1
            
        return min(1.0, stress)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "speaking_speed": 0.0,
            "pitch_variation": 0.0,
            "energy_level": 0.0,
            "hesitation_count": 0,
            "pause_duration": 0.0,
            "voice_confidence": 0.5,
            "stress_level": 0.0
        }


def get_voice_analyzer(sample_rate: int = 22050) -> VoiceEmotionAnalyzer:
    return VoiceEmotionAnalyzer(sample_rate)