from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, Any, List
from backend.config import EMOTION_MODEL

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral"
]

WORKPLACE_EMOTIONS = {
    "joy": "happy",
    "sadness": "disengaged",
    "anger": "frustrated",
    "fear": "anxious",
    "surprise": "confused",
    "disgust": "frustrated",
    "neutral": "satisfied"
}


class EmotionDetector:
    def __init__(self, model_name: str = EMOTION_MODEL):
        self.model_name = model_name
        self.classifier = None
        self.device = 0 if torch.cuda.is_available() else -1

    def load_model(self):
        if self.classifier is None:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                return_all_scores=True
            )
            logger.info(f"Emotion model {self.model_name} loaded successfully")

    def analyze(self, text: str) -> Dict[str, Any]:
        self.load_model()
        try:
            results = self.classifier(text)
            
            scores = {r["label"]: r["score"] for r in results[0]}
            
            max_emotion = max(scores, key=scores.get)
            confidence = scores[max_emotion]
            
            workplace_emotion = WORKPLACE_EMOTIONS.get(max_emotion, max_emotion)
            
            return {
                "emotion": workplace_emotion,
                "confidence": float(confidence),
                "raw_emotion": max_emotion,
                "raw_scores": scores,
                "all_emotions": {WORKPLACE_EMOTIONS.get(k, k): float(v) for k, v in scores.items()}
            }
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "raw_emotion": "neutral",
                "raw_scores": {},
                "all_emotions": {}
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        self.load_model()
        try:
            results = self.classifier(texts)
            return [self._parse_result(r) for r in results]
        except Exception as e:
            logger.error(f"Batch emotion detection failed: {e}")
            return [{"emotion": "neutral", "confidence": 0.5} for _ in texts]

    def _parse_result(self, result: list) -> Dict[str, Any]:
        scores = {r["label"]: r["score"] for r in result}
        
        max_emotion = max(scores, key=scores.get)
        confidence = scores[max_emotion]
        
        workplace_emotion = WORKPLACE_EMOTIONS.get(max_emotion, max_emotion)
        
        return {
            "emotion": workplace_emotion,
            "confidence": float(confidence),
            "raw_emotion": max_emotion,
            "raw_scores": scores,
            "all_emotions": {WORKPLACE_EMOTIONS.get(k, k): float(v) for k, v in scores.items()}
        }


def get_emotion_detector(model_name: str = EMOTION_MODEL) -> EmotionDetector:
    return EmotionDetector(model_name)