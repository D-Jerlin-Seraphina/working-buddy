from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, Any
from backend.config import SENTIMENT_MODEL

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, model_name: str = SENTIMENT_MODEL):
        self.model_name = model_name
        self.classifier = None
        self.device = 0 if torch.cuda.is_available() else -1

    def load_model(self):
        if self.classifier is None:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                return_all_scores=True
            )
            logger.info(f"Sentiment model {self.model_name} loaded successfully")

    def analyze(self, text: str) -> Dict[str, Any]:
        self.load_model()
        try:
            results = self.classifier(text)
            
            scores = {r["label"]: r["score"] for r in results[0]}
            
            if "POSITIVE" in scores and "NEGATIVE" in scores:
                sentiment = "positive" if scores["POSITIVE"] > scores["NEGATIVE"] else "negative"
                confidence = max(scores["POSITIVE"], scores["NEGATIVE"])
            elif "LABEL_1" in scores and "LABEL_0" in scores:
                sentiment = "positive" if scores["LABEL_1"] > scores["LABEL_0"] else "negative"
                confidence = max(scores["LABEL_1"], scores["LABEL_0"])
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            sentiment_score = scores.get("POSITIVE", scores.get("LABEL_1", 0.5))
            if sentiment == "negative":
                sentiment_score = 1 - sentiment_score
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "sentiment_score": float(sentiment_score),
                "raw_scores": scores
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "sentiment_score": 0.5,
                "raw_scores": {}
            }

    def analyze_batch(self, texts: list) -> list:
        self.load_model()
        try:
            results = self.classifier(texts)
            return [self._parse_result(r) for r in results]
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            return [{"sentiment": "neutral", "confidence": 0.5, "sentiment_score": 0.5} for _ in texts]

    def _parse_result(self, result: list) -> Dict[str, Any]:
        scores = {r["label"]: r["score"] for r in result}
        
        if "POSITIVE" in scores and "NEGATIVE" in scores:
            sentiment = "positive" if scores["POSITIVE"] > scores["NEGATIVE"] else "negative"
            confidence = max(scores["POSITIVE"], scores["NEGATIVE"])
        elif "LABEL_1" in scores and "LABEL_0" in scores:
            sentiment = "positive" if scores["LABEL_1"] > scores["LABEL_0"] else "negative"
            confidence = max(scores["LABEL_1"], scores["LABEL_0"])
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        sentiment_score = scores.get("POSITIVE", scores.get("LABEL_1", 0.5))
        if sentiment == "negative":
            sentiment_score = 1 - sentiment_score
            
        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "sentiment_score": float(sentiment_score),
            "raw_scores": scores
        }


def get_sentiment_analyzer(model_name: str = SENTIMENT_MODEL) -> SentimentAnalyzer:
    return SentimentAnalyzer(model_name)