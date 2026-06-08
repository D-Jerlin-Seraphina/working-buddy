from transformers import pipeline
import logging
import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from backend.config import VECTORIZER_PATH, LABEL_ENCODER_PATH

logger = logging.getLogger(__name__)

PROBLEM_CATEGORIES = {
    "training_issue": [
        "tool", "software", "system", "platform", "application", "crm", "erp",
        "training", "learn", "tutorial", "documentation", "guide", "manual",
        "difficult", "confusing", "complex", "complicated", "hard to use",
        "don't understand", "not sure how", "unclear", "overwhelming"
    ],
    "manager_issue": [
        "manager", "supervisor", "lead", "boss", "management",
        "communication", "feedback", "guidance", "direction", "support",
        "unavailable", "busy", "ignored", "dismissive", "micromanage",
        "expectations", "unclear goals", "no direction"
    ],
    "team_issue": [
        "team", "colleague", "coworker", "teammate", "peer",
        "collaboration", "communication", "isolated", "alone", "lonely",
        "not included", "left out", "clique", "unwelcoming", "unfriendly",
        "nobody helps", "no one replies", "silent"
    ],
    "workload_issue": [
        "workload", "work load", "tasks", "assignments", "projects",
        "overwhelmed", "swamped", "drowning", "too much", "excessive",
        "burnout", "exhausted", "tired", "stress", "pressure",
        "deadline", "rush", "crunch", "overtime", "long hours"
    ],
    "career_growth_issue": [
        "growth", "career", "development", "promotion", "advancement",
        "opportunity", "learning", "skill", "challenge", "stagnant",
        "stuck", "dead end", "no future", "nowhere to go",
        "utilized", "potential", "wasted", "bored"
    ],
    "compensation_concern": [
        "salary", "pay", "compensation", "benefits", "bonus",
        "raise", "underpaid", "market rate", "fair", "equity",
        "financial", "money", "budget"
    ],
    "work_life_balance_issue": [
        "balance", "personal", "family", "time off", "vacation",
        "weekend", "after hours", "late", "early", "flexible",
        "remote", "work from home", "wfh", "commute"
    ],
    "culture_issue": [
        "culture", "environment", "atmosphere", "values", "toxic",
        "politics", "favoritism", "bias", "discrimination",
        "inclusive", "diversity", "belonging", "fit"
    ]
}

DEFAULT_KEYWORDS = {
    "training_issue": ["tool", "software", "training", "learn", "difficult", "confusing"],
    "manager_issue": ["manager", "supervisor", "feedback", "guidance", "communication"],
    "team_issue": ["team", "colleague", "isolated", "alone", "included"],
    "workload_issue": ["workload", "overwhelmed", "too much", "burnout", "stress"],
    "career_growth_issue": ["growth", "career", "development", "opportunity", "learning"],
    "compensation_concern": ["salary", "pay", "compensation", "benefits", "raise"],
    "work_life_balance_issue": ["balance", "personal", "family", "time off", "flexible"],
    "culture_issue": ["culture", "environment", "toxic", "inclusive", "belonging"]
}


class TopicExtractor:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self._load_or_create_model()

    def _load_or_create_model(self):
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            try:
                with open(VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(LABEL_ENCODER_PATH, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Loaded existing topic classification model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        
        self._create_default_model()

    def _create_default_model(self):
        texts = []
        labels = []
        
        for category, keywords in DEFAULT_KEYWORDS.items():
            for kw in keywords:
                texts.append(kw)
                labels.append(category)
            for kw in PROBLEM_CATEGORIES.get(category, []):
                texts.append(kw)
                labels.append(category)
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X = self.vectorizer.fit_transform(texts)
        
        self.classifier = MultinomialNB(alpha=0.1)
        self.classifier.fit(X, labels)
        
        self._save_model()

    def _save_model(self):
        try:
            os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(LABEL_ENCODER_PATH, 'wb') as f:
                pickle.dump(self.classifier, f)
            logger.info("Saved topic classification model")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def extract_topics(self, text: str) -> List[str]:
        text_lower = text.lower()
        topics = []
        
        for category, keywords in PROBLEM_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if category not in topics:
                        topics.append(category)
        
        return topics if topics else ["general"]

    def detect_problem_category(self, text: str) -> Dict[str, Any]:
        if self.vectorizer is None or self.classifier is None:
            self._create_default_model()
        
        try:
            X = self.vectorizer.transform([text])
            probs = self.classifier.predict_proba(X)[0]
            classes = self.classifier.classes_
            
            max_idx = probs.argmax()
            category = classes[max_idx]
            confidence = float(probs[max_idx])
            
            all_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}
            
            return {
                "category": category,
                "confidence": confidence,
                "all_categories": all_probs
            }
        except Exception as e:
            logger.error(f"Problem category detection failed: {e}")
            return {
                "category": "none",
                "confidence": 0.0,
                "all_categories": {}
            }


def get_topic_extractor() -> TopicExtractor:
    return TopicExtractor()