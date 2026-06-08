import numpy as np
import pickle
import os
import logging
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from backend.config import RISK_MODEL_PATH, DATA_DIR

logger = logging.getLogger(__name__)


class AttritionRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            "sentiment_score",
            "emotion_score",
            "stress_level",
            "voice_confidence",
            "participation_rate",
            "feedback_consistency",
            "training_completion",
            "week_number",
            "problem_count",
            "negative_sentiment_streak"
        ]

    def prepare_features(self, employee_data: Dict[str, Any]) -> np.ndarray:
        features = []
        for name in self.feature_names:
            features.append(employee_data.get(name, 0.0))
        return np.array(features).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.save()
        logger.info("Risk model trained and saved")

    def predict(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            self.load_or_create_default()
        
        X = self.prepare_features(employee_data)
        X_scaled = self.scaler.transform(X)
        
        risk_score = self.model.predict_proba(X_scaled)[0][1] * 100
        
        if risk_score >= 60:
            risk_level = "High Risk"
        elif risk_score >= 30:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
        
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        )) if hasattr(self.model, 'feature_importances_') else {}
        
        return {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "confidence": float(max(self.model.predict_proba(X_scaled)[0])),
            "feature_importance": feature_importance
        }

    def predict_batch(self, employees_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(emp) for emp in employees_data]

    def save(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(RISK_MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained
                }, f)
            logger.info(f"Model saved to {RISK_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_or_create_default(self):
        if os.path.exists(RISK_MODEL_PATH):
            try:
                with open(RISK_MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
                self.is_trained = data['is_trained']
                logger.info("Loaded existing risk model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        
        self._create_default_model()

    def _create_default_model(self):
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.rand(n_samples, len(self.feature_names))
        X[:, 0] = np.random.beta(2, 5, n_samples)
        X[:, 1] = np.random.beta(2, 3, n_samples)
        X[:, 2] = np.random.beta(2, 5, n_samples)
        X[:, 3] = np.random.beta(5, 2, n_samples)
        X[:, 4] = np.random.beta(5, 2, n_samples)
        X[:, 5] = np.random.beta(3, 2, n_samples)
        X[:, 6] = np.random.beta(3, 2, n_samples)
        X[:, 7] = np.random.uniform(1, 14, n_samples)
        X[:, 8] = np.random.poisson(1, n_samples)
        X[:, 9] = np.random.poisson(0.5, n_samples)
        
        risk_score = (
            (1 - X[:, 0]) * 30 +
            (1 - X[:, 1]) * 20 +
            X[:, 2] * 25 +
            (1 - X[:, 3]) * 15 +
            (1 - X[:, 4]) * 10 +
            (1 - X[:, 5]) * 10 +
            (1 - X[:, 6]) * 5 +
            X[:, 7] * 1 +
            X[:, 8] * 5 +
            X[:, 9] * 5
        ) + np.random.normal(0, 5, n_samples)
        
        y = (risk_score > 50).astype(int)
        
        self.train(X, y)
        logger.info("Created default risk model with synthetic data")


def get_risk_model() -> AttritionRiskModel:
    model = AttritionRiskModel()
    model.load_or_create_default()
    return model