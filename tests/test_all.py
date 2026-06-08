"""
Tests for Employee Retention Buddy
Run: pytest tests/ -v
"""
import pytest
import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Analysis Tests ────────────────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        from backend.analysis.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        with patch.object(analyzer, 'load_model'):
            analyzer.classifier = MagicMock(return_value=[[
                {"label": "POSITIVE", "score": 0.9},
                {"label": "NEGATIVE", "score": 0.1}
            ]])
            result = analyzer.analyze("I love my job, everything is great!")
        assert result["sentiment"] == "positive"
        assert result["sentiment_score"] > 0.5
        assert "confidence" in result

    def test_negative_sentiment(self):
        from backend.analysis.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        with patch.object(analyzer, 'load_model'):
            analyzer.classifier = MagicMock(return_value=[[
                {"label": "POSITIVE", "score": 0.1},
                {"label": "NEGATIVE", "score": 0.9}
            ]])
            result = analyzer.analyze("I hate this place, so stressful")
        assert result["sentiment"] == "negative"
        assert result["sentiment_score"] < 0.5


class TestEmotionDetector:
    def test_emotion_detection(self):
        from backend.analysis.emotion import EmotionDetector
        detector = EmotionDetector()
        with patch.object(detector, 'load_model'):
            detector.classifier = MagicMock(return_value=[[
                {"label": "joy", "score": 0.8},
                {"label": "sadness", "score": 0.1},
                {"label": "anger", "score": 0.1}
            ]])
            result = detector.analyze("This is amazing!")
        assert result["emotion"] == "happy"
        assert result["confidence"] > 0.5
        assert "raw_emotion" in result

    def test_emotion_mapping(self):
        from backend.analysis.emotion import WORKPLACE_EMOTIONS
        assert WORKPLACE_EMOTIONS["joy"] == "happy"
        assert WORKPLACE_EMOTIONS["anger"] == "frustrated"
        assert WORKPLACE_EMOTIONS["fear"] == "anxious"


class TestTopicExtractor:
    def setup_method(self):
        from backend.analysis.topics import TopicExtractor
        self.extractor = TopicExtractor()

    def test_extract_training_topic(self):
        topics = self.extractor.extract_topics("I don't understand how to use the software training system")
        assert "training_issue" in topics

    def test_extract_workload_topic(self):
        topics = self.extractor.extract_topics("I'm overwhelmed with work, too many deadlines")
        assert "workload_issue" in topics

    def test_extract_no_topic_returns_general(self):
        topics = self.extractor.extract_topics("Everything is fine")
        assert isinstance(topics, list)

    def test_detect_problem_category_structure(self):
        result = self.extractor.detect_problem_category("My manager never gives feedback or direction")
        assert "category" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0


# ─── Voice Analysis Tests ──────────────────────────────────────────────────────

class TestVoiceEmotionAnalyzer:
    def test_empty_result_structure(self):
        from backend.speech.voice_analysis import VoiceEmotionAnalyzer
        analyzer = VoiceEmotionAnalyzer()
        result = analyzer._empty_result()
        for key in ["speaking_speed", "pitch_variation", "energy_level",
                    "hesitation_count", "pause_duration", "voice_confidence", "stress_level"]:
            assert key in result

    def test_voice_confidence_range(self):
        from backend.speech.voice_analysis import VoiceEmotionAnalyzer
        analyzer = VoiceEmotionAnalyzer()
        features = {"hesitation_count": 0, "pause_duration": 0.5,
                    "pitch_variation": 0.15, "energy_level": 0.05, "speaking_speed": 2.0}
        confidence = analyzer._calculate_voice_confidence(features)
        assert 0.0 <= confidence <= 1.0

    def test_stress_level_high_indicators(self):
        from backend.speech.voice_analysis import VoiceEmotionAnalyzer
        analyzer = VoiceEmotionAnalyzer()
        features = {"speaking_speed": 5.0, "pitch_variation": 0.4,
                    "hesitation_count": 5, "pause_duration": 3.0, "energy_level": 0.15}
        stress = analyzer._calculate_stress_level(features)
        assert stress > 0.5


# ─── Conversation Memory Tests ─────────────────────────────────────────────────

class TestConversationMemory:
    def setup_method(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.models.database import Base, Employee, Conversation
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.db = Session()
        self.Conversation = Conversation

        emp = Employee(
            employee_id="TEST001", name="Test User",
            email="test@company.com", joining_date=datetime.utcnow()
        )
        self.db.add(emp)
        self.db.commit()
        self.db.refresh(emp)
        self.emp_id = emp.id

    def test_empty_history(self):
        from backend.conversation.memory import ConversationMemory
        memory = ConversationMemory(self.db)
        assert memory.get_conversation_history(self.emp_id) == []

    def test_history_with_data(self):
        from backend.conversation.memory import ConversationMemory
        conv = self.Conversation(
            employee_id=self.emp_id, week_number=1,
            scheduled_date=datetime.utcnow(), completed_date=datetime.utcnow(),
            user_response="Great week!", ai_response="Happy to hear that.",
            sentiment="positive", emotion="happy", problem_category="none",
            sentiment_score=0.85, topics=["general"]
        )
        self.db.add(conv)
        self.db.commit()

        memory = ConversationMemory(self.db)
        history = memory.get_conversation_history(self.emp_id)
        assert len(history) == 1
        assert history[0]["sentiment"] == "positive"

    def test_recurring_topics_counted(self):
        from backend.conversation.memory import ConversationMemory
        for week in [1, 2, 4]:
            conv = self.Conversation(
                employee_id=self.emp_id, week_number=week,
                scheduled_date=datetime.utcnow(),
                user_response="struggling", sentiment="negative",
                problem_category="workload_issue",
                topics=["workload_issue"], sentiment_score=0.3
            )
            self.db.add(conv)
        self.db.commit()

        memory = ConversationMemory(self.db)
        topics = memory.get_recurring_topics(self.emp_id)
        assert topics.get("workload_issue") == 3


# ─── Risk Model Tests ──────────────────────────────────────────────────────────

class TestRiskModel:
    def setup_method(self):
        from backend.models.risk_model import AttritionRiskModel
        self.model = AttritionRiskModel()
        self.model._create_default_model()

    def test_prediction_structure(self):
        result = self.model.predict({
            "sentiment_score": 0.3, "emotion_score": 0.6, "stress_level": 0.7,
            "voice_confidence": 0.4, "participation_rate": 0.8, "feedback_consistency": 0.5,
            "training_completion": 0.6, "week_number": 8, "problem_count": 2,
            "negative_sentiment_streak": 3
        })
        assert "risk_score" in result
        assert "risk_level" in result
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_high_risk_data(self):
        result = self.model.predict({
            "sentiment_score": 0.05, "emotion_score": 0.9, "stress_level": 0.95,
            "voice_confidence": 0.1, "participation_rate": 0.2, "feedback_consistency": 0.1,
            "training_completion": 0.1, "week_number": 14, "problem_count": 5,
            "negative_sentiment_streak": 8
        })
        assert result["risk_level"] in ["Medium Risk", "High Risk"]

    def test_low_risk_data(self):
        result = self.model.predict({
            "sentiment_score": 0.95, "emotion_score": 0.9, "stress_level": 0.05,
            "voice_confidence": 0.95, "participation_rate": 1.0, "feedback_consistency": 0.95,
            "training_completion": 1.0, "week_number": 4, "problem_count": 0,
            "negative_sentiment_streak": 0
        })
        assert result["risk_level"] in ["Low Risk", "Medium Risk"]


# ─── Response Generator Tests ──────────────────────────────────────────────────

class TestResponseGenerator:
    def setup_method(self):
        from backend.conversation.response_generator import ResponseGenerator
        self.rg = ResponseGenerator()

    def test_generate_question_returns_string(self):
        q = self.rg.generate_question(week_number=1)
        assert isinstance(q, str) and len(q) > 0

    def test_generate_response_positive(self):
        r = self.rg.generate_response("I love my job!", "positive", "happy", "none", 1)
        assert isinstance(r, str) and len(r) > 0

    def test_generate_response_with_problem(self):
        r = self.rg.generate_response("I'm struggling", "negative", "frustrated", "workload_issue", 4)
        assert isinstance(r, str) and len(r) > 0

    def test_weekly_checkin_includes_name(self):
        msg = self.rg.generate_weekly_checkin_message(week_number=1, employee_name="Alex")
        assert "Alex" in msg


# ─── Recommendation Engine Tests ───────────────────────────────────────────────

class TestRecommendationEngine:
    def setup_method(self):
        from backend.conversation.recommendations import RecommendationEngine
        self.engine = RecommendationEngine()

    def test_recommendations_for_category(self):
        recs = self.engine.get_recommendations(["workload_issue"], "High Risk", "declining")
        assert len(recs) > 0
        assert recs[0]["category"] == "workload_issue"
        assert "action_items" in recs[0]

    def test_high_risk_no_category_gives_generic(self):
        recs = self.engine.get_recommendations([], "High Risk", "stable")
        assert len(recs) > 0
        assert recs[0]["category"] == "general"

    def test_declining_trend_escalates_to_high(self):
        recs = self.engine.get_recommendations(["team_issue"], "Medium Risk", "declining")
        assert recs[0]["priority"] == "High"


# ─── TTS Engine Tests ──────────────────────────────────────────────────────────

class TestElevenLabsTTS:
    def test_no_api_key_raises(self):
        from backend.speech.vibe_voice_tts import ElevenLabsTTS
        tts = ElevenLabsTTS(api_key=None)
        with pytest.raises(ValueError, match="API key"):
            tts.synthesize("Hello")

    def test_get_tts_engine_returns_elevenlabs(self):
        from backend.speech.vibe_voice_tts import get_tts_engine, ElevenLabsTTS
        engine = get_tts_engine("elevenlabs", api_key="fake_key")
        assert isinstance(engine, ElevenLabsTTS)


# ─── Database Schema Tests ─────────────────────────────────────────────────────

class TestDatabaseSchema:
    def test_all_tables_created(self):
        from sqlalchemy import create_engine, inspect
        from backend.models.database import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        tables = inspect(engine).get_table_names()
        for table in ["employees", "conversations", "risk_assessments", "alerts"]:
            assert table in tables

    def test_employee_model_fields(self):
        from backend.models.database import Employee
        emp = Employee(
            employee_id="E001", name="Jane Doe",
            email="jane@co.com", joining_date=datetime.utcnow()
        )
        assert emp.employee_id == "E001"
        assert emp.is_active is None or emp.is_active

    def test_risk_level_enum_values(self):
        from backend.models.database import RiskLevel
        assert RiskLevel.HIGH.value == "High Risk"
        assert RiskLevel.MEDIUM.value == "Medium Risk"
        assert RiskLevel.LOW.value == "Low Risk"
