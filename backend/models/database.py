from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Enum, JSON, Boolean
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class RiskLevel(str, enum.Enum):
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


class ProblemCategory(str, enum.Enum):
    TRAINING = "training_issue"
    MANAGER = "manager_issue"
    TEAM = "team_issue"
    WORKLOAD = "workload_issue"
    CAREER_GROWTH = "career_growth_issue"
    COMPENSATION = "compensation_concern"
    WORK_LIFE_BALANCE = "work_life_balance_issue"
    CULTURE = "culture_issue"
    NONE = "none"


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    department = Column(String(100))
    role = Column(String(100))
    joining_date = Column(DateTime, nullable=False)
    manager_id = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="employee", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="employee", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="employee", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    week_number = Column(Integer, nullable=False)
    scheduled_date = Column(DateTime, nullable=False)
    completed_date = Column(DateTime, nullable=True)
    audio_file_path = Column(String(500))
    transcript = Column(Text)
    user_response = Column(Text)
    ai_response = Column(Text)
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    emotion = Column(String(30))
    emotion_confidence = Column(Float)
    voice_confidence = Column(Float)
    stress_level = Column(Float)
    speaking_speed = Column(Float)
    pitch_variation = Column(Float)
    energy_level = Column(Float)
    hesitation_count = Column(Integer)
    pause_duration = Column(Float)
    problem_category = Column(String(50))
    problem_confidence = Column(Float)
    topics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="conversations")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    week_number = Column(Integer, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    sentiment_trend = Column(Float)
    emotion_trend = Column(Float)
    participation_rate = Column(Float)
    feedback_consistency = Column(Float)
    voice_confidence_avg = Column(Float)
    training_completion = Column(Float)
    primary_concern = Column(String(50))
    recommendations = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="risk_assessments")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    risk_score = Column(Float)
    details = Column(JSON)
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    employee = relationship("Employee", back_populates="alerts")