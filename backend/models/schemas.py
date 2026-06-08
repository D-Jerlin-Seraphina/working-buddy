from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


class ProblemCategory(str, Enum):
    TRAINING = "training_issue"
    MANAGER = "manager_issue"
    TEAM = "team_issue"
    WORKLOAD = "workload_issue"
    CAREER_GROWTH = "career_growth_issue"
    COMPENSATION = "compensation_concern"
    WORK_LIFE_BALANCE = "work_life_balance_issue"
    CULTURE = "culture_issue"
    NONE = "none"


class EmployeeCreate(BaseModel):
    employee_id: str
    name: str
    email: EmailStr
    department: Optional[str] = None
    role: Optional[str] = None
    joining_date: datetime
    manager_id: Optional[str] = None


class EmployeeResponse(BaseModel):
    id: int
    employee_id: str
    name: str
    email: str
    department: Optional[str]
    role: Optional[str]
    joining_date: datetime
    manager_id: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    employee_id: int
    week_number: int
    scheduled_date: datetime
    audio_file_path: Optional[str] = None


class ConversationResponse(BaseModel):
    id: int
    employee_id: int
    week_number: int
    scheduled_date: datetime
    completed_date: Optional[datetime]
    transcript: Optional[str]
    user_response: Optional[str]
    ai_response: Optional[str]
    sentiment: Optional[str]
    sentiment_score: Optional[float]
    emotion: Optional[str]
    emotion_confidence: Optional[float]
    voice_confidence: Optional[float]
    stress_level: Optional[float]
    problem_category: Optional[str]
    problem_confidence: Optional[float]
    topics: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True


class RiskAssessmentResponse(BaseModel):
    id: int
    employee_id: int
    week_number: int
    risk_score: float
    risk_level: RiskLevel
    sentiment_trend: Optional[float]
    emotion_trend: Optional[float]
    participation_rate: Optional[float]
    feedback_consistency: Optional[float]
    voice_confidence_avg: Optional[float]
    training_completion: Optional[float]
    primary_concern: Optional[str]
    recommendations: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    id: int
    employee_id: int
    alert_type: str
    severity: str
    message: str
    risk_score: Optional[float]
    details: Optional[Dict[str, Any]]
    is_read: bool
    is_resolved: bool
    created_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class VoiceAnalysisRequest(BaseModel):
    employee_id: int
    week_number: int
    audio_file_path: str


class VoiceAnalysisResponse(BaseModel):
    transcript: str
    sentiment: str
    sentiment_score: float
    emotion: str
    emotion_confidence: float
    voice_confidence: float
    stress_level: float
    speaking_speed: float
    pitch_variation: float
    energy_level: float
    hesitation_count: int
    pause_duration: float
    problem_category: str
    problem_confidence: float
    topics: List[str]
    risk_score: float
    risk_level: RiskLevel
    ai_response: str
    recommendations: List[str]


class HRDashboardResponse(BaseModel):
    total_employees: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    employees: List[Dict[str, Any]]
    alerts: List[AlertResponse]
    recent_conversations: List[ConversationResponse]


class TrendData(BaseModel):
    week: int
    sentiment: float
    emotion_score: float
    risk_score: float
    participation: bool


class EmployeeTrendResponse(BaseModel):
    employee_id: int
    employee_name: str
    trends: List[TrendData]
    current_risk_level: RiskLevel
    primary_concerns: List[str]