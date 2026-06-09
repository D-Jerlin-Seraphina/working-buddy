from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid
from datetime import datetime

from backend.config import AUDIO_DIR, DATABASE_URL
from backend.utils.database import init_db, get_db
from backend.models.database import Employee, Conversation, RiskAssessment, Alert, RiskLevel, ProblemCategory
from backend.speech import get_stt, get_tts_engine, get_voice_analyzer
from backend.analysis import get_sentiment_analyzer, get_emotion_detector, get_topic_extractor
from backend.conversation import get_response_generator, get_recommendation_engine
from backend.conversation.memory import get_conversation_memory
from backend.models.risk_model import get_risk_model

app = FastAPI(title="Employee Retention Buddy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    get_stt().load_model()
    get_sentiment_analyzer().load_model()
    get_emotion_detector().load_model()
    get_topic_extractor()
    get_risk_model()
    print("All models preloaded successfully")

class EmployeeCreate(BaseModel):
    employee_id: str
    name: str
    email: str
    department: Optional[str] = None
    role: Optional[str] = None
    joining_date: datetime
    manager_id: Optional[str] = None

class ConversationRequest(BaseModel):
    employee_id: int
    week_number: int
    audio_file_path: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    employee_id: int
    week_number: int
    text: str

@app.get("/")
def root():
    return {"message": "Employee Retention Buddy API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Employee endpoints
@app.post("/employees")
def create_employee(emp: EmployeeCreate):
    db = next(get_db())
    db_emp = Employee(**emp.dict())
    db.add(db_emp)
    db.commit()
    db.refresh(db_emp)
    return db_emp

@app.get("/employees")
def list_employees():
    db = next(get_db())
    return db.query(Employee).filter(Employee.is_active == True).all()

@app.get("/employees/{emp_id}")
def get_employee(emp_id: int):
    db = next(get_db())
    emp = db.query(Employee).filter(Employee.id == emp_id).first()
    if not emp:
        raise HTTPException(404, "Employee not found")
    return emp

# Voice processing endpoint
@app.post("/process-voice")
async def process_voice(
    employee_id: int,
    week_number: int,
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    db = next(get_db())
    
    # Save audio
    audio_path = str(AUDIO_DIR / f"{employee_id}_w{week_number}_{uuid.uuid4().hex[:8]}.wav")
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Transcribe
    stt = get_stt()
    transcript_result = stt.transcribe(audio_path)
    transcript = transcript_result["text"]
    
    # Analyze
    sentiment = get_sentiment_analyzer().analyze(transcript)
    emotion = get_emotion_detector().analyze(transcript)
    voice = get_voice_analyzer().analyze(audio_path)
    topic = get_topic_extractor().detect_problem_category(transcript)
    topics = get_topic_extractor().extract_topics(transcript)
    
    # Risk prediction
    risk_model = get_risk_model()
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    risk_data = {
        "sentiment_score": sentiment["sentiment_score"],
        "emotion_score": emotion["confidence"],
        "stress_level": voice["stress_level"],
        "voice_confidence": voice["voice_confidence"],
        "participation_rate": 1.0,
        "feedback_consistency": 0.8,
        "training_completion": 0.5,
        "week_number": week_number,
        "problem_count": 1 if topic["category"] != "none" else 0,
        "negative_sentiment_streak": 0
    }
    risk = risk_model.predict(risk_data)
    
    # Generate AI response
    rg = get_response_generator(db)
    ai_response = rg.generate_response(
        transcript, sentiment["sentiment"], emotion["emotion"],
        topic["category"], week_number, employee_id
    )
    
    # Generate TTS response using configured engine (vibevoice or elevenlabs)
    from backend.config import TTS_ENGINE, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
    tts_kwargs = {}
    if TTS_ENGINE == "elevenlabs" and ELEVENLABS_API_KEY:
        tts_kwargs = {"api_key": ELEVENLABS_API_KEY, "voice_id": ELEVENLABS_VOICE_ID}
    tts = get_tts_engine(TTS_ENGINE, **tts_kwargs)
    try:
        ai_audio_path = tts.synthesize(ai_response)
    except Exception:
        ai_audio_path = None
    
    # Save conversation
    conv = Conversation(
        employee_id=employee_id,
        week_number=week_number,
        scheduled_date=datetime.utcnow(),
        completed_date=datetime.utcnow(),
        audio_file_path=audio_path,
        transcript=transcript,
        user_response=transcript,
        ai_response=ai_response,
        sentiment=sentiment["sentiment"],
        sentiment_score=sentiment["sentiment_score"],
        emotion=emotion["emotion"],
        emotion_confidence=emotion["confidence"],
        voice_confidence=voice["voice_confidence"],
        stress_level=voice["stress_level"],
        speaking_speed=voice["speaking_speed"],
        pitch_variation=voice["pitch_variation"],
        energy_level=voice["energy_level"],
        hesitation_count=voice["hesitation_count"],
        pause_duration=voice["pause_duration"],
        problem_category=topic["category"],
        problem_confidence=topic["confidence"],
        topics=topics
    )
    db.add(conv)
    
    # Save risk assessment
    risk_assessment = RiskAssessment(
        employee_id=employee_id,
        week_number=week_number,
        risk_score=risk["risk_score"],
        risk_level=RiskLevel(risk["risk_level"]),
        primary_concern=topic["category"],
        recommendations=[]
    )
    db.add(risk_assessment)
    
    # Create alert if high risk
    if risk["risk_level"] == "High Risk":
        alert = Alert(
            employee_id=employee_id,
            alert_type="HIGH_RISK",
            severity="high",
            message=f"Employee {employee.name} at high attrition risk ({risk['risk_score']:.1f}%)",
            risk_score=risk["risk_score"],
            details={"week": week_number, "primary_concern": topic["category"]}
        )
        db.add(alert)
    
    db.commit()
    
    return {
        "transcript": transcript,
        "sentiment": sentiment,
        "emotion": emotion,
        "voice": voice,
        "topic": topic,
        "topics": topics,
        "risk": risk,
        "ai_response": ai_response,
        "ai_audio_path": ai_audio_path
    }

# Text analysis endpoint (fallback)
@app.post("/analyze-text")
def analyze_text(req: TextAnalysisRequest):
    db = next(get_db())
    
    sentiment = get_sentiment_analyzer().analyze(req.text)
    emotion = get_emotion_detector().analyze(req.text)
    topic = get_topic_extractor().detect_problem_category(req.text)
    topics = get_topic_extractor().extract_topics(req.text)
    
    risk_model = get_risk_model()
    risk_data = {
        "sentiment_score": sentiment["sentiment_score"],
        "emotion_score": emotion["confidence"],
        "stress_level": 0.3,
        "voice_confidence": 0.8,
        "participation_rate": 1.0,
        "feedback_consistency": 0.8,
        "training_completion": 0.5,
        "week_number": req.week_number,
        "problem_count": 1 if topic["category"] != "none" else 0,
        "negative_sentiment_streak": 0
    }
    risk = risk_model.predict(risk_data)
    
    rg = get_response_generator(db)
    ai_response = rg.generate_response(
        req.text, sentiment["sentiment"], emotion["emotion"],
        topic["category"], req.week_number, req.employee_id
    )
    
    return {
        "sentiment": sentiment,
        "emotion": emotion,
        "topic": topic,
        "topics": topics,
        "risk": risk,
        "ai_response": ai_response
    }

# Dashboard endpoints
@app.get("/dashboard")
def get_dashboard():
    db = next(get_db())
    
    employees = db.query(Employee).filter(Employee.is_active == True).all()
    high_risk = db.query(RiskAssessment).filter(RiskAssessment.risk_level == RiskLevel.HIGH).count()
    medium_risk = db.query(RiskAssessment).filter(RiskAssessment.risk_level == RiskLevel.MEDIUM).count()
    low_risk = db.query(RiskAssessment).filter(RiskAssessment.risk_level == RiskLevel.LOW).count()
    
    recent_alerts = db.query(Alert).filter(Alert.is_resolved == False).order_by(Alert.created_at.desc()).limit(10).all()
    recent_convs = db.query(Conversation).order_by(Conversation.created_at.desc()).limit(10).all()
    
    emp_data = []
    for emp in employees:
        latest_risk = db.query(RiskAssessment).filter(RiskAssessment.employee_id == emp.id).order_by(RiskAssessment.week_number.desc()).first()
        emp_data.append({
            "id": emp.id,
            "employee_id": emp.employee_id,
            "name": emp.name,
            "department": emp.department,
            "role": emp.role,
            "risk_score": latest_risk.risk_score if latest_risk else 0,
            "risk_level": latest_risk.risk_level.value if latest_risk else "Unknown",
            "primary_concern": latest_risk.primary_concern if latest_risk else None
        })
    
    return {
        "total_employees": len(employees),
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": low_risk,
        "employees": emp_data,
        "alerts": recent_alerts,
        "recent_conversations": recent_convs
    }

@app.get("/employees/{emp_id}/trends")
def get_employee_trends(emp_id: int):
    db = next(get_db())
    memory = get_conversation_memory(db)
    summary = memory.get_employee_summary(emp_id)
    return summary

@app.get("/recommendations/{emp_id}")
def get_recommendations(emp_id: int):
    db = next(get_db())
    memory = get_conversation_memory(db)
    summary = memory.get_employee_summary(emp_id)
    
    recent_risk = db.query(RiskAssessment).filter(RiskAssessment.employee_id == emp_id).order_by(RiskAssessment.week_number.desc()).first()
    risk_level = recent_risk.risk_level.value if recent_risk else "Low Risk"
    
    engine = get_recommendation_engine()
    problem_cats = list(summary.get("recurring_topics", {}).keys())
    sentiment_trend = "declining" if summary.get("avg_sentiment", 0.5) < 0.4 else "stable"
    
    recs = engine.get_recommendations(problem_cats, risk_level, sentiment_trend)
    return {"recommendations": recs}

@app.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    db = next(get_db())
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.is_resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    return {"status": "resolved"}

class TrainingDataRequest(BaseModel):
    n_employees: int = 50

@app.post("/admin/generate-training-data")
def generate_training_data(req: TrainingDataRequest):
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "data.generate_synthetic", "--employees", str(req.n_employees)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        get_risk_model()._create_default_model()
        return {"status": "ok", "output": result.stdout}
    raise HTTPException(500, result.stderr)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300, timeout_graceful_shutdown=300)