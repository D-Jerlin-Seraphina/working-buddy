import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "retention.db"
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

SPEECH_RECOGNITION_ENGINE = "whisper"
WHISPER_MODEL_SIZE = "base"

# TTS engine: "vibevoice" | "elevenlabs" | "speecht5" | "coqui"
TTS_ENGINE = os.getenv("TTS_ENGINE", "speecht5")
TTS_LANG = "en"

# ElevenLabs
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# VibeVoice
VIBEVOICE_MODEL_PATH = os.getenv("VIBEVOICE_MODEL_PATH", "microsoft/vibevoice-1.5B")

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "bhadresh-savani/bert-base-uncased-emotion"

RISK_MODEL_PATH = DATA_DIR / "risk_model.pkl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = DATA_DIR / "label_encoder.pkl"

SENTIMENT_THRESHOLD = 0.6
HIGH_RISK_THRESHOLD = 60
MEDIUM_RISK_THRESHOLD = 30

WEEKLY_CHECKIN_SCHEDULE = {
    1: "Week 1 - First impressions",
    2: "Week 2 - Early challenges",
    4: "Week 4 - Team integration",
    6: "Week 6 - Role clarity",
    8: "Week 8 - Skill utilization",
    10: "Week 10 - Growth & future",
    12: "Week 12 - Mid-point review",
    14: "Week 14 - Progress check"
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
