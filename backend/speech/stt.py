import whisper
import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Whisper STT on {self.device} with model {model_size}")

    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model {self.model_size} loaded successfully")

    def transcribe(self, audio_path: str, language: str = "en") -> dict:
        self.load_model()
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=torch.cuda.is_available(),
                verbose=False
            )
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_with_timestamps(self, audio_path: str, language: str = "en") -> dict:
        self.load_model()
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=torch.cuda.is_available(),
                verbose=False,
                word_timestamps=True
            )
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "words": result.get("words", [])
            }
        except Exception as e:
            logger.error(f"Transcription with timestamps failed: {e}")
            raise


def get_stt(model_size: str = "base") -> SpeechToText:
    return SpeechToText(model_size)