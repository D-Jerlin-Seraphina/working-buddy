import os
import torch
from typing import Optional
import logging
import uuid
from backend.config import AUDIO_DIR

logger = logging.getLogger(__name__)

try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    logger.warning("Coqui TTS not available. TTS functionality will be limited.")


class TextToSpeech:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if COQUI_TTS_AVAILABLE:
            logger.info(f"Initializing Coqui TTS on {self.device} with model {model_name}")
        else:
            logger.info(f"Coqui TTS not available. Using mock TTS.")

    def load_model(self):
        if self.tts is None:
            if COQUI_TTS_AVAILABLE:
                self.tts = TTS(model_name=self.model_name, progress_bar=False).to(self.device)
                logger.info(f"Coqui TTS model {self.model_name} loaded successfully")
            else:
                logger.warning("Coqui TTS not available. Cannot load model.")

    def synthesize(self, text: str, output_path: Optional[str] = None, speaker_wav: Optional[str] = None) -> str:
        if not COQUI_TTS_AVAILABLE:
            logger.warning("TTS synthesis requested but Coqui TTS not available. Returning mock path.")
            if output_path is None:
                output_path = str(AUDIO_DIR / f"response_{uuid.uuid4().hex[:8]}.wav")
            return output_path
            
        self.load_model()
        
        if output_path is None:
            output_path = str(AUDIO_DIR / f"response_{uuid.uuid4().hex[:8]}.wav")
        
        try:
            if speaker_wav and os.path.exists(speaker_wav):
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language="en"
                )
            else:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language="en"
                )
            logger.info(f"TTS synthesis completed: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    def synthesize_streaming(self, text: str) -> bytes:
        if not COQUI_TTS_AVAILABLE:
            logger.warning("TTS streaming requested but Coqui TTS not available. Returning empty bytes.")
            return b""
            
        self.load_model()
        try:
            wav = self.tts.tts(text=text, language="en")
            import io
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, wav, 22050, format='WAV')
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise


def get_tts(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> TextToSpeech:
    return TextToSpeech(model_name)