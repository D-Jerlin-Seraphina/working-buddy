import os
import torch
import logging
from typing import Optional
import uuid
from backend.config import AUDIO_DIR

logger = logging.getLogger(__name__)


class VibeVoiceTTS:
    def __init__(self, model_path: str = "microsoft/vibevoice-1.5B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Vibe Voice TTS on {self.device}")

    def load_model(self):
        if self.model is None:
            try:
                from transformers import VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor
                self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                )
                self.model.eval()
                logger.info("Vibe Voice model loaded successfully")
            except ImportError:
                logger.error("Vibe Voice requires transformers>=4.45.0. Install with: pip install transformers>=4.45.0 accelerate")
                raise
            except Exception as e:
                logger.error(f"Failed to load Vibe Voice: {e}")
                raise

    def synthesize(self, text: str, speaker: str = "en-US-female-1", output_path: Optional[str] = None) -> str:
        self.load_model()
        
        if output_path is None:
            output_path = str(AUDIO_DIR / f"vibe_response_{uuid.uuid4().hex[:8]}.wav")
        
        try:
            inputs = self.processor(
                text=[text],
                voice_samples=[speaker],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            import soundfile as sf
            sf.write(output_path, outputs.audio[0].cpu().numpy(), samplerate=24000)
            
            logger.info(f"Vibe Voice synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Vibe Voice synthesis failed: {e}")
            raise


class ElevenLabsTTS:
    def __init__(self, api_key: str = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            logger.warning("ElevenLabs API key not set. Set ELEVENLABS_API_KEY env var.")

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        if not self.api_key:
            raise ValueError("ElevenLabs API key required")
        
        if output_path is None:
            output_path = str(AUDIO_DIR / f"eleven_response_{uuid.uuid4().hex[:8]}.wav")
        
        import requests
        
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"ElevenLabs synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            raise


def get_tts_engine(engine: str = "coqui", **kwargs):
    if engine == "vibevoice":
        return VibeVoiceTTS(**kwargs)
    elif engine == "elevenlabs":
        return ElevenLabsTTS(**kwargs)
    else:
        from backend.speech.tts import TextToSpeech, get_tts
        return get_tts(**kwargs)