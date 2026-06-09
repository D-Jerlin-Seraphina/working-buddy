import os
import torch
import logging
from typing import Optional
import uuid
from backend.config import AUDIO_DIR

logger = logging.getLogger(__name__)

try:
    from transformers import VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor
    VIBEVOICE_AVAILABLE = True
except ImportError:
    VIBEVOICE_AVAILABLE = False

try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset
    SPEECHT5_AVAILABLE = True
except ImportError:
    SPEECHT5_AVAILABLE = False


class VibeVoiceTTS:
    def __init__(self, model_path: str = "microsoft/vibevoice-1.5B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available = VIBEVOICE_AVAILABLE
        if self.available:
            logger.info(f"Initializing Vibe Voice TTS on {self.device}")
        else:
            logger.warning("VibeVoice not available in transformers. Using fallback TTS.")

    def load_model(self):
        if self.model is None:
            if not self.available:
                logger.warning("VibeVoice architecture not in transformers. Cannot load model.")
                return
            try:
                self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                )
                self.model.eval()
                logger.info("Vibe Voice model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Vibe Voice: {e}")
                raise

    def synthesize(self, text: str, speaker: str = "en-US-female-1", output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(AUDIO_DIR / f"vibe_response_{uuid.uuid4().hex[:8]}.wav")
        
        if not self.available:
            logger.warning("VibeVoice not available. Returning mock audio path.")
            return output_path
        
        self.load_model()
        
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


class SpeechT5TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.available = SPEECHT5_AVAILABLE
        if self.available:
            logger.info(f"Initializing SpeechT5 TTS on {self.device}")
        else:
            logger.warning("SpeechT5 not available. Install with: pip install datasets accelerate")

    def load_model(self):
        if self.model is None and self.available:
            try:
                logger.info("Loading SpeechT5 models (first run downloads ~500MB)...")
                self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
                self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
                
                # Create a default speaker embedding (random but deterministic)
                # This avoids the dataset loading issue
                torch.manual_seed(42)
                self.speaker_embeddings = torch.randn(1, 512).to(self.device)
                
                logger.info("SpeechT5 TTS loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SpeechT5: {e}")
                self.available = False
                raise

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(AUDIO_DIR / f"speecht5_response_{uuid.uuid4().hex[:8]}.wav")
        
        if not self.available:
            logger.warning("SpeechT5 not available. Returning mock audio path.")
            return output_path
        
        self.load_model()
        
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    self.speaker_embeddings,
                    vocoder=self.vocoder
                )
            
            import soundfile as sf
            sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
            
            logger.info(f"SpeechT5 synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SpeechT5 synthesis failed: {e}")
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
    elif engine == "speecht5":
        return SpeechT5TTS()
    else:
        from backend.speech.tts import TextToSpeech, get_tts
        return get_tts(**kwargs)