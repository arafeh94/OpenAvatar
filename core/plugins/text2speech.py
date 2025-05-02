import io
import os

import librosa
import numpy as np
import torch
from elevenlabs import VoiceSettings, ElevenLabs
from pydub import AudioSegment
from transformers import pipeline
from datasets import load_dataset
from core.interfaces.base_tts import Text2Speech
from core.interfaces.va import VoiceConvertable, Audio
from core.tools import utils
from manifest import Manifest
import soundfile as sf


class MicrosoftText2Speech(Text2Speech):
    def __init__(self, device='cuda'):
        batch_size = Manifest().query('tts.prediction_batch_size', 8000)
        self.model = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device, batch_size=batch_size)
        self.dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    def convert(self, text, **kwargs) -> VoiceConvertable:
        if 'voice_id' not in kwargs:
            raise Exception("MicrosoftText2Speech requires voice_id parameter")
        voice_id = kwargs['voice_id']
        speaker_embeddings = torch.tensor(self.dataset[voice_id]["xvector"]).unsqueeze(0)
        voice = self.model(text, forward_params={"speaker_embeddings": speaker_embeddings})
        return VoiceConvertable(voice)


class ElevenLabsText2Speech(Text2Speech):
    def __init__(self):
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def convert(self, text, **kwargs) -> VoiceConvertable:
        voice_id = kwargs['voice_id'] if 'voice_id' in kwargs else "pNInz6obpgDQGcFmaJgB"
        voice_id = "pNInz6obpgDQGcFmaJgB"
        response = self.client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="pcm_16000",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
        bytes = utils.as_bytes(response)
        audio, sr = sf.read(bytes, samplerate=16000, channels=1, subtype='PCM_16', format='RAW')

        audio_info = {
            'audio': audio,
            'sampling_rate': sr,
        }

        return VoiceConvertable(audio_info)
