import os

import numpy as np
import torch
from elevenlabs import VoiceSettings, ElevenLabs
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
        self.model = Manifest().query('elevenlabs.model', 'eleven_turbo_v2_5')
        self.voice_settings = Manifest().query('elevenlabs.voice_settings', {
            "stability": 0.8,
            "similarity_boost": 0.7,
            "style": 0.4,
            "speed": 1.0
        })

    def convert(self, text, **kwargs) -> VoiceConvertable:
        voice_id = kwargs['voice_id'] if 'voice_id' in kwargs else "pNInz6obpgDQGcFmaJgB"
        response = self.client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="pcm_16000",
            text=text,
            model_id=self.model,
            voice_settings=VoiceSettings(
                use_speaker_boost=True,
                **self.voice_settings
            ),
        )
        bytes = utils.as_bytes(response)
        audio, sr = sf.read(bytes, samplerate=16000, channels=1, subtype='PCM_16', format='RAW')

        audio_info = {
            'audio': audio,
            'sampling_rate': sr,
        }

        return VoiceConvertable(audio_info)


class FakeText2Speech(Text2Speech):
    """
    Returns silent audio whose duration is estimated from text length.
    Output format matches ElevenLabsText2Speech: {'audio': np.ndarray, 'sampling_rate': int}
    """

    def __init__(
            self,
            sampling_rate: int = 16000,
            chars_per_second: float = 14.0,  # ~ 160-190 wpm depending on language; tweak as needed
            min_seconds: float = 0.25,
            max_seconds: float | None = None,
            pad_seconds: float = 0.0,  # optional extra silence
    ):
        self.sampling_rate = sampling_rate
        self.chars_per_second = chars_per_second
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.pad_seconds = pad_seconds

    def _estimate_seconds(self, text: str, cps: float, min_s: float, max_s: float | None, pad_s: float) -> float:
        # normalize whitespace a bit so "aaaa   bbb" doesn't overcount too much
        normalized = " ".join((text or "").split())
        n_chars = len(normalized)

        secs = (n_chars / cps) if cps > 0 else 0.0
        secs = max(secs, float(min_s))
        secs += float(pad_s)

        if max_s is not None:
            secs = min(secs, float(max_s))

        return float(secs)

    def convert(self, text, **kwargs) -> VoiceConvertable:
        sr = int(kwargs.get("sampling_rate", self.sampling_rate))
        cps = float(kwargs.get("chars_per_second", self.chars_per_second))
        min_s = float(kwargs.get("min_seconds", self.min_seconds))
        max_s = kwargs.get("max_seconds", self.max_seconds)
        pad_s = float(kwargs.get("pad_seconds", self.pad_seconds))

        duration_s = self._estimate_seconds(text, cps=cps, min_s=min_s, max_s=max_s, pad_s=pad_s)
        n_samples = int(round(duration_s * sr))

        audio = np.zeros((n_samples,), dtype=np.float32)

        audio_info = {
            "audio": audio,
            "sampling_rate": sr,
        }
        return VoiceConvertable(audio_info)
