from abc import ABC, abstractmethod

import librosa
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import io

from core.interfaces.base_convertable import Convertable
from manifest import Manifest


class Audio:
    def __init__(self, audio, sampling_rate=None):
        if isinstance(audio, dict):
            self.samples = audio['audio']
            self.sampling_rate = audio['sampling_rate']
        else:
            self.samples = audio
            self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.samples)

    def clip(self, start, end):
        return Audio(self.samples[start:end], self.sampling_rate)

    def write(self, path):
        import soundfile as sf
        sf.write(path, self.samples, self.sampling_rate)

    @staticmethod
    def load(audio_path, sr=None):
        samples, sampling_rate = librosa.core.load(audio_path, sr=sr)
        return Audio(samples, sampling_rate)


class VoiceConvertable(Convertable):
    def __init__(self, voice):
        super().__init__()
        self._voice = voice

    def as_byte_buffer(self):
        buffer = io.BytesIO()
        sf.write(buffer, self._voice["audio"], samplerate=self._voice["sampling_rate"], format="WAV")
        buffer.seek(0)
        return buffer

    def as_streaming_response(self):
        from fastapi.responses import StreamingResponse
        return StreamingResponse(self.as_byte_buffer(), media_type="audio/wav")

    def as_file(self, filename, format="WAV"):
        try:
            with open(filename, 'wb') as file:
                sf.write(file, self._voice["audio"], samplerate=self._voice["sampling_rate"], format=format)
                return True
        except Exception as e:
            return False

    def as_audio(self):
        return Audio(self._voice)


class Text2Speech(ABC):
    @abstractmethod
    def convert(self, text, **kwargs) -> VoiceConvertable:
        pass


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
