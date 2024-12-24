import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import io

from external.core.interfaces.base_convertable import Convertable


class VoiceConvertable(Convertable):
    def __init__(self, voice):
        super().__init__()
        self.voice = voice

    def as_byte_buffer(self):
        buffer = io.BytesIO()
        sf.write(buffer, self.voice["audio"], samplerate=self.voice["sampling_rate"], format="WAV")
        buffer.seek(0)
        return buffer

    def as_streaming_response(self):
        from fastapi.responses import StreamingResponse
        return StreamingResponse(self.as_byte_buffer(), media_type="audio/wav")

    def as_file(self, filename, format="WAV"):
        try:
            with open(filename, 'wb') as file:
                sf.write(file, self.voice["audio"], samplerate=self.voice["sampling_rate"], format=format)
                return True
        except Exception as e:
            return False


class Text2Speech:
    def __init__(self, device='cuda'):
        self.model = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)
        self.dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    def convert(self, text, voice_id):
        speaker_embeddings = torch.tensor(self.dataset[voice_id]["xvector"]).unsqueeze(0)
        voice = self.model(text, forward_params={"speaker_embeddings": speaker_embeddings})
        return VoiceConvertable(voice)
