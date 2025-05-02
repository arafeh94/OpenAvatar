import numpy as np
import soundfile as sf
import io
from core.interfaces.base_convertable import Convertable
import librosa


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

    def extends(self, audio: 'Audio'):
        self.samples = np.append(self.samples, audio.samples)

    def clip(self, start, end):
        return Audio(self.samples[start:end], self.sampling_rate)

    def write(self, path, format='wav'):
        import soundfile as sf
        sf.write(path, self.samples, self.sampling_rate, format=format)

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
