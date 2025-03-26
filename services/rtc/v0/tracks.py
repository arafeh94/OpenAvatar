import asyncio
from typing import Union
import time
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame

from core.plugins.text2speech import Audio


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(self):
        super().__init__()
        self.audio: Union[Audio, None] = None
        self.current_sample = 0
        self.written = []
        self.event = asyncio.Event()

    def stream(self, audio: Audio):
        self.audio = audio
        self.event.set()

    def av_frame(self, samples, sampling_rate, pts):
        block = (np.array(samples) / max(1, np.max(np.abs(samples))) * 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(block.reshape(1, -1), format='s16', layout='mono')
        frame.sample_rate = sampling_rate
        frame.pts = pts

        return frame

    async def recv(self):
        if self.audio is None:
            await self.event.wait()
            self.event.clear()

        sample_size = int(AUDIO_PTIME * self.audio.sampling_rate)
        end_sample_index = self.current_sample + sample_size

        if end_sample_index >= len(self.audio.samples) > self.current_sample:
            end_sample_index = len(self.audio.samples)
        elif end_sample_index >= len(self.audio.samples):
            self.current_sample = 0
            self.audio = None
            del self._timestamp
            return await self.recv()

        samples = self.audio.samples[self.current_sample:end_sample_index]
        self.written.extend(samples)
        self.current_sample += sample_size

        if hasattr(self, "_timestamp"):
            self._timestamp += len(samples)
            wait = self._start + (self._timestamp / self.audio.sampling_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        return self.av_frame(samples, self.audio.sampling_rate, self._timestamp)
