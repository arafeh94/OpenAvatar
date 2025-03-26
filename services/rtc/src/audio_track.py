import asyncio
import logging
from abc import ABC, abstractmethod
import time
from typing import Union, Callable

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame

from core.plugins.text2speech import Audio


class ASTrack(MediaStreamTrack, ABC):
    kind = "audio"
    _start: float
    _timestamp: int

    def reset_timestamp(self):
        del self._start
        del self._timestamp

    async def recv(self):
        av_frame = await self.next_frame()

        if hasattr(self, "_timestamp"):
            self._timestamp += av_frame.samples
            wait = self._start + (self._timestamp / av_frame.sample_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        av_frame.pts = self._timestamp
        return av_frame

    @abstractmethod
    async def next_frame(self) -> Union[AudioFrame]:
        pass


class AudioFrameCreator(ABC):
    @abstractmethod
    def av_frame(self, samples, sample_rate) -> AudioFrame:
        pass


class MonoFrame(AudioFrameCreator):

    def av_frame(self, samples, sample_rate) -> AudioFrame:
        block = (np.array(samples) / max(1, np.max(np.abs(samples))) * 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(block.reshape(1, -1), format='s16', layout='mono')
        frame.sample_rate = sample_rate

        return frame


class AudioStream(ASTrack):
    MONO = MonoFrame()

    def __init__(self, frame_creator: AudioFrameCreator):
        super().__init__()
        self._audio: Union[Audio, None] = None
        self._current_sample = 0
        self._frame_creator = frame_creator
        self._event = asyncio.Event()
        self.logger = logging.getLogger("AudioStream")

    def stream(self, audio: Audio):
        self._audio = audio
        self._event.set()

    def reset(self):
        self._current_sample = 0
        self._audio = None
        self._event.clear()
        self.reset_timestamp()

    async def next_frame(self) -> Union[AudioFrame]:
        if self._audio is None:
            await self._event.wait()

        sample_size = int(AUDIO_PTIME * self._audio.sampling_rate)
        end_sample = self._current_sample + sample_size

        if end_sample >= len(self._audio.samples) > self._current_sample:
            end_sample = len(self._audio.samples)

        if end_sample >= len(self._audio.samples):
            self.reset()
            return await self.next_frame()

        samples = self._audio.samples[self._current_sample:end_sample]
        self._current_sample += sample_size

        return self._frame_creator.av_frame(samples, self._audio.sampling_rate)
