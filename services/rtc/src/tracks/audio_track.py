import asyncio
import logging
from abc import ABC, abstractmethod
import time
from asyncio import Queue
from typing import Union

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame

from core.interfaces.va import Audio


# noinspection PyTypeChecker
class ASTrack(MediaStreamTrack, ABC):
    kind = "audio"
    _start: float
    _timestamp: int

    def __init__(self):
        super().__init__()
        self.last_time_stamp = 0

    def reset_timestamp(self):
        if hasattr(self, "_timestamp"):
            del self._start
            del self._timestamp

    async def recv(self):
        av_frame = await self.next_frame()

        if hasattr(self, "_timestamp"):
            self._timestamp += av_frame.samples
            ttst = time.time()
            wait = self._start + (self._timestamp / av_frame.sample_rate) - ttst
            wait = min(max(0, wait), 0.03)
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        av_frame.pts = self._timestamp
        return av_frame

    @abstractmethod
    async def next_frame(self) -> AudioFrame:
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
        self.logger = logging.getLogger("AudioStream")
        self.buffer_queue: Queue[Audio] = Queue()

    def reset(self):
        self._current_sample = 0
        self._audio = None
        self.reset_timestamp()

    async def next_frame(self) -> Union[AudioFrame]:
        if self._audio is None:
            self._audio = await self.buffer_queue.get()

        sample_size = int(AUDIO_PTIME * self._audio.sampling_rate)
        end_sample = self._current_sample + sample_size

        if end_sample >= len(self._audio.samples) > self._current_sample:
            end_sample = len(self._audio.samples)

        if end_sample >= len(self._audio.samples):
            self._audio = None
            self._current_sample = 0
            return await self.next_frame()

        samples = self._audio.samples[self._current_sample:end_sample]
        self._current_sample += sample_size

        return self._frame_creator.av_frame(samples, self._audio.sampling_rate)

    async def stream(self, audio: Audio):
        await self.buffer_queue.put(audio)
