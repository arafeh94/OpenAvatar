import asyncio
import fractions
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union, Iterator
import time

import av
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import Packet
from av.frame import Frame


class VSTrack(MediaStreamTrack, ABC):
    kind = "video"

    _start: float
    _timestamp: int

    def __init__(self):
        super().__init__()
        self.p_time = 1 / 24
        self.clock_rate = VIDEO_CLOCK_RATE
        self.time_base = VIDEO_TIME_BASE
        self.logger = logging.getLogger("VSTrack")

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int(self.p_time * self.clock_rate)
            wait = self._start + (self._timestamp / self.clock_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        return self._timestamp, self.time_base

    def reset_timestamp(self):
        del self._start
        del self._timestamp

    @abstractmethod
    async def next_frame(self) -> Frame:
        pass

    def av_frame(self, frame, pts, tb):
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = tb
        return av_frame

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.next_frame()
        pts, time_base = await self.next_timestamp()
        return self.av_frame(frame, pts, time_base)


class VideoStream(VSTrack):

    def __init__(self):
        super().__init__()
        self._frame_buffer: Union[Iterator, None] = None
        self._event = asyncio.Event()

    def reset(self):
        self._frame_buffer = None
        self.reset_timestamp()
        self._event.clear()

    async def next_frame(self) -> Frame:
        if self._frame_buffer is None:
            await self._event.wait()

        try:
            frame = next(self._frame_buffer)
        except StopIteration:
            self.reset()
            return await self.next_frame()

        return frame

    def stream(self, frames_buffer):
        self._frame_buffer = frames_buffer
        self._event.set()
