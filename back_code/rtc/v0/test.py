import asyncio
import fractions
import logging
import pickle
import time
from logging import DEBUG

import numpy as np
import soundfile as sf
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame

from core.tools import utils

utils.enable_logging()


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(self, sample_rate=48000, frame_rate=30):
        super().__init__()
        self.audio_data, self.sample_rate = sf.read('../../../files/harvard.wav', dtype='int16')
        self.current_sample = 0

    async def recv(self):
        sample_size = int(AUDIO_PTIME * self.sample_rate)
        end_sample_index = self.current_sample + sample_size

        if end_sample_index >= len(self.audio_data) > self.current_sample:
            end_sample_index = len(self.audio_data)
        elif end_sample_index >= len(self.audio_data):
            raise RuntimeError("Audio sample out of range")

        samples = self.audio_data[self.current_sample:end_sample_index]
        self.current_sample += sample_size

        if hasattr(self, "_timestamp"):
            self._timestamp += len(samples)
            wait = self._start + (self._timestamp / self.sample_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        frame = AudioFrame(format="s16", layout="stereo", samples=len(samples))
        for p in frame.planes:
            p.update(samples.tobytes())

        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame


if __name__ == '__main__':
    track = AudioTrack()
    while True:
        au = track.recv()
        print(au)
