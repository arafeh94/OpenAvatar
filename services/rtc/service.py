import asyncio
import fractions
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Union, Tuple

import cv2
import av
import numpy as np

import soundfile as sf
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError, VIDEO_PTIME, VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, AUDIO_PTIME
from av import Packet, AudioFrame
from av.frame import Frame
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from external.core.utils.token_generator import generate_token
from external.tools import utils

utils.enable_logging()
# Create an instance of FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

peers: dict[str, 'ServerPeer'] = {}


def as_buffer(batches):
    for batch in batches:
        yield batch


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

    @abstractmethod
    async def recv(self) -> Union[Frame, Packet]:
        pass


class VideoTrack(VSTrack):

    def __init__(self):
        super().__init__()
        self.frame_buffer = as_buffer(pickle.load(open("../../files/harvard.pkl", "rb")))
        self.frame_index = 0
        self.batch = next(self.frame_buffer)
        self._start: float
        self._timestamp: int

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.frame_index >= len(self.batch):
            self.batch = next(self.frame_buffer)
            if self.batch is None:
                raise RuntimeError("Failed to read frame from MP4 file")
            self.frame_index = 0
        frame = self.batch[self.frame_index]
        self.frame_index += 1

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(self, sample_rate=48000, frame_rate=30):
        super().__init__()
        self.audio_data, self.sample_rate = sf.read('../../files/harvard.wav', dtype='int16')
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


class ServerPeer:
    def __init__(self, id: str):
        self.peer = RTCPeerConnection()
        self.channel = self.peer.createDataChannel("chat")
        self.register_events()
        self.id = id
        self.peer.addTrack(VideoTrack())
        self.peer.addTrack(AudioTrack())

    async def offer(self):
        await self.peer.setLocalDescription(await self.peer.createOffer())
        return self.peer.localDescription

    async def accept(self, remote_sdp):
        await self.peer.setRemoteDescription(remote_sdp)

    def register_events(self):
        @self.channel.on('open')
        def on_open():
            print("channel opened")

        @self.channel.on('close')
        def on_close():
            del peers[self.id]
            print(peers)

    def send_message(self, message):
        self.channel.send(message)


@app.get("/register")
async def register():
    token = generate_token()
    server = ServerPeer(token)
    sdp = await server.offer()
    peers[token] = server
    return {"sdp": sdp, 'token': token}


@app.get("/confirm")
async def confirm(token, sdp):
    if token in peers.keys():
        server = peers[token]
        client_sdp = json.loads(sdp)
        client_sdp = RTCSessionDescription(client_sdp['sdp'], client_sdp['type'])
        await server.accept(client_sdp)
        return {"status": "accepted"}
    return {"status": "not registered"}


@app.get("/broadcast")
async def broadcast(message):
    for peer in peers.values():
        peer.send_message(message)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
