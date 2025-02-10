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
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError, VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, AUDIO_PTIME
from av import Packet, AudioFrame
from av.frame import Frame
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from external.core.utils.token_generator import generate_token
from external.plugins.lip_sync.core.avatar import AvatarManager
from external.plugins.lip_sync.core.models import AvatarWave2LipModel
from external.plugins.text2speech import MicrosoftText2Speech, Audio
from external.tools import utils
from services.rtc.src.audio_track import AudioStream, MonoFrame
from services.rtc.src.video_track import VideoStream

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
avatar_manager = AvatarManager(AvatarWave2LipModel(), MicrosoftText2Speech())


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

    def reset_timestamp(self):
        self._start = time.time()
        self._timestamp = 0

    @abstractmethod
    async def recv(self) -> Union[Frame, Packet]:
        pass


class VideoTrack(VSTrack):

    def __init__(self):
        super().__init__()
        self.frame_index = 0
        self.live_stream = None
        self.live_buffer = None
        self._start: float
        self._timestamp: int
        self.event = asyncio.Event()

    def live_stream_monitor(self):
        if self.live_stream is None:
            return
        if self.frame_index >= len(self.live_stream):
            self.frame_index = 0
            try:
                self.live_stream = next(self.live_buffer)
            except StopIteration:
                self.live_buffer = None
                self.live_stream = None
                self.event.clear()

    async def next_frame(self):
        self.live_stream_monitor()

        if self.live_buffer is None:
            self.frame_index = 0
            await self.event.wait()
            self.reset_timestamp()

        frame = self.live_stream[self.frame_index]
        self.frame_index += 1
        return frame

    def av_frame(self, frame, pts, tb):
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = tb
        return av_frame

    def stream(self, frames_buffer):
        self.live_buffer = frames_buffer
        self.live_stream = next(frames_buffer)
        self.event.set()

    async def recv(self):
        frame = await self.next_frame()
        pts, time_base = await self.next_timestamp()
        return self.av_frame(frame, pts, time_base)


class ServerPeer:
    def __init__(self, id: str):
        self.id = id
        self.peer = RTCPeerConnection()

        self.channel = self.peer.createDataChannel("chat")
        # self.avatar_video = VideoStream()
        self.avatar_audio = AudioStream(AudioStream.MONO)

        # self.peer.addTrack(self.avatar_video)
        self.peer.addTrack(self.avatar_audio)
        self.register_events()

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

    def stream(self, message):
        buffer = avatar_manager.tts_buffer('lisa_casual_720_pl', message, voice_id=7406)
        frames, audio = next(buffer)
        # self.avatar_video.stream(frames)
        self.avatar_audio.stream(audio)
        # @continue from here
        # while True:
        #     try:
        #         frames, audios = next(buffer)
        #         self.avatar_video.stream(frames)
        #     except StopIteration:
        #         break


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
    return {"status": "accepted"}


@app.get("/answer")
async def answer(message, token):
    if token not in peers.keys():
        return {"status": "not registered"}
    peer = peers[token]
    peer.stream(message)
    return {"status": "accepted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
