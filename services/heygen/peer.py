import fractions
import inspect
import logging
import pickle
from typing import Callable

from aiortc import RTCPeerConnection, MediaStreamTrack
from livekit.rtc import VideoBufferType

from core.plugins.lip_sync.core.decoder import AvatarVideoDecoder
from services.heygen.heygen import Heygen
from services.rtc.src.tool import Requests, Packet
from services.rtc.src.tracks.avatar_player import AvatarMediaPlayer
import asyncio
import logging
import aiohttp
import av
import numpy as np
from aiortc import MediaStreamTrack
from livekit import rtc
import cv2


class LiveKitAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        # self.livekit_track = livekit_track
        self.livekit_track = asyncio.Future()
        self.queue = asyncio.Queue()
        self.pts = 0
        self.frames = []

        asyncio.create_task(self._pull_frames())

    async def _pull_frames(self):
        livekit_track = await self.livekit_track

        async for frame_event in livekit_track:
            audio_frame = frame_event.frame
            samples = np.frombuffer(audio_frame.data, dtype=np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format='s16', layout='mono')
            frame.sample_rate = 48000
            frame.pts = self.pts
            frame.time_base = fractions.Fraction(1, frame.sample_rate)

            self.pts += frame.samples
            await self.queue.put(frame)

    async def recv(self):
        frame = await self.queue.get()
        return frame


class LiveKitVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        # self.livekit_track = livekit_track
        self.livekit_track = asyncio.Future()
        self.queue = asyncio.Queue()
        asyncio.create_task(self._pull_frames())

    async def _pull_frames(self):
        livekit_track = await self.livekit_track
        async for frame_event in livekit_track:
            video_frame = frame_event.frame
            rgb_frame = video_frame.convert(VideoBufferType.RGB24)
            rgb_np = np.frombuffer(rgb_frame.data, dtype=np.uint8)
            rgb_np = rgb_np.reshape((rgb_frame.height, rgb_frame.width, 3))

            # Create aiortc frame
            aiortc_frame = av.VideoFrame.from_ndarray(rgb_np, format="rgb24")
            aiortc_frame.pts = int(frame_event.timestamp_us * AvatarVideoDecoder.CLOCK_RATE / 1_000_000)
            aiortc_frame.time_base = fractions.Fraction(1, AvatarVideoDecoder.CLOCK_RATE)

            # Put frame in queue
            await self.queue.put(aiortc_frame)

    async def recv(self):
        frame = await self.queue.get()
        return frame


class HeygenPeer:
    def __init__(self, token: str, heygen: Heygen, on_close: Callable[[str], None]):
        self.__token = token
        self._on_close = on_close
        self.heygen = heygen

        self.__peer = RTCPeerConnection()
        self.__channel = self.peer.createDataChannel("chat")
        self.__audio_track = LiveKitAudioTrack()
        self.__video_track = LiveKitVideoTrack()
        self.__peer.addTrack(self.__audio_track)
        self.__peer.addTrack(self.__video_track)

        self._register_events()

        self.logger = logging.getLogger("Peer#{}".format(self.__token))

    @property
    def peer(self):
        return self.__peer

    @property
    def channel(self):
        return self.__channel

    @property
    def token(self):
        return self.__token

    async def offer(self):
        await self.__peer.setLocalDescription(await self.peer.createOffer())
        return self.__peer.localDescription

    async def close(self):
        await self.heygen.close_session()
        await self.__peer.close()

    async def accept(self, remote_sdp):
        await self.__peer.setRemoteDescription(remote_sdp)

    def _register_events(self):
        @self.peer.on("connectionstatechange")
        async def on_connection():
            if self.peer.connectionState == "connected":
                logging.info("Initiating heygen connection")
                await self.heygen.create_session()
                while not self.heygen.is_ready():
                    print("waiting for heygen connection")
                    await asyncio.sleep(1)
                self.__video_track.livekit_track.set_result(self.heygen.video_track)
                self.__audio_track.livekit_track.set_result(self.heygen.audio_track)

        @self.channel.on('open')
        async def on_open():
            ...

        @self.channel.on('close')
        async def on_close():
            await self.close()
            self._on_close(self.token)

        @self.channel.on('message')
        async def on_message(message):
            self.logger.info("DataChannel: {}".format(message))
            data = Requests(message)
            for tool_request in data.parse_tools():
                process = tool_request.process
                await process(self) if inspect.iscoroutinefunction(process) else process(self)

    def send_message(self, message):
        self.__channel.send(message)

    def send_packet(self, packet: Packet):
        self.__channel.send(packet.as_json())
