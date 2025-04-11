import logging
import inspect

from typing import Callable
from aiortc import RTCPeerConnection
from services.rtc.src.agent import Requests
from services.rtc.src.audio_track import AudioStream
from services.rtc.src.video_track import VideoStream


class ServerPeer:
    def __init__(self, token: str, on_close: Callable[[str], None]):
        self.token = token
        self.on_close = on_close

        self.peer = RTCPeerConnection()

        self.channel = self.peer.createDataChannel("chat")

        self.video_track = VideoStream()
        self.peer.addTrack(self.video_track)

        self.audio_track = AudioStream(AudioStream.MONO)
        self.peer.addTrack(self.audio_track)

        self.register_events()

        self.logger = logging.getLogger("Peer#{}".format(self.token))

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
            self.on_close(self.token)

        @self.channel.on('message')
        async def on_message(message):
            self.logger.info("message received: {}".format(message))
            data = Requests(message)
            for agent_request in data.parse_agents():
                if agent_request.is_valid():
                    process = agent_request.process
                    await process(self) if inspect.iscoroutinefunction(process) else process(self)

    def send_message(self, message):
        self.channel.send(message)
