from typing import Callable

from aiortc import RTCPeerConnection

from core.plugins.lip_sync.core.avatar import AvatarManager
from services.rtc.src.video_track import VideoStream


class ServerPeer:
    def __init__(self, token: str, on_close: Callable[[str], None], avatar_manager: AvatarManager):
        self.token = token
        self.on_close = on_close
        self.avatar_manager = avatar_manager

        self.peer = RTCPeerConnection()

        self.channel = self.peer.createDataChannel("chat")
        self.avatar_video = VideoStream()
        self.peer.addTrack(self.avatar_video)

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
            self.on_close(self.token)

    def send_message(self, message):
        self.channel.send(message)

    def stream(self, message, persona):
        buffer = self.avatar_manager.tts_buffer(persona, message, voice_id=7406)
        frames, audio = next(buffer)
        self.avatar_video.stream(frames)
