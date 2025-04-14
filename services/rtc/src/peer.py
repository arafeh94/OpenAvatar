import logging
import inspect

from typing import Callable
from aiortc import RTCPeerConnection
from services.rtc.src.agent import Requests
from services.rtc.src.tracks.avatar_player import AvatarMediaPlayer


class ServerPeer:
    def __init__(self, token: str, on_close: Callable[[str], None]):
        self.__token = token
        self._on_close = on_close

        self.__peer = RTCPeerConnection()
        self.__channel = self.peer.createDataChannel("chat")

        self.__player = AvatarMediaPlayer(token, "lisa_casual_720_pl")
        self.__peer.addTrack(self.player.video)
        self.__peer.addTrack(self.player.audio)

        self._register_events()

        self.logger = logging.getLogger("Peer#{}".format(self.__token))

    @property
    def player(self):
        return self.__player

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

    async def accept(self, remote_sdp):
        await self.__peer.setRemoteDescription(remote_sdp)

    def _register_events(self):
        @self.channel.on('open')
        def on_open():
            print("channel opened")

        @self.channel.on('close')
        def on_close():
            self._on_close(self.token)

        @self.channel.on('message')
        async def on_message(message):
            self.logger.info("message received: {}".format(message))
            data = Requests(message)
            for agent_request in data.parse_agents():
                if agent_request.is_valid():
                    process = agent_request.process
                    await process(self) if inspect.iscoroutinefunction(process) else process(self)

    def send_message(self, message):
        self.__channel.send(message)
