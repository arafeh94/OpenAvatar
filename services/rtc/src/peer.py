import inspect
import logging
from typing import Callable

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

from services.rtc.src.tool import Requests, Packet
from services.rtc.src.tracks.avatar_player import AvatarMediaPlayer


class ServerPeer:
    def __init__(self, token: str, persona: str, on_close: Callable[[str], None]):
        self.__token = token
        self._on_close = on_close

        self.__peer = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        ))
        self.__channel = self.peer.createDataChannel("chat")

        self.__player = AvatarMediaPlayer(token, persona)
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

    async def close(self):
        await self.__peer.close()

    async def accept(self, remote_sdp):
        await self.__peer.setRemoteDescription(remote_sdp)

    def _register_events(self):
        @self.channel.on('open')
        def on_open():
            ...

        @self.channel.on('close')
        async def on_close():
            print("closing connection")
            self.player.quit()
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
        as_json = packet.as_json()
        print("sending: {}".format(as_json))
        self.__channel.send(as_json)
