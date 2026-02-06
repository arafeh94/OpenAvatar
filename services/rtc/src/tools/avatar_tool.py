import asyncio
import functools
from typing import TYPE_CHECKING

from av import packet

from services.rtc.context import AppContext
from services.rtc.src.tool import ToolRequest

if TYPE_CHECKING:
    from services.rtc.src.peer import ServerPeer


class AvatarTool(ToolRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        self.voice_id = None
        self.stop_streaming = False
        self.latest_task_id = None
        self.loop = asyncio.get_event_loop()
        super().__init__(**kwargs)

    def __streaming_started_listener(self, peer: 'ServerPeer'):
        self.transmit(peer, self.packet("streaming started", status='210'))

    def transmit(self, peer, packet):
        async def as_coroutine():
            peer.send_packet(packet)

        asyncio.run_coroutine_threadsafe(as_coroutine(), self.loop)

    def subscribe(self, peer: 'ServerPeer'):
        def on_task_created(value: dict) -> None:
            self.latest_task_id = value
            if not isinstance(value, dict):
                value = {'task_id': value}
            packet = self.packet({'event': 'task_created', **value}, status='200')
            self.transmit(peer, packet)

        def on_task_done(value: dict) -> None:
            if not value:
                value = self.latest_task_id
            if not isinstance(value, dict):
                value = {'task_id': value}
            self.transmit(peer, self.packet({'event': 'task_done', **value}, status='200'))

        def on_new_buffer(value: dict) -> None:
            self.transmit(peer, self.packet({'event': 'new_buffer', **value}, status='200'))

        if not peer.player.callbacks.task_created.has_subscriber():
            peer.player.callbacks.task_created.subscribe(on_task_created)
        if not peer.player.callbacks.task_done.has_subscriber():
            peer.player.callbacks.task_done.subscribe(on_task_done)
        if not peer.player.callbacks.new_buffer.has_subscriber():
            peer.player.callbacks.new_buffer.subscribe(on_new_buffer)

    def process(self, peer: 'ServerPeer'):
        self.subscribe(peer)
        if self.stop_streaming:
            peer.player.stop()
            peer.send_packet(self.packet("stopped streaming", status='211'))
        else:
            text = self.repeat if self.repeat else ""
            peer.player.on_ffs(self.__streaming_started_listener, args=(peer,))
            AppContext().run_in_thread(
                lambda: AppContext().avatar_manager.tts_buffer(self.persona, text, voice_id=self.voice_id),
                lambda buffer: peer.player.start(buffer)
            )
