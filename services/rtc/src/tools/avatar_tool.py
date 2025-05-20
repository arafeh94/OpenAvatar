import asyncio
from typing import TYPE_CHECKING

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
        self.loop = asyncio.get_event_loop()
        super().__init__(**kwargs)

    def __streaming_started_listener(self, peer: 'ServerPeer'):
        async def as_coroutine():
            peer.send_packet(self.packet("streaming started", status='210'))

        asyncio.run_coroutine_threadsafe(as_coroutine(), self.loop)

    def process(self, peer: 'ServerPeer'):
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
