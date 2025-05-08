from services.rtc.context import AppContext
from services.rtc.src.tool import ToolRequest
from services.rtc.src.typing import ServerPeer


class AvatarTool(ToolRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        self.voice_id = None
        self.stop_streaming = False
        super().__init__(**kwargs)

    def process(self, peer: ServerPeer):
        if self.stop_streaming:
            peer.player.stop()
            peer.send_packet(self.packet("stopped streaming"))
        else:
            text = self.repeat if self.repeat else ""
            AppContext().run_in_thread(
                lambda: AppContext().avatar_manager.tts_buffer(self.persona, text, voice_id=self.voice_id),
                lambda buffer: peer.player.start(buffer)
            )
