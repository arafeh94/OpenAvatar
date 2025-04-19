from services.rtc.context import AppContext
from services.rtc.src.agent import AgentRequest
from services.rtc.src.typing import ServerPeer


class AvatarAgent(AgentRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        self.voice_id = None
        super().__init__(**kwargs)

    def process(self, peer: ServerPeer):
        if self.voice_id is None:
            self.voice_id = 7406
        AppContext().run_in_thread(
            lambda: AppContext().avatar_manager.tts_buffer(self.persona, self.repeat, voice_id=self.voice_id),
            lambda buffer: peer.player.start(buffer)
        )
