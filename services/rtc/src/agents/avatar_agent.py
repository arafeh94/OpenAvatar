import asyncio
import threading
from services.rtc.context import AppContext
from services.rtc.src.agent import AgentRequest
from services.rtc.src.typing import ServerPeer


class AvatarAgent(AgentRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        super().__init__(**kwargs)

    def process(self, peer: ServerPeer):
        buffer = AppContext().avatar_manager.tts_buffer(self.persona, self.repeat, voice_id=7406)
        peer.player.start(buffer)
