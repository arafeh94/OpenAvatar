import asyncio
import threading
from services.rtc.context import AppContext
from services.rtc.src.agent import AgentRequest
from services.rtc.src.typing import ServerPeer
import time


class AvatarAgent(AgentRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        super().__init__(**kwargs)

    def process(self, peer: ServerPeer):
        # todo: this shit is taking >4 sec
        AppContext().run_in_thread(
            lambda: AppContext().avatar_manager.tts_buffer(self.persona, self.repeat, voice_id=7406),
            lambda buffer: peer.player.start(buffer)
        )
