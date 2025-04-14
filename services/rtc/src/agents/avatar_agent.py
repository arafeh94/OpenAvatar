import asyncio
import threading
import time
from services.rtc.context import AppContext
from services.rtc.src.agent import AgentRequest


class AvatarAgent2(AgentRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        super().__init__(**kwargs)

    def process(self, peer):
        self.astream(peer, self.repeat, self.persona)

    async def stream(self, peer, message, persona):
        buffer = AppContext().avatar_manager.tts_buffer(persona, message, voice_id=7406)
        while True:
            try:
                video, audio = next(buffer)
                await peer.video_track.stream(video)
                await peer.audio_track.stream(audio)
                await asyncio.sleep((len(audio.samples) / audio.sampling_rate) * 0.8)
            except StopIteration:
                return

    def _new_thread(self, peer, message, persona):
        asyncio.run(self.stream(peer, message, persona))

    def astream(self, peer, message, persona):
        stream_thread = threading.Thread(target=self._new_thread, args=(peer, message, persona))
        stream_thread.start()
        return stream_thread


class AvatarAgent(AgentRequest):
    def __init__(self, **kwargs):
        self.repeat = None
        self.persona = None
        super().__init__(**kwargs)

    def process(self, peer):
        buffer = AppContext().avatar_manager.tts_buffer(self.persona, self.repeat, voice_id=7406)
        peer.player.start(buffer)
