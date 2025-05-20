from typing import List, Optional, TYPE_CHECKING

from services.rtc.context import AppContext
from services.rtc.src.tool import ToolRequest

if TYPE_CHECKING:
    from services.rtc.src.peer import ServerPeer


class LLMTool(ToolRequest):
    def __init__(self, **kwargs):
        self.query: Optional[str] = None
        self.history: Optional[List] = None
        self.stream: Optional[bool] = True
        super().__init__(**kwargs)

    async def process(self, peer: ServerPeer):
        stream = AppContext().chat.ask_ai(self.query, stream=self.stream)
        async for message in stream:
            peer.send_packet(self.packet(message.content))
        peer.send_packet(self.end_packet())
