from typing import TYPE_CHECKING

from services.rtc.src.tool import ToolRequest

if TYPE_CHECKING:
    from services.heygen.peer import HeygenPeer


class HeygenTool(ToolRequest):
    def __init__(self, **kwargs):
        self.text = None
        self.interrupt = True
        super().__init__(**kwargs)

    async def process(self, peer: 'HeygenPeer'):
        if self.interrupt:
            await peer.heygen.interrupt()
        await peer.heygen.repeat(self.text)
        print(self.text)
