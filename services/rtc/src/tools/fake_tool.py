from typing import TYPE_CHECKING

from services.rtc.src.tool import ToolRequest

if TYPE_CHECKING:
    from services.rtc.src.peer import ServerPeer


class FakeTool(ToolRequest):
    def __init__(self, **kwargs):
        self.data = None
        super().__init__(**kwargs)

    def process(self, peer: 'ServerPeer'):
        print(self.data)
