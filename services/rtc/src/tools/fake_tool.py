from services.rtc.src.tool import ToolRequest


# noinspection PyUnresolvedReferences
class FakeTool(ToolRequest):
    def __init__(self, **kwargs):
        self.data = None
        super().__init__(**kwargs)

    def process(self, peer: 'ServerPeer'):
        print(self.data)
