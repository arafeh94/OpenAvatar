from services.rtc.src.agent import AgentRequest


# noinspection PyUnresolvedReferences
class FakeAgent(AgentRequest):
    def __init__(self, **kwargs):
        self.data = None
        super().__init__(**kwargs)

    def process(self, peer: 'ServerPeer'):
        print(self.data)
