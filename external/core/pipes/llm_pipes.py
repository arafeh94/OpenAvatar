from external.core.pipeline import Pipe
from external.plugins.vllm import LLM


class LLMPipe(Pipe):
    def __init__(self, model, url, token):
        self.llm = LLM(model, url, token)

    def exec(self, obj: any, flow: []) -> any:
        return self.llm.exec(obj)
