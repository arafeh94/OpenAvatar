from openai import OpenAI

from src.core.interfaces.base_service import BaseService


class LLM(BaseService):

    def __init__(self, model, base_url, api_key):
        self.model = model
        self.client = None
        self.process = None
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        if not self.is_hosting():
            raise Exception('VLLM is not working. Please try again after starting the VLLM server.')

    def is_hosting(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello!"}]
            )
            return True
        except Exception as ex:
            return False

    def exec(self, *args, **kwargs):
        prompt = self.extract_first('prompt', *args, **kwargs)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
