import os
import re
import threading
import time
from typing import override

from openai import OpenAI, AssistantEventHandler


def clean_string(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



class FileAI:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(FileAI, cls).__new__(cls)
            cls.__instance.__init_once()
        return cls.__instance

    def __init_once(self):
        self.__client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.__model_name = "gpt-4o"

    def _wait_for_run_completion(self, thread_id, run_id, interval=2):
        while True:
            run_status = self.__client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run_status.status == "completed":
                return
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run {run_status.status}")
            time.sleep(interval)

    def _get_assistant_response(self, thread_id):
        messages = self.__client.beta.threads.messages.list(thread_id=thread_id)
        for message in reversed(messages.data):
            print(message)
            if message.role == "assistant":
                return message.content[0].text.value
        return "No assistant response found."

    def query(self, file_path, instruction, query: str):
        file = self.__client.files.create(
            file=open(file_path, "rb"),
            purpose='assistants'
        )

        assistant = self.__client.beta.assistants.create(
            instructions=instruction,
            model=self.__model_name,
            tools=[{"type": "code_interpreter"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": [file.id]
                }
            }
        )

        thread = self.__client.beta.threads.create()
        self.__client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )

        run = self.__client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        self._wait_for_run_completion(thread_id=thread.id, run_id=run.id)
        response = self._get_assistant_response(thread_id=thread.id)
        return response


