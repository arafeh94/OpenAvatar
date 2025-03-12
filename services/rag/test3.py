import json
import os.path
import pickle

from external.tools.utils import enable_logging
from services.rag.chat import ChatService

enable_logging(level='INFO')
chat_service = ChatService.create(model_name='gpt-4o')

docs = chat_service.embed_pdf('./files/i1.pdf')

chat = chat_service.new_chat()

response = chat.answer("what is this document is talking about", ['./files/i1.pdf'])

print(response)
