import logging
from app.rag.chat import ChatService, Filters
import os

logging.basicConfig(level=logging.INFO)
chat_service = ChatService.create(model_name='gpt-4o')

docs = chat_service.embed_pdf('../files/Chapter4 - Threads.pdf', {'source': '../files/Chapter4 - Threads.pdf'},
                              Filters.length_filter(4))

chat = chat_service.new_chat()

response = chat.answer("what is the document is talking about", ['../files/Chapter4 - Threads.pdf'], enhancements={
    'Summary': 'introduction to physics',
})

logging.info(response['messages'][-1].content)
