import base64

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage

from services.rag.chat import ChatService, PDFImage

chat_service = ChatService.create(model_name='gpt-4o')

images = PDFImage.extract_images("./files/1.pdf")
image = images[0]
response = chat_service.ask_ai([image.as_message('You are an AI Instructor. Explain the following image')])
print(response)
