import json
import logging

from app.rag.ai_prompts import AIPrompts
from app.rag.chat import ChatService, Filters, QuizTextAnswer
import os

from app.rag.checkers import assert_scores, assert_ai_quiz

logging.basicConfig(level=logging.INFO)
chat_service = ChatService.create(model_name='gpt-4o-mini')

chat = chat_service.new_chat()

s = [
    QuizTextAnswer('what is int?', 'ohhh, int is awesome', 100),
    QuizTextAnswer('what is int?',
                   "In programming, 'int' is a data type that stands for 'integer.' It is used to represent whole numbers, both positive and negative, without any decimal points. For example, in languages like C, C++, and Java, 'int' can be used to declare variables that store integer values.")
]

response = chat_service.score(s, level=2)
logging.info(response)
results = assert_scores(response.content)
print(json.dumps(results, indent=2))
