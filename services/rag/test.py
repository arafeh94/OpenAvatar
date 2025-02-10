import json
import os.path
import pickle

from external.tools.utils import enable_logging
from services.rag.chat import ChatService

enable_logging(level='INFO')
chat_service = ChatService.create()
chat_service.dummy() if False else ''

enhancement_1 = ("the main documents is talking about improving non-iid in privacy preserving federated learning "
                 "during the warmup phase (initialization) using genetic algorithms.")
if os.path.exists('ai_enhance.pkl'):
    ai_enhance = pickle.load(open('ai_enhance.pkl', 'rb'))
else:
    docs = chat_service.embed_pdf('./files/1.pdf')
    ai_enhance = chat_service.get_ai_keywords_summary(docs, enhancement_1)
    pickle.dump(ai_enhance, open('ai_enhance.pkl', 'wb'))
enhancements = {
    "keywords": ','.join(ai_enhance['keywords']),
    "summary_1": ai_enhance['summary'],
    "summary_2": enhancement_1,
}
print(json.dumps(enhancements, indent=2))
chat = chat_service.new_chat(v=2)
source = './files/1.pdf'.split(',')

while True:
    question = input("Question: ")
    if question == 'quit':
        break
    chat.answer(question, source, enhancements).pretty_print()
