from core.plugins.rag.chat import ChatService

chat_service = ChatService.create(model_name='gpt-4o-mini', max_tokens=16384)
r = chat_service.ask_ai("how are you tody", stream=True)
for rr in r:
    print(rr.content)
