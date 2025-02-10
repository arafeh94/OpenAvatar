from typing import Union, Dict

from fastapi import FastAPI
import uvicorn

from services.rag.chat import ChatService, Chat

chat_service = ChatService.create()
chat_manager: Dict[str, Chat] = []
app = FastAPI()


@app.get("/register")
def register(v: int = 2):
    chat = chat_service.new_chat(v)
    chat_manager[chat.id] = chat
    return {"status": "success", 'token': chat.id}


@app.get("/answer")
def answer(token: str, question: str, sources=''):
    if token not in chat_manager:
        raise Exception('Token is not available, use /register first')
    source = sources.split(',') if sources else None
    return chat_manager[token].answer(question, source).model_dump_json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
