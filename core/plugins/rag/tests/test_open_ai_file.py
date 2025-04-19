from app.rag.tools import FileAI

response = FileAI().query('./test_open_ai_file.py','you are a code instructor','explain the code for me')
print(response)