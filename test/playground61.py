import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./local_neo4jWorkDir2"

# PYTHONUNBUFFERED=1;NEO4J_DATABASE=neo4j;NEO4J_PASSWORD=neo4j123!;NEO4J_USERNAME=neo4j;NEO4J_URI=neo4j://localhost:7687

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1:8b",
    llm_model_max_async=2,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    graph_storage="Neo4JStorage",
    log_level="INFO",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

ts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_paths = ["./files/1.pdf", "./files/2.pdf", "./files/3.pdf"]
# doc_paths = ["./files/1.pdf"]
documents = []
for path in doc_paths:
    documents.extend(PyPDFLoader(path).load_and_split(ts))

text = ''
for doc in documents:
    text += doc.page_content

rag.insert(text)

# results = rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# print(results)
