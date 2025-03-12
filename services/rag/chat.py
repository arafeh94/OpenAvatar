import base64
import json
import logging
import os
import random
import string
from pathlib import Path
from typing import TypedDict, List, Annotated, Sequence, Union

import fitz
import langchain_community.document_loaders as loaders
import langchain_text_splitters as text_splits
import loguru
from langchain_chroma import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from rake_nltk import Rake
from torch.fx.experimental.unification.multipledispatch.dispatcher import source


def generate_token(length: int = 32, chars: str = string.ascii_letters + string.digits) -> str:
    return ''.join(random.choices(chars, k=length))


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    enhancements: dict[str, str]
    context: List[Document]
    source: List[str]
    enhance_keywords: bool
    enhance_query: bool


class AIPrompts:
    ENHANCE_CHAPTER_DESCRIPTION = (
        "I am generating a CHAPTER SUMMARY. "
        "Summarize the following content in less than 50 words to generate the chapter description: {}"
    )
    ENHANCE_COURSE_DESCRIPTION = (
        "I am generating a course SUMMARY. "
        "Summarize the following course chapters in less than 50 words to generate the course description: {}"
    )
    IMPROVE_KEYWORDS = (
        "From the following keywords: {}, identify the most important and meaningful ones. "
        "If two or more keywords can be merged into a more meaningful phrase, do so. "
        "Return at most {} keywords as a single string of values separated by commas. "
        "Do not provide explanations or additional text—only the final list of keywords in a parsable format."
        "Avoid repeating keywords."
    )
    GENERATE_SUMMARY = (
        "Using the provided key phrases, keywords, and written summary, generate a concise summary (at least 4 "
        "sentences). While generating the new summary, follow these guidelines:\n"
        "- Select only the most relevant and meaningful key phrases, disregarding insignificant ones.\n"
        "- If keywords exist, ensure the summary aligns with them, as they represent the most frequently mentioned "
        "terms.\n"
        "- If a written summary is provided, use it as a reference but do not copy it verbatim. Instead, enhance it "
        "by incorporating additional relevant details from the key phrases and keywords.\n"
        "- Do not include any information in the summary unless you are 100% certain it is correct.\n"
        "--Guidelines Ends--\n"
        "Key Phrases: [{}] \n"
        "Keywords: [{}] \n"
        "Provided Summary: [{}].\n"
        "Ensure the generated summary is well-structured, clear, and concise. "
        "Only return the summary itself—no explanations, introductions, or additional text."
    )
    IMPROVE_INPUT = (
        "You are given a user input (question, task, etc.) and a list of enhancements. "
        "The enhancements may include keywords or document summaries that you can use to refine the user input, "
        "making it clearer, more precise, and more informative. While adhering to the following guidelines:\n"
        "- Ensure that the improved version retains all critical information from the original user input.\n"
        "- Incorporate only the most relevant enhancements, avoiding unnecessary modifications that may alter the "
        "core intent.\n"
        "- Do not remove or replace important keywords from the original user input. Instead, enhance clarity and "
        "relevance.\n"
        "- Do not provide an answer to the question; only improve its phrasing.\n\n"
        "User Input: \"{}\"\n\n"
        "Enhancements: {}\n"
        "Return only the improved version of the question without any explanation or additional text."
    )
    RETRIEVAL_1 = (
        "You are a professional AI assistant replacing the university teacher (no emojis) with retrieval-augmented "
        "generation capabilities. "
        "When given a user query, first retrieve the most relevant information. Information can be collected from the "
        "provided knowledge base, documents, and chat history. Then, generate a well-structured response that "
        "accurately integrates the retrieved information. If the retrieved content is insufficient, provide a "
        "best-effort response  while indicating uncertainty."
    )
    RETRIEVAL_2 = (
        "You are a professional AI assistant replacing the university teacher (no emojis) with retrieval-augmented "
        "generation capabilities. "
        "When given a user query, first retrieve the most relevant information. Information can be collected from the "
        "provided knowledge base, documents, keywords, summaries, and chat history. However, do not mention or refer "
        "directly to the virtual documents in your answer. Then, generate a well-structured response that accurately "
        "integrates the retrieved information. If the retrieved content is insufficient, provide a best-effort "
        "response while indicating uncertainty, but do not refer to the documents themselves. Always respond with "
        "confidence, even when the available information is limited, and avoid suggesting uncertainty unless "
        "absolutely necessary."
    )
    RETRIEVAL_DEFAULT = (
        "You are a professional AI assistant replacing the university teacher (no emojis) with retrieval-augmented "
        "generation capabilities. "
        "When given a user query, first retrieve the most relevant information. Information can be collected from the "
        "provided knowledge base, documents, keywords, summaries, and chat history. However, do not mention or refer "
        "directly to the virtual documents in your answer. Then, generate a well-structured response that accurately "
        "integrates the retrieved information.\n\n"
        "-- Rules --\n"
        "- Retrieve only the most relevant information from available sources.\n"
        "- Ensure the response is clear, coherent, and well-structured.\n"
        "- Do not explicitly reference virtual documents or chat history in the response.\n"
        "- If information is insufficient, provide a best-effort response but do not fabricate details.\n"
        "- Always respond with confidence, even when the available information is limited.\n"
        "- Avoid suggesting uncertainty unless absolutely necessary.\n"
        "-- Rules End --"
        "Very Important: RESPECT TEACHER ANSWERING RULES PER SOURCE CONTENT WHEN ANSWERING THE STUDENT PROMPT."
        "EACH SOURCE CONTENT HAS ITS OWN SET OF RULES, DO NOT MIX UP THE RULES PER CONTENT."
        "Meaning, if a content has rules a, b, and c, the other content could have other rules."
    )
    ADD_KEYWORDS = prompt = (
        "You are an AI system designed to enhance document retrieval in a Retrieval-Augmented Generation (RAG) system. "
        "Your task is to improve a given user query by appending relevant keywords from a provided list as tags at "
        "the end."
        "These tags should help the retrieval system find the most relevant documents.\n\n"
        "Instructions:\n"
        "1. Analyze the user query: {}\n"
        "2. Select the most relevant keywords from the provided list: {}\n"
        "3. Append the chosen keywords as comma-separated tags at the end of the original query.\n"
        "4. Ensure the final query remains natural and readable.\n"
        "5. Return only the modified query without explanation.\n"
        "6. If no modifications are needed, return the query as is."
    )
    DESCRIBE_IMAGE = (
        "You are an AI assistant. Given an image, return a valid JSON object. "
        "The JSON should have two keys: 'ocr' and 'summary'. "
        "The 'ocr' key should contain the extracted text from the image. "
        "The 'summary' key should provide a concise description of the image's content. "
        "Ensure the JSON output is correctly formatted without extra escape characters or invalid structures."
        "Return a valid JSON object without markdown formatting or code blocks. "
        "Ensure it's directly parsable in Python using json.loads()."
    )


class PDFImage:
    def __init__(self, name, image_bytes, ext):
        self.name = name
        self.image_bytes = image_bytes
        self.ext = ext

    def as_base64(self):
        return base64.b64encode(self.image_bytes).decode("utf-8")

    def as_bytes(self):
        return self.image_bytes

    def save(self, path=None):
        if path is None:
            path = self.name
        if path.endswith('/') or path.endswith('\\'):
            path += self.name
        with open(path, "wb") as img_file:
            img_file.write(self.as_bytes())

    def as_message(self, prompt):
        return HumanMessage(content=[
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', "image_url": {"url": f"data:image/{self.ext};base64,{self.as_base64()}"}},
        ])

    @staticmethod
    def extract_images(pdf_path):
        doc = fitz.open(pdf_path)
        images = []

        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image['ext']
                image_name = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                pdf_image = PDFImage(image_name, image_bytes, image_ext)
                images.append(pdf_image)
        return images


class Chat:

    def __init__(self, id, graph: CompiledStateGraph):
        self.id = id
        self.graph = graph
        self.logger = logging.getLogger("Chat:{}".format(id))

    def answer(self, query, source=Union[None, List], enhancements: Union[dict[str, str], None] = None,
               content_only=True) -> BaseMessage:
        enhancements = enhancements or {}
        config = {"configurable": {"thread_id": self.id}}
        query = {"query": query, "source": source, 'enhancements': enhancements}
        self.logger.info("Query: {}".format(query))
        response = self.graph.invoke(query, config)
        return response["messages"][-1] if content_only else response


class ChatService:
    def __init__(self, vector_store, llm, q_prompt=AIPrompts.RETRIEVAL_DEFAULT, k=10):
        self.vector_store = vector_store
        self.llm = llm
        self.q_prompt = q_prompt
        self.k = k
        self.logger = logging.getLogger("ChatService")

    def embed_pdf(self, path):
        ts = text_splits.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = loaders.PyPDFLoader(path).load_and_split(ts)
        docs.extend(self._extract_images(path))
        self.vector_store.add_documents(docs)
        return docs

    def _extract_images(self, path):
        images = PDFImage.extract_images(path)
        docs = []
        for image in images:
            if len(docs) > 3:
                break
            response = self.ask_ai([image.as_message(prompt=AIPrompts.DESCRIBE_IMAGE)])
            try:
                image_content = json.loads(response.content)
                ocr = image_content["ocr"]
                summary = image_content["summary"]
                ocr_doc = Document(
                    page_content=ocr,
                    metadata={"source": path, 'image': image.name, 'type': 'ocr'},
                )
                summary_doc = Document(
                    page_content=summary,
                    metadata={"source": path, 'image': image.name, 'type': 'summary'},
                )
                docs.append(ocr_doc)
                docs.append(summary_doc)
            except Exception:
                doc = Document(
                    page_content=response.content,
                    metadata={"source": path, 'image': image.name, 'type': 'gpt_error'},
                )
                docs.append(doc)

        return docs

    def get_source(self, doc: Union[Document, List[Document]]) -> str:
        if isinstance(doc, list):
            return self.get_source(doc[0])
        return doc.metadata['source']

    def get_ai_keywords_summary(self, docs, helping_summary=''):
        full_text = " ".join([doc.page_content for doc in docs])
        rake = Rake()
        rake.extract_keywords_from_text(full_text)
        keywords = self._extract_keywords(rake, ask_ai=True)
        summary = self._summarise(rake, keywords, helping_summary)
        self.logger.info("Generated Summary: {}, Keywords: {}".format(summary, keywords))
        return keywords, summary

    def _extract_keywords(self, rake, ask_ai=True):
        keywords = rake.get_word_degrees()
        sorted_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))
        top_keywords = [word for word in list(sorted_keywords.keys())[:100]]
        if ask_ai:
            keywords_str = ", ".join(top_keywords)
            ai_prompt = AIPrompts.IMPROVE_KEYWORDS.format(keywords_str, 50)
            res: BaseMessage = self.ask_ai(ai_prompt)
            top_keywords = res.content.split(',')
            self.logger.info("AI Prompt - Improve keywords: {}".format(ai_prompt))
            self.logger.info("AI Results: {}".format(res.content))
        return top_keywords

    def _summarise(self, rake, additional_keywords=None, additional_summary=''):
        keywords = additional_keywords or []
        keywords = ','.join(keywords) if isinstance(keywords, list) else keywords
        sentences = rake.get_ranked_phrases_with_scores()
        sorted_sentences = dict(sorted(sentences, key=lambda x: x[0], reverse=True))
        top_sentences = [sentence for sentence in list(sorted_sentences.values())[:100]]
        top_sentences_str = ", ".join(top_sentences)
        ai_prompt = AIPrompts.GENERATE_SUMMARY.format(top_sentences_str, keywords, additional_summary)
        res: BaseMessage = self.ask_ai(ai_prompt).content
        self.logger.info("AI Prompt - Generate Summary: {}".format(ai_prompt))
        self.logger.info("AI Results: {}".format(res))
        return res

    def _enhance_query_quality(self, query, enhancements):
        enhancement_str = ''
        for index, (key, enhancement) in enumerate(enhancements.items()):
            enhancement_str += "\t{} - {}: {}\n".format(index, key, enhancement)
        ai_prompt = AIPrompts.IMPROVE_INPUT.format(query, enhancement_str)
        enhanced_query = self.ask_ai(ai_prompt).content
        self.logger.info("AI Prompt - Improve Input: {}".format(ai_prompt))
        self.logger.info("AI Results: {}".format(enhanced_query))
        return enhanced_query

    def _enhance_query_keywords(self, query, enhancements: dict[str, str]):
        keywords = enhancements.get('keywords', None)
        if not keywords:
            return query

        keywords = ','.join(keywords) if isinstance(keywords, list) else keywords
        ai_prompt = AIPrompts.ADD_KEYWORDS.format(query, keywords)
        enhanced_query = self.ask_ai(ai_prompt).content
        self.logger.info("AI Prompt - Improve Input: {}".format(ai_prompt))
        self.logger.info("AI Results: {}".format(enhanced_query))
        return enhanced_query

    def _enhance(self, state: State):
        enhancements = state["enhancements"]
        query = state["query"]
        if not enhancements:
            return {}
        enhanced_query = query

        if state['enhance_query']:
            enhanced_query = self._enhance_query_quality(query, enhancements)
        if state['enhance_keywords']:
            enhanced_query = self._enhance_query_keywords(enhanced_query, enhancements)

        return {'query': enhanced_query}

    def _retrieve(self, state: State):
        if not state['source']:
            context = []
        else:
            _filter = {'source': {'$in': state['source']}}
            results = self.vector_store.similarity_search_with_score(state['query'], filter=_filter, k=self.k)
            context = [d for d, s in results]
            self.logger.info("Retrival Results: {}".format(context))
        return {"context": context}

    def _prompt(self, state: State):
        docs_content = "\n\n".join(
            'Source ({}) - Content: {}'.format(Path(doc.metadata['source']).name, doc.page_content)
            for doc in state["context"])
        history = "\n\n".join(self._role(msg) + ":" + msg.content for msg in state["messages"])
        context = "Document Information Start:\n" + docs_content + "\nDocument Information  End\n\n"
        context += "Chat History Start:\n" + history + "\nChat History End\n\n"
        context = "Using the following information:\n" + context + "\nAnswer the human query"
        prompt = self.q_prompt
        context = prompt + "\n" + context
        self.logger.info("AI Prompt - Retrival Context: {}".format(context))
        return {"context": context}

    def _prompt3(self, state: State):
        messages = []
        prompt = self.q_prompt
        messages.append(('system', prompt))
        for doc in state['context']:
            int_msg = 'Source ({}) - Content: {}'.format(doc.metadata['source'], doc.page_content)
            messages.append(('system', int_msg))
        for key, enhancement in state['enhancements'].items():
            int_msg = '{}: {}'.format(key, enhancement)
            messages.append(('system', int_msg))
        for message in state['messages']:
            messages.append((self._role(message), message.content))
        self.logger.info("AI Prompt - Retrival Context: {}".format(messages))
        return {"context": messages}

    def _prompt2(self, state: State):
        messages = []
        prompt = self.q_prompt
        messages.append(('system', prompt))
        for doc in state['context']:
            int_msg = 'Source ({}) - Content: {}'.format(doc.metadata['source'], doc.page_content)
            messages.append(('system', int_msg))
        for key, enhancement in state['enhancements'].items():
            int_msg = '{}: {}'.format(key, enhancement)
            messages.append(('system', int_msg))
        for message in state['messages']:
            messages.append((self._role(message), message.content))
        self.logger.info("AI Prompt - Retrival Context: {}".format(messages))
        return {"context": messages}

    def _generate(self, state: State):
        query = state["query"]
        messages = [("system", state['context']), ('human', state['query'])]
        loguru.logger.debug(messages)
        response = self.llm.invoke(messages)
        return {"messages": [HumanMessage(query), response]}

    def _generate2(self, state: State):
        query = state["query"]
        messages = [*state['context'], ('human', state['query'])]
        loguru.logger.debug(messages)
        response = self.llm.invoke(messages)
        return {"messages": [HumanMessage(query), response]}

    def _role(self, message: BaseMessage):
        return 'human' if message.__class__.__name__ == 'HumanMessage' else 'assistant'

    def new_chat(self, v=2, improve_input=True):
        sequence = [self._enhance] if improve_input else []
        sequence += [self._retrieve]
        sequence += [self._prompt, self._generate] if v == 1 else [self._prompt2, self._generate2]
        token = generate_token()
        graph_builder = StateGraph(State).add_sequence(sequence)
        graph_builder.add_edge(START, sequence[0].__name__)
        graph = graph_builder.compile(checkpointer=MemorySaver())
        return Chat(token, graph)

    def ask_ai(self, query):
        return self.llm.invoke(query)

    def dummy(self):
        self.embed_pdf('./files/1.pdf')
        self.embed_pdf('./files/2.pdf')
        self.embed_pdf('./files/3.pdf')

    @staticmethod
    def setup_nltk():
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt_tab')

    @staticmethod
    def create(prompt=AIPrompts.RETRIEVAL_DEFAULT, model_name="gpt-3.5-turbo"):
        return ChatService.create_openai(prompt=prompt, model_name=model_name)

    @staticmethod
    def create_openai(model_name='gpt-3.5-turbo', temperature=0, prompt=AIPrompts.RETRIEVAL_DEFAULT):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

        if model_name == 'deepseek-chat':
            llm = ChatOpenAI(model="deepseek-chat", temperature=0.7, openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                             openai_api_base="https://api.deepseek.com", )
        else:
            llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY'))

        vector_store = Chroma(embedding_function=embeddings, persist_directory='./vs')
        return ChatService(vector_store, llm, prompt)

    @staticmethod
    def create_ollama(model_name="gemma2:2b", temperature=0, prompt=AIPrompts.RETRIEVAL_DEFAULT):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model=model_name, temperature=temperature)

        vector_store = Chroma(embedding_function=embeddings,
                              persist_directory=os.path.join(os.path.expanduser("~"), "vector-store"))
        return ChatService(vector_store, llm, prompt)
