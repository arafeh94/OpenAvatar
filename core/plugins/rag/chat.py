import base64
import json
import logging
import os
import random
import string
from logging import Logger
from typing import TypedDict, List, Annotated, Sequence, Union, AsyncIterator, Any

import fitz
import langchain_text_splitters as text_splits
import loguru
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm

from core.plugins.rag.ai_prompts import AIPrompts
from core.tools.atomic_id import AtomicID


def generate_token(length: int = 32, chars: str = string.ascii_letters + string.digits) -> str:
    return ''.join(random.choices(chars, k=length))


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    query_rag: str
    enhancements: dict[str, str]
    state_config: dict[str, str]
    context: List[Document]
    source: List[str]
    rules: dict[str, str]
    enhance_keywords: bool
    enhance_query: bool
    enhance_rag: bool
    is_quiz: bool
    enhance_quiz: bool


class PDFImage:
    def __init__(self, name, image_bytes, ext, page):
        self.name = name
        self.image_bytes = image_bytes
        self.ext = ext
        self.page = page

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
                pdf_image = PDFImage(image_name, image_bytes, image_ext, page_num + 1)
                images.append(pdf_image)
        return images


class QuizTextAnswer:
    def __init__(self, question: str, answer: str, id=None):
        self.question = question
        self.answer = answer
        self.id = id or AtomicID().fetch()


class Chat:

    def __init__(self, id, graph: CompiledStateGraph):
        self.id = id
        self.graph: CompiledStateGraph = graph
        self.logger = logging.getLogger("Chat:{}".format(id))

    def answer(self, query, source=Union[None, List], enhancements: Union[dict[str, str], None] = None,
               rules: Union[dict[str, str], None] = None, enhance_rag=True, stream=False, is_quiz=False, **kwargs) -> \
            Union[dict[str, Any], AsyncIterator]:
        """
        answer user queries while taking into consideration the chat history
        :param enhance_rag: improve rag querying by integrating the previous history to focus on the context
        :param query: user query
        :param source: an array of file path to limit the retrival ['1.pdf', '2.pdf']
        :param enhancements: enhancements applied on the system
        :param rules: separate rules from the enhancement applied on the system prompt
        :param stream: return streaming events for llm response
        :param is_quiz: force the retrival to ignore the query for similarity search and return all source docs.
                        also it affect the prompt in the system
        :return:  Union[dict[str, Any], AsyncIterator]
        """
        enhancements = enhancements or {}
        rules = rules or {}
        config = {"configurable": {"thread_id": self.id}}
        invoke_query = {"query": query, "source": source, 'enhancements': enhancements, 'rules': rules,
                        'enhance_rag': enhance_rag, 'is_quiz': is_quiz}
        invoke_query.update(**kwargs)
        if stream:
            return self.graph.astream(invoke_query, config, stream_mode=["messages", "values"])
        else:
            return self.graph.invoke(invoke_query, config)


class Filters:
    @staticmethod
    def length_filter(length):
        def filter_fn(docs):
            return list(filter(lambda doc: len(doc.page_content) >= length, docs))

        return filter_fn


class ChatService:
    def __init__(self, vector_store, llm, q_prompt=AIPrompts.SYSTEM_PROMPT, k=10, max_history_size=10):
        self.vector_store: VectorStore = vector_store
        self.llm: BaseChatModel = llm
        self.q_prompt = q_prompt
        self.k = k
        self.max_history_size = max_history_size
        self.logger = logging.getLogger("ChatService")

    def embed_document(self, document):
        self.vector_store.add_documents([document])
        return [document]

    def embed_pdf(self, path, metadata=None, filters=None):
        metadata = metadata or {}
        filters = filters or []
        filters = [filters] if not isinstance(filters, list) else filters
        ts = text_splits.RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = PyPDFLoader(path).load_and_split(ts)
        docs.extend(self._extract_images(path))
        for filter in filters:
            docs = filter(docs)
        for doc in docs:
            doc.page_content = "[CHUNK START][PAGE:{}]{}[CHUNK END]\n".format(self._get_label(doc), doc.page_content)
        [doc.metadata.update(metadata) for doc in docs]
        docs and self.vector_store.add_documents(docs)
        return docs

    def _extract_images(self, path):
        images = PDFImage.extract_images(path)
        docs = []
        for image in tqdm(images):
            response = self.ask_ai([image.as_message(prompt=AIPrompts.DESCRIBE_IMAGE)])
            try:
                image_content = json.loads(response.content)
                ocr = image_content["ocr"]
                smr = image_content["summary"]
                str_format = "[IMAGE START][TRANSCRIPT START]{}[TRANSCRIPT END]\n[SUMMARY START]: {}[SUMMARY END][IMAGE END]"
                ocr_doc = Document(
                    page_content=str_format.format(ocr, smr),
                    metadata={"source": path, 'image': image.name, 'type': 'image', 'page': image.page,
                              'page_label': image.page})
                docs.append(ocr_doc)
            except Exception as e:
                self.logger.error(
                    "Error while extracting image: Error: {}. Response: {}. Image:{}".format(e, response, image))
        return docs

    def get_source(self, doc: Union[Document, List[Document]]) -> str:
        if isinstance(doc, list):
            return self.get_source(doc[0])
        return doc.metadata['source']

    def get_ai_keywords_summary(self, docs, helping_summary=''):
        return '', ''

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

    def _enhance_rag_query(self, history, query):
        ai_prompt = AIPrompts.ENHANCE_RAG_SIMPLE.format(history, query)
        results = self.ask_ai(ai_prompt).content
        self.logger.info("AI Prompt - Improve RAG Query: {}".format(ai_prompt))
        self.logger.info("AI Results: {}".format(results))
        return results

    def _enhance(self, state: State):
        enhancements = state["enhancements"]
        query = state["query"]
        if not enhancements:
            return {}
        enhanced_query = query
        query_rag = query

        if 'enhance_query' in state and state['enhance_query']:
            enhanced_query = self._enhance_query_quality(query, enhancements)
        if 'enhance_keywords' in state and state['enhance_keywords']:
            enhanced_query = self._enhance_query_keywords(enhanced_query, enhancements)
        if 'enhance_rag' in state and state['enhance_rag']:
            query_rag = self._enhance_rag_query(self._history(state), enhanced_query)

        return {'query': enhanced_query, 'query_rag': query_rag}

    def _retrieve(self, state: State):
        context = []
        if state['source']:
            if state['is_quiz']:
                context = self._source_retrieve(state)
            else:
                context = self._similarity_search(state)
        not context and self.logger.error("Context retrieve - No results - State: {}".format(state))
        return {"context": context}

    def _source_retrieve(self, state: State):
        _filter = {'source': {'$in': state['source']}}
        results = self.vector_store.similarity_search("", k=999, filter=_filter)
        if 'enhance_quiz' in state and state['enhance_quiz']:
            ai_prompt = AIPrompts.QUIZ_SUMMARY.format('\n'.join(['{}'.format(doc.page_content) for doc in results]))
            self.logger.info("AI Prompt - Context Retrieve: {}".format(ai_prompt))
            ai_results = self.ask_ai(ai_prompt).content
            self.logger.info("AI Prompt - Quiz Summary Results: {}".format(results))
            results = [Document(ai_results, metadata={'source': 'ai', 'page': 1, 'page_label': 1})]
        return results

    def _similarity_search(self, state: State):
        context = []
        _filter = {'source': {'$in': state['source']}}
        results = self.vector_store.similarity_search_with_score(state['query_rag'], filter=_filter, k=self.k)
        for doc, score in results:
            doc.metadata.update({'score': score})
            context.append(doc)
        return context

    def _history(self, state: State):
        return "\n\n".join(self._role(msg) + ":" + msg.content for msg in state["messages"])

    def _get_label(self, doc):
        return doc.metadata['page_label'] if 'page_label' in doc.metadata \
            else doc.metadata['page'] if 'page' in doc.metadata else 'n/a'

    def _prompt(self, state: State):
        prompt = self.q_prompt
        system_message = prompt

        system_message += "[Answering Rules Start]"
        for key, rule in state['rules'].items():
            int_msg = '{}: {}'.format(key, rule)
            system_message += "\n" + int_msg
        system_message += "[Answering Rules End]"

        system_message += "[Retrieved Course Content Start]"
        for doc in state['context']:
            int_msg = '{}'.format(doc.page_content)
            system_message += "\n" + int_msg
        system_message += "[Retrieved Course Content End]"

        for key, enhancement in state['enhancements'].items():
            int_msg = '{}: {}'.format(key, enhancement)
            system_message += "\n" + int_msg

        context: list = [SystemMessage(system_message)]

        last_messages = state['messages'][-self.max_history_size:]
        for message in last_messages:
            if self._role(message) == 'human':
                context.append(HumanMessage(message.content))
            else:
                context.append(AIMessage(message.content))

        self.logger.info("AI Prompt - Retrival Context: {}".format(context))
        return {"context": context}

    def _generate(self, state: State):
        query = state["query"]
        messages = [*state['context'], HumanMessage(state['query'])]
        response = self.llm.invoke(messages)
        self.logger.info("AI Prompt - Generated Context: {}".format(messages))
        self.logger.info("AI Prompt - Generated Response: {}".format(response))
        return {"messages": [HumanMessage(query), response]}

    def _role(self, message: BaseMessage):
        return 'human' if message.__class__.__name__ == 'HumanMessage' else 'assistant'

    def new_chat(self, v=1):
        sequence = [self._enhance, self._retrieve, self._prompt, self._generate]
        token = generate_token()
        graph_builder = StateGraph(State).add_sequence(sequence)
        graph_builder.add_edge(START, sequence[0].__name__)
        graph = graph_builder.compile(checkpointer=MemorySaver())
        return Chat(token, graph)

    def ask_ai(self, query, stream=False):
        if stream:
            return self.llm.astream(query)
        else:
            return self.llm.invoke(query)

    def score(self, samples: list[QuizTextAnswer], level=0):
        if not 0 <= level <= 9:
            raise ValueError("level must be between 0 and 9")
        j = '['
        for sample in samples:
            sample_formatted = AIPrompts.TEMPLATE_QUIZ_ANSWER.format(sample.id, sample.question, sample.answer) + ','
            sample_formatted = AIPrompts.format(sample_formatted)
            j += sample_formatted
        j = j.rstrip(',') + ']'
        ai_prompt = AIPrompts.QUIZ_SCORE_ANSWERS.format(AIPrompts.TEACHERS_TYPES[level], j)
        self.logger.info("AI Prompt - Score: {}".format(ai_prompt))
        response = self.ask_ai(ai_prompt)
        self.logger.info("AI Prompt - Generated Response: {}".format(response))
        return response

    @staticmethod
    def create(prompt=AIPrompts.SYSTEM_PROMPT, model_name="gpt-3.5-turbo", max_tokens=2000):
        return ChatService.create_openai(prompt=prompt, model_name=model_name, max_tokens=max_tokens)

    @staticmethod
    def create_openai(model_name='gpt-3.5-turbo', temperature=0.2, prompt=AIPrompts.SYSTEM_PROMPT, max_tokens=2000):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

        if model_name == 'deepseek-chat':
            llm = ChatOpenAI(model="deepseek-chat", temperature=0.7, openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                             openai_api_base="https://api.deepseek.com", max_tokens=max_tokens)
        else:
            llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')
                             , max_tokens=max_tokens)

        vector_store = Chroma(embedding_function=embeddings,
                              persist_directory=os.path.join(os.path.expanduser("~"), "vector-store"))
        return ChatService(vector_store, llm, prompt)

    @staticmethod
    def create_ollama(model_name="gemma2:2b", temperature=0, prompt=AIPrompts.SYSTEM_PROMPT):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model=model_name, temperature=temperature)

        vector_store = Chroma(embedding_function=embeddings,
                              persist_directory=os.path.join(os.path.expanduser("~"), "vector-store"))
        return ChatService(vector_store, llm, prompt)
