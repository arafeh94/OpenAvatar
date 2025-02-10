import langchain_community.document_loaders as loaders
import langchain_text_splitters as text_splits
from langchain_core.messages import BaseMessage
from rake_nltk import Rake

from external.tools.utils import enable_logging
from services.rag.chat import ChatService, Chat

service = ChatService.create()
from services.rag.chat import ChatService

enable_logging()

ts = text_splits.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = loaders.PyPDFLoader('./files/1.pdf').load_and_split(ts)
full_text = " ".join([doc.page_content for doc in docs])
rake = Rake()
rake.extract_keywords_from_text(full_text)
keywords = rake.get_word_degrees()
sorted_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))
top_100_keywords = [word for word in list(sorted_keywords.keys())[:100]]
keywords_str = ", ".join(top_100_keywords)

ai_prompt = """
From the following keywords: {}, identify the most important and meaningful ones.  
If two or more keywords can be merged into a more meaningful phrase, do so.  
Return at most {} keywords as a single string of values separated by commas.  
Do not provide explanations or additional text—only the final list of keywords in a parsable format.
"""

res: BaseMessage = service.ask_ai(ai_prompt.format(keywords_str, 50))
print(res.content.split(','))
print(keywords_str)
