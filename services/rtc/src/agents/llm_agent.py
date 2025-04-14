import asyncio
import threading
from services.rtc.context import AppContext
from services.rtc.src.agent import AgentRequest

static_response = """
<div class="search-engine-description">
    <p>A search engine is a software system designed to find information on the World Wide Web. It works by:</p>
    <p>
        <strong>Crawling:</strong> Search engines use automated programs called "crawlers" or "spiders" to scan the internet, discovering new and updated web pages by following links.
    </p>
    <p>
        <strong>Indexing:</strong> The information gathered by crawlers is then analyzed and stored in a massive database called an "index." This index organizes the content of web pages, including keywords, links, and other relevant data.
    </p>
    <p>
        <strong>Ranking:</strong> When a user enters a search query, the search engine's algorithms analyze the query and compare it to the information in the index. It then ranks the most relevant web pages based on various factors to present the user with a list of search results.
    </p>
    <p><strong>Key aspects of a search engine:</strong></p>
    <ul>
        <li>
            <strong>Algorithms:</strong> Complex sets of rules and processes that determine the relevance and ranking of search results. These algorithms are constantly updated to improve accuracy and combat spam.
        </li>
        <li>
            <strong>Index:</strong> A vast database containing information about the billions of web pages the search engine has crawled.
        </li>
        <li>
            <strong>User Interface:</strong> The search bar and results page that allow users to enter queries and view the ranked list of relevant web pages.
        </li>
    </ul>
    <p><strong>Popular examples of search engines include:</strong></p>
    <ul>
        <li>Google</li>
        <li>Bing</li>
        <li>Yahoo!</li>
        <li>DuckDuckGo (known for its privacy focus)</li>
        <li>Baidu (dominant in China)</li>
        <li>Yandex (popular in Russia)</li>
    </ul>
    <p>Search engines are essential tools for navigating the vast amount of information available online, helping users find what they are looking for quickly and efficiently.</p>
    <p>Sources and related content</p>
</div>
"""


class LLMAgent(AgentRequest):
    def __init__(self, **kwargs):
        self.query = None
        self.history = None
        super().__init__(**kwargs)

    def process(self, peer):
        peer.send_message(static_response)
