import logging
import os
from typing import List, Any, Dict, TypedDict, Tuple
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from tabulate import tabulate

from core.tools.token_generator import generate_token


class AnalystQuery:
    def __init__(self, features: List[List[Any]], columns: Tuple = None, axioms: Dict[str, str] = None,
                 propositions: Dict[str, str] = None, facts: Dict[str, str] = None, rules: List[str] = None,
                 constraints: List[str] = None):
        self.features = features
        self.columns = columns
        self.axioms = axioms
        self.propositions = propositions
        self.facts = facts
        self.rules = rules
        self.constraints = constraints


class Analyst:
    def __init__(self, id, graph: CompiledStateGraph):
        self.id = id
        self.graph = graph

    def analyse(self, query: AnalystQuery):
        config = {"configurable": {"thread_id": self.id}}
        invoke_query = {
            'features': query.features, 'columns': query.columns, 'rules': query.rules,
            'constraints': query.constraints, 'axioms': query.axioms, 'propositions': query.propositions,
            'facts': query.facts,
        }
        return self.graph.invoke(invoke_query, config)


class State(TypedDict):
    features: List[List[Any]]
    columns: List[Any]
    rules: List[Any]
    axioms: Dict[str, str]
    propositions: Dict[str, str]
    constraints: List[str]
    facts: Dict[str, str]
    response: Dict[str, Any]
    context: str


class AnalystService:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger("ChatService")

    def _prompt(self, state: State):
        ai_prompt = '''
Using the given data features and axioms, process the given propositions sequentially, following these rules:
------Rules Starts------
- analyse the features given the axioms. 
- do not give descriptions. only execute propositions.
- do not give any results besides what required in the propositions.
- do not repeat whats being said in the context.
{}
------Rules Ends------

------Dataset Starts------
{}
------Dataset Ends------
------Axioms Starts------
{}
------Axioms Ends------
------Propositions Start------
{}
------Propositions Ends------
------Supporting Facts Starts------
{}
------Supporting Facts Ends------
------Constraints Starts------
{}
------Constraints Ends------
        '''

        features = tabulate(state['features'], headers=state['columns'], tablefmt='grid')
        rules = '\n'.join('- {}'.format(r) for r in state["rules"])
        axioms = '\n'.join(['{} - Axiom: {}'.format(a, b) for a, b in state['axioms'].items()])
        propositions = '\n'.join(['{} - Proposition: {}'.format(a, b) for a, b in state['propositions'].items()])
        facts = '\n'.join(['{}, Fact: {}'.format(a, b) for a, b in state['facts'].items()])
        constraints = '\n'.join('- {}'.format(r) for r in state["constraints"])
        return {'context': ai_prompt.format(rules, features, axioms, propositions, facts, constraints)}

    def _generate(self, state: State):
        analyst_query = state["context"]
        self.logger.info("AI Prompt - Query: {}".format(analyst_query))
        response = self.llm.invoke(analyst_query)
        self.logger.info("AI Prompt - Generated Response: {}".format(response))
        return {"response": [response]}

    def new_analyst(self):
        sequence = [self._prompt, self._generate]
        token = generate_token()
        graph_builder = StateGraph(State).add_sequence(sequence)
        graph_builder.add_edge(START, sequence[0].__name__)
        graph = graph_builder.compile(checkpointer=MemorySaver())
        return Analyst(token, graph)

    @staticmethod
    def create(model_name="gpt-3.5-turbo", max_tokens=8000):
        return AnalystService.create_openai(model_name=model_name, max_tokens=max_tokens)

    @staticmethod
    def create_openai(model_name='gpt-3.5-turbo', temperature=0.2, max_tokens=2000):
        if model_name == 'deepseek-chat':
            llm = ChatOpenAI(model="deepseek-chat", temperature=0.7, openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                             openai_api_base="https://api.deepseek.com", max_tokens=max_tokens)
        else:
            llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY'),
                             max_tokens=max_tokens)

        return AnalystService(llm)
