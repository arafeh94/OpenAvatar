import logging
from tabulate import tabulate

from app.rag.analyst import AnalystService, AnalystQuery

logging.basicConfig(level=logging.INFO)
analyst_service = AnalystService.create(model_name='gpt-4o', max_tokens=4096)
analyst = analyst_service.new_analyst()
features = [
    [0, 'What is federated learning'],
    [0, 'what is non-iid'],
    [0, 'how researchers addressed non_iid'],
    [1, 'hi'],
    [1, 'ho'],
    [1, 'hiii']
]
columns = ('Student ID', 'Query')

query = AnalystQuery(
    features, columns=columns,
    axioms={
        'T1': 'given student intents (Query), we want to calculate the student improvement rate based on how deep the student question regarding the topic',
        'T2': 'given student intents (Query), we want to know what is the problem the student is facing',
    },
    propositions={
        'T1': 'calculate the improvement rate (/1) as measurement called score',
        'T2': 'give explanation regarding the score as field called explanation',
    },
    facts={
        'Course Topic': 'The course is talking about federated learning',
    },
    constraints=['Return only a valid JSON array (no additional text)',
                 'The explanation should be concise (max 1 sentence per student).'
                 'No code backticks',
                 'No newlines, format string to return pure json',
                 'json object template student_id, score, explanation'],
    rules=['If given any ids, return all of them']
)

state = analyst.analyse(query)
print(state['response'])
