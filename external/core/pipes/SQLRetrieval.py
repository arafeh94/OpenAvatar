from abc import abstractmethod, ABC
from external.core.pipeline import Pipe
import numpy as np


@abstractmethod
class InnerSQL(ABC):
    @abstractmethod
    def query(self, q) -> any:
        pass

    @abstractmethod
    def get_conn(self) -> any:
        pass


# noinspection PyUnresolvedReferences
class SQLRetrieval(Pipe):
    def __init__(self, context: 'AppContext', top_k):
        super().__init__()
        self.context = context
        self.top_k = top_k

    def exec(self, arg, flow) -> any:
        query_embedding = self.context.encoder.encode(arg, convert_to_numpy=True).astype(np.float32)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.context.indexes.search(query_embedding, self.top_k)
        result_ids = [str(self.context.search_ids[idx]) for idx in indices[0]]

        q = f'select * from flights_fts where id in ({",".join(result_ids)})'
        self.context.logger.error('Query: {}'.format(q))
        results = self.context.sql.query(q)
        self.context.logger.error(f'{results}')
        return results
