import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# noinspection PyArgumentList
class FaissIndexer:
    def __init__(self, embeddings, search_ids=None):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.indexes = self._init_indexes(embeddings)
        self.embeddings = embeddings
        self.search_ids = search_ids

    # noinspection PyArgumentList
    def _init_indexes(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 distance
        index.add(embeddings)
        return index

    def encode(self, msg):
        return self.encoder.encode(msg, convert_to_numpy=True).astype(np.float32).reshape(1, -1)

    def search(self, msg, top_k):
        msg = self.encode(msg)
        distances, indices = self.indexes.search(msg, k=top_k)
        result_ids = [str(self.search_ids[idx]) for idx in indices[0]]
        result_embeddings = [self.embeddings[idx] for idx in indices[0]]
        search_result = (indices, result_embeddings) if self.search_ids else (indices, result_embeddings, result_ids)
        return search_result
