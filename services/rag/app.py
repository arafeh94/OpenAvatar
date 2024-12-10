import numpy as np
from fastapi import FastAPI, HTTPException
from typing import List

from src.plugins.faiss_indexer import FaissIndexer

app = FastAPI()

indexer = FaissIndexer(np.array([['arafeh'], ['kareem']]), np.array([0, 1]))


@app.get("/search", response_model=List[str])
def search(search_query: str):
    return indexer.search(search_query, 10)


@app.get("/item", response_model=List[str])
def get_item(item_id: int):
    return ['1', '2']
