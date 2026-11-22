"""Retrieval: find the chunks most relevant to a query.

Only TF-IDF is implemented so far. Dense and hybrid retrieval need
sentence-transformers, which is a large download on a slow connection.
"""

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(REPO_ROOT, "chunks.jsonl")

# Built on the first call and reused, so we read chunks.jsonl and fit the
# vectorizer once per session instead of on every query. If chunks.jsonl is
# rebuilt, restart the Python session to pick up the new chunks.
_chunks = None
_vectorizer = None
_matrix = None


def load_tfidf_index():
    global _chunks, _vectorizer, _matrix
    if _chunks is not None:
        return

    _chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            _chunks.append(json.loads(line))

    # The default token pattern is Unicode-aware, so Arabic and French words
    # are tokenized too. TF-IDF only matches the same words, though, so an
    # Arabic question will not find an English chunk. That is what the dense
    # multilingual model is for.
    _vectorizer = TfidfVectorizer()
    _matrix = _vectorizer.fit_transform([chunk["text"] for chunk in _chunks])


def retrieve(query: str, k: int = 5, method: str = "hybrid") -> list[dict]:
    """method is "tfidf", "dense" or "hybrid".
    Returns k chunk records, each with an added "score" float, best first."""
    if method == "dense":
        # Will embed the query and every chunk with multilingual-e5-small and
        # rank by cosine similarity in a FAISS index. Because the model maps
        # all three languages into one space, this handles cross-lingual
        # questions (Arabic question, English chunk) that TF-IDF cannot.
        raise NotImplementedError("dense retrieval needs sentence-transformers")
    if method == "hybrid":
        # Will run both TF-IDF and dense retrieval and combine their scores,
        # so exact matches on names and numbers (TF-IDF's strength) and
        # meaning across languages (dense's strength) both count.
        raise NotImplementedError("hybrid retrieval needs sentence-transformers")
    if method != "tfidf":
        raise ValueError(f"unknown method: {method}")

    load_tfidf_index()
    query_vector = _vectorizer.transform([query])

    # Cosine similarity rather than the raw dot product: the dot product grows
    # with the length of the vectors, so a long chunk that repeats common words
    # would outscore a short chunk that matches the query better. Cosine divides
    # by both lengths and only measures how similar the word mix is. Our chunks
    # range from 47 to 149 words, so this matters. (TfidfVectorizer already
    # L2-normalizes rows by default, which makes the two equal here, but using
    # cosine_similarity keeps the ranking correct if that setting changes.)
    scores = cosine_similarity(query_vector, _matrix)[0]

    # Highest score first. Asking for more than we have returns everything.
    ranked = sorted(range(len(_chunks)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked[:k]:
        # Copy so adding "score" does not change the cached chunk records.
        record = dict(_chunks[i])
        record["score"] = float(scores[i])
        results.append(record)
    return results
