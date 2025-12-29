"""
Manga Vectorizer

A package for manga character retrieval using CLIP embeddings, LLM descriptions,
and advanced retrieval methods including late fusion and QCFR.

Modules:
    core: Core encoders (CLIP, LLM) and FAISS search utilities
    retrieval: Retrieval algorithms (QCFR)
    evaluation: Recall evaluation and clustering metrics
"""

__version__ = "1.0.0"

from manga_vectorizer.core import (
    load_clip_model,
    encode_image,
    encode_text,
    encode_query_llm,
    load_faiss_index,
    search_index,
    get_page_key,
)

from manga_vectorizer.retrieval import (
    qcfr_search,
    qcfr_search_with_description,
)

__all__ = [
    # Core - CLIP
    "load_clip_model",
    "encode_image",
    "encode_text",
    # Core - LLM
    "encode_query_llm",
    # Core - FAISS
    "load_faiss_index",
    "search_index",
    "get_page_key",
    # Retrieval
    "qcfr_search",
    "qcfr_search_with_description",
]
