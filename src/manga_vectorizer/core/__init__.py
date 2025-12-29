"""
Core Module

Contains encoders and search utilities:
- clip_encoder: CLIP model loading and image/text encoding
- llm_encoder: LLM-based character description generation
- faiss_search: FAISS index loading and similarity search
- fusion: Late fusion reranking strategies
"""

from .clip_encoder import (
    load_clip_model,
    encode_image,
    encode_text,
)

from .llm_encoder import encode_query_llm

from .faiss_search import (
    load_faiss_index,
    search_index,
    build_text_lookup,
    build_image_lookup,
    compute_text_similarities,
    compute_image_similarities,
    deduplicate_text_candidates,
    get_page_key,
)

from .fusion import (
    late_fusion_rerank,
    late_fusion_rerank_two,
    print_fusion_results,
)

__all__ = [
    # CLIP encoder
    "load_clip_model",
    "encode_image",
    "encode_text",
    # LLM encoder
    "encode_query_llm",
    # FAISS search
    "load_faiss_index",
    "search_index",
    "build_text_lookup",
    "build_image_lookup",
    "compute_text_similarities",
    "compute_image_similarities",
    "deduplicate_text_candidates",
    "get_page_key",
    # Fusion
    "late_fusion_rerank",
    "late_fusion_rerank_two",
    "print_fusion_results",
]
