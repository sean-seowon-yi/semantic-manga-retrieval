"""
Retrieval Module

Contains retrieval algorithms:
- qcfr: Query-Conditioned Feedback Re-ranking with Rocchio-style refinement
"""

from .qcfr import (
    qcfr_search,
    qcfr_search_with_description,
    aggregate_text_scores_by_page,
    compute_hybrid_scores,
    select_pseudo_labels,
    compute_feedback_centroids,
    refine_query_rocchio,
    softmax,
    build_page_to_text_indices,
    build_page_to_faiss_mapping,
    compute_true_max_text_score,
)

__all__ = [
    # Main search functions
    "qcfr_search",
    "qcfr_search_with_description",
    # Helper functions
    "aggregate_text_scores_by_page",
    "compute_hybrid_scores",
    "select_pseudo_labels",
    "compute_feedback_centroids",
    "refine_query_rocchio",
    "softmax",
    "build_page_to_text_indices",
    "build_page_to_faiss_mapping",
    "compute_true_max_text_score",
]
