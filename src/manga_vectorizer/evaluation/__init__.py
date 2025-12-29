"""
Evaluation Module

Contains evaluation scripts and utilities:
- utils: Common utilities for recall evaluation
- recall_image: Image-based query evaluation
- recall_text: Text-based query evaluation
- recall_fusion: Late fusion evaluation
- recall_qcfr: QCFR evaluation with grid search
- cluster: Embedding clustering and visualization
"""

from .utils import (
    get_device,
    check_faiss_available,
    load_faiss_index,
    load_faiss_index_direct,
    search_faiss_index,
    load_ground_truth_labels,
    match_result_to_ground_truth,
    compute_recall_at_k,
    compute_map_at_k,
    compute_average_metrics,
    plot_recall_curve,
    plot_aggregate_recall_curves,
    add_border_to_image,
    create_summary_table,
    load_text_query,
    load_llm_description,
    save_llm_description,
    find_query_image,
    find_image_path_from_result,
    FAISS_AVAILABLE,
)

__all__ = [
    # Device and FAISS
    "get_device",
    "check_faiss_available",
    "load_faiss_index",
    "load_faiss_index_direct",
    "search_faiss_index",
    "FAISS_AVAILABLE",
    # Ground truth
    "load_ground_truth_labels",
    "match_result_to_ground_truth",
    # Metrics
    "compute_recall_at_k",
    "compute_map_at_k",
    "compute_average_metrics",
    # Visualization
    "plot_recall_curve",
    "plot_aggregate_recall_curves",
    "add_border_to_image",
    "create_summary_table",
    # Query loading
    "load_text_query",
    "load_llm_description",
    "save_llm_description",
    "find_query_image",
    "find_image_path_from_result",
]
