#!/usr/bin/env python3
"""
Recall Evaluation Script for QCFR Queries

Evaluates recall@k using QCFR (Query-Conditioned Feedback Re-ranking):
Two-pass retrieval with hybrid scoring and Rocchio-style query refinement.

Supports grid search over hyperparameters for finding optimal configurations.

Usage (QCFR - single run):
    python -m manga_vectorizer.evaluation.recall_qcfr --queries ../queries \
        --alpha 0.5 --m-img 800 --l-pos 30 --b 0.35 --c 0.14 \
        --image-index ../final_dataset_embeddings/faiss_index --text-index ../final_dataset_text_embeddings/faiss_index \
        --output results/qcfr

Usage (QCFR - grid search):
    python -m manga_vectorizer.evaluation.recall_qcfr --queries ../queries \
        --alphas 0.3 0.5 0.8 \
        --m-imgs 100 200 300 \
        --l-pos-values 20 30 50 \
        --bs 0.2 0.35 0.5 \
        --cs 0.1 0.2 0.3 \
        --image-index ../final_dataset_embeddings/faiss_index \
        --text-index ../final_dataset_text_embeddings/faiss_index \
        --output results/qcfr_grid_test

Grid Search Parameters (5 tunable):
    --alpha/--alphas      Hybrid score weight (default: 0.5)
    --m-img/--m-imgs      Image candidates in Pass-1 (default: 800, m_txt = 3*m_img)
    --l-pos/--l-pos-values  Pseudo-positive/negative count (default: 30, l_neg = l_pos)
    --b/--bs              Rocchio positive feedback weight (default: 0.35)
    --c/--cs              Rocchio negative feedback weight (default: 0.14)

Fixed Parameters:
    --a           Rocchio original query weight (default: 1.0)
    --d           Rocchio text query weight (default: 0.21)
    --no-true-max-pooling  Disable true max over all text embeddings per page (faster)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torch
import open_clip
from itertools import product

# Import utility functions
from manga_vectorizer.evaluation.utils import (
    get_device,
    check_faiss_available,
    load_faiss_index_direct,
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

# Import QCFR module
from manga_vectorizer.retrieval.qcfr import qcfr_search, qcfr_search_with_description
from manga_vectorizer.core.clip_encoder import load_clip_model, encode_image, encode_text
from manga_vectorizer.core.faiss_search import (
    search_index,
    get_page_key,
)
from manga_vectorizer.core.llm_encoder import encode_query_llm


def qcfr_search_with_cached_description(
    image_path: Path,
    user_query: str,
    image_index_dir: Path,
    text_index_dir: Path,
    m_img: int = 800,
    m_txt: int = 2000,
    l_pos: int = 30,
    l_neg: int = 30,
    alpha: float = 0.5,
    a: float = 1.0,
    b: float = 0.35,
    c: float = 0.14,
    d: float = 0.21,
    k: int = 50,
    true_max_pooling: bool = True,
    cached_description: Optional[str] = None,
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    text_index=None,
    text_id_to_meta=None,
    image_index=None,
    image_id_to_meta=None,
    verbose: bool = False
) -> dict:
    """
    Perform QCFR search with two-pass retrieval and Rocchio-style query refinement.
    """
    # Step 1: Generate or use cached LLM description
    if cached_description is not None:
        description = cached_description
        if verbose:
            print(f"Using cached LLM description: {description[:80]}...")
    else:
        if verbose:
            print("\n[1/8] Generating LLM description...")
        description = encode_query_llm(image_path, user_query, verbose)
        if verbose:
            print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (only if not provided)
    if model is None or preprocess is None or tokenizer is None:
        if verbose:
            print("\n[2/8] Loading CLIP model and generating embeddings...")
        model, preprocess, tokenizer, device = load_clip_model(device)
    elif verbose:
        print("\n[2/8] Using provided CLIP model for generating embeddings...")
    
    query_image_embedding = encode_image(model, preprocess, device, image_path)
    query_text_embedding = encode_text(model, tokenizer, device, description)
    if verbose:
        print(f"  Query image embedding: {query_image_embedding.shape}")
        print(f"  Query text embedding: {query_text_embedding.shape}")
    
    # Step 3: Load FAISS indexes (only if not provided)
    if text_index is None or text_id_to_meta is None:
        if verbose:
            print("\n[3/8] Loading FAISS indexes...")
        text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
        if verbose:
            print(f"  Text index: {text_index.ntotal} vectors")
    elif verbose:
        print("\n[3/8] Using provided FAISS indexes...")
    
    if image_index is None or image_id_to_meta is None:
        image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
        if verbose:
            print(f"  Image index: {image_index.ntotal} vectors")
    
    # Steps 4-8: Run QCFR algorithm
    if verbose:
        print(f"\n[4-8] Running QCFR algorithm...")
        print(f"  m_img={m_img}, m_txt={m_txt}, l_pos={l_pos}, l_neg={l_neg}")
        print(f"  alpha={alpha}, a={a}, b={b}, c={c}, d={d}")
    
    qcfr_result = qcfr_search(
        query_image_embedding=query_image_embedding,
        query_text_embedding=query_text_embedding,
        image_index=image_index,
        image_id_to_meta=image_id_to_meta,
        text_index=text_index,
        text_id_to_meta=text_id_to_meta,
        m_img=m_img,
        m_txt=m_txt,
        l_pos=l_pos,
        l_neg=l_neg,
        alpha=alpha,
        a=a,
        b=b,
        c=c,
        d=d,
        k=k,
        true_max_pooling=true_max_pooling,
        verbose=verbose
    )
    
    final_results = qcfr_result['final_results']
    
    # Mark pseudo-positive and pseudo-negative status in results
    positive_pages = set(qcfr_result['positive_pages'])
    negative_pages = set(qcfr_result['negative_pages'])
    
    for result in final_results:
        page_key = get_page_key(result)
        result["page_key"] = page_key
        result["is_pseudo_positive"] = page_key in positive_pages
        result["is_pseudo_negative"] = page_key in negative_pages
    
    return {
        "query_image": str(image_path),
        "user_query": user_query,
        "llm_description": description,
        "params": {
            "m_img": m_img,
            "m_txt": m_txt,
            "l_pos": l_pos,
            "l_neg": l_neg,
            "alpha": alpha,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "true_max_pooling": true_max_pooling,
        },
        "k": k,
        "query_image_embedding": query_image_embedding,
        "query_text_embedding": query_text_embedding,
        "num_image_candidates": qcfr_result['num_image_candidates'],
        "num_text_candidates_raw": qcfr_result['num_text_candidates_raw'],
        "num_text_pages": qcfr_result['num_text_pages'],
        "num_union_candidates": qcfr_result['num_union_candidates'],
        "num_pseudo_positives": len(qcfr_result['positive_pages']),
        "num_pseudo_negatives": len(qcfr_result['negative_pages']),
        "positive_pages": qcfr_result['positive_pages'],
        "negative_pages": qcfr_result['negative_pages'],
        "final_results": final_results,
    }


def evaluate_single_query_qcfr(
    query_folder: Path,
    image_index_dir: Path,
    text_index_dir: Path,
    m_img: int,
    m_txt: int,
    l_pos: int,
    l_neg: int,
    alpha: float,
    a: float,
    b: float,
    c: float,
    d: float,
    k_values: List[int],
    map_k_values: List[int] = [10, 20, 30, 50],
    true_max_pooling: bool = True,
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    text_index=None,
    text_id_to_meta=None,
    image_index=None,
    image_id_to_meta=None,
    verbose: bool = False
) -> Dict:
    """Evaluate recall for a single query using QCFR search."""
    # Find query image
    query_image = find_query_image(query_folder)
    
    if query_image is None:
        return {
            "query_folder": str(query_folder),
            "error": "Query image not found (expected query.png, query.jpg, etc.)"
        }
    
    # Load text query
    user_query = load_text_query(query_folder)
    if not user_query:
        user_query = ""
    
    # Load cached LLM description or generate and save it
    cached_description = load_llm_description(query_folder)
    if cached_description is None:
        try:
            cached_description = encode_query_llm(query_image, user_query, verbose=False)
            save_llm_description(query_folder, cached_description)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate LLM description: {e}")
            cached_description = user_query if user_query else "manga image"
    
    # Load ground truth
    labels_file = query_folder / "labels.txt"
    ground_truth = load_ground_truth_labels(labels_file)
    
    if len(ground_truth) == 0:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": "No ground truth labels found",
            "total_ground_truth": 0
        }
    
    # Run QCFR search
    try:
        max_k = max(k_values)
        
        result = qcfr_search_with_cached_description(
            image_path=query_image,
            user_query=user_query,
            image_index_dir=image_index_dir,
            text_index_dir=text_index_dir,
            m_img=m_img,
            m_txt=m_txt,
            l_pos=l_pos,
            l_neg=l_neg,
            alpha=alpha,
            a=a,
            b=b,
            c=c,
            d=d,
            k=max_k,
            true_max_pooling=true_max_pooling,
            cached_description=cached_description,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            text_index=text_index,
            text_id_to_meta=text_id_to_meta,
            image_index=image_index,
            image_id_to_meta=image_id_to_meta,
            verbose=verbose
        )
        
        search_results = result['final_results']
        
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": f"Error running QCFR search: {e}",
            "total_ground_truth": len(ground_truth)
        }
    
    if len(search_results) == 0:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": "No search results",
            "total_ground_truth": len(ground_truth)
        }
    
    # Compute recall@k for each k value
    recall_metrics = {}
    for k in k_values:
        metric_name = f"recall@{k}"
        recall_metrics[metric_name] = compute_recall_at_k(search_results, ground_truth, k, include_page_key=True, include_source_file=False)
    
    # Compute mAP@k for each k value
    map_metrics = {}
    for k in map_k_values:
        metric_name = f"map@{k}"
        map_metrics[metric_name] = compute_map_at_k(search_results, ground_truth, k, include_page_key=True, include_source_file=False)
    
    # Get all relevant retrieved results
    all_relevant = []
    for res in search_results:
        if match_result_to_ground_truth(res, ground_truth, include_page_key=True, include_source_file=False):
            all_relevant.append(res)
    
    # Compute QCFR statistics
    pos_count = sum(1 for r in search_results if r.get('is_pseudo_positive', False))
    neg_count = sum(1 for r in search_results if r.get('is_pseudo_negative', False))
    qcfr_stats = {
        "num_pseudo_positives_in_results": pos_count,
        "num_pseudo_negatives_in_results": neg_count,
        "pseudo_positive_percentage": pos_count / len(search_results) if len(search_results) > 0 else 0.0,
        "num_union_candidates": result.get('num_union_candidates', 0),
    }
    
    # Print QCFR summary (use tqdm.write to avoid progress bar interference)
    if not verbose:
        query_name = query_folder.name
        tqdm.write(f"  {query_name}: {pos_count}/{len(search_results)} pseudo-positives in results")
    
    return {
        "query_folder": str(query_folder),
        "query_image": str(query_image),
        "user_query": user_query,
        "llm_description": result.get('llm_description', ''),
        "params": result.get('params', {}),
        "total_ground_truth": len(ground_truth),
        "total_retrieved": len(search_results),
        "relevant_retrieved": len(all_relevant),
        "recall_metrics": recall_metrics,
        "map_metrics": map_metrics,
        "all_retrieved": search_results,
        "relevant_results": all_relevant,
        "qcfr_stats": qcfr_stats,
        "num_image_candidates": result.get('num_image_candidates', 0),
        "num_text_candidates_raw": result.get('num_text_candidates_raw', 0),
        "num_text_pages": result.get('num_text_pages', 0),
    }


def visualize_query_results(
    query_path: str,
    all_retrieved: List[Dict],
    output_path: Path,
    num_images: int = 10,
    highlight_relevant: bool = True,
    image_dir: Path = None
):
    """Visualize search results for a QCFR query."""
    query_path = Path(query_path)
    if not query_path.exists():
        print(f"Warning: Query image not found: {query_path}")
        return
    
    results = all_retrieved[:num_images]
    n_results = len(results)
    
    if n_results == 0:
        print(f"Warning: No results to visualize for {query_path.name}")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, (n_results + 2) // 2, figsize=(3 * ((n_results + 2) // 2), 6))
    axes = axes.flatten() if n_results > 1 else [axes]
    
    # Plot query image
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Plot results
    for i, result in enumerate(results):
        ax = axes[i + 1]
        
        # Find image path
        image_path = find_image_path_from_result(result, image_dir)
        
        if image_path and image_path.exists():
            try:
                img = Image.open(image_path)
                
                # Add colored border based on relevance
                if highlight_relevant and result.get('is_relevant', False):
                    img = add_border_to_image(img, 'green', 5)
                elif result.get('is_pseudo_positive', False):
                    img = add_border_to_image(img, 'blue', 3)
                elif result.get('is_pseudo_negative', False):
                    img = add_border_to_image(img, 'red', 3)
                
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{str(e)[:30]}", ha='center', va='center', fontsize=8)
                ax.set_facecolor('lightgray')
        else:
            page_key = result.get('page_key', '?')[:30]
            ax.text(0.5, 0.5, f"Not found\n{page_key}", ha='center', va='center', fontsize=8)
            ax.set_facecolor('lightgray')
        
        # Title with rank and score
        rank = result.get('rank', i + 1)
        score = result.get('similarity', 0.0)
        status = "✓" if result.get('is_relevant', False) else ""
        pp = "P+" if result.get('is_pseudo_positive', False) else ""
        title = f"#{rank} ({score:.3f}) {status} {pp}"
        
        # Color title based on status
        if result.get('is_relevant', False):
            color = 'green'
        elif result.get('is_pseudo_positive', False):
            color = 'blue'
        elif result.get('is_pseudo_negative', False):
            color = 'orange'
        else:
            color = 'gray'
        
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    query_name = query_path.parent.name
    title_text = f"Query: {query_name} (QCFR)"
    plt.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recall@k for QCFR queries with grid search support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # QCFR - single run
  python -m manga_vectorizer.evaluation.recall_qcfr --queries queries \\
      --alpha 0.5 --m-img 800 --l-pos 30 --b 0.35 --c 0.14 \\
      --image-index indexes/image --text-index indexes/text --output results/qcfr
  
  # QCFR - grid search (5 parameters: alpha, m_img, l_pos, b, c)
  python -m manga_vectorizer.evaluation.recall_qcfr --queries queries \\
      --alphas 0.3 0.5 0.7 \\
      --m-imgs 300 500 800 \\
      --l-pos-values 20 30 50 \\
      --bs 0.2 0.35 0.5 \\
      --cs 0.1 0.14 0.2 \\
      --image-index indexes/image --text-index indexes/text --output results/qcfr_grid
        """
    )
    parser.add_argument("--queries", "-q", type=str, required=True, help="Path to queries directory")
    
    # QCFR parameters
    parser.add_argument("--m-img", type=int, default=None, help="Single m_img value (default: 800)")
    parser.add_argument("--m-imgs", type=int, nargs='+', default=None, help="Multiple m_img values for grid search")
    parser.add_argument("--l-pos", type=int, default=None, help="Single l_pos value (default: 30)")
    parser.add_argument("--l-pos-values", type=int, nargs='+', default=None, help="Multiple l_pos values for grid search")
    parser.add_argument("--alpha", type=float, default=None, help="Single alpha value (default: 0.5)")
    parser.add_argument("--alphas", type=float, nargs='+', default=None, help="Multiple alpha values for grid search")
    parser.add_argument("--a", type=float, default=1.0, help="Rocchio parameter for original query weight")
    parser.add_argument("--b", type=float, default=None, help="Single b value (default: 0.35)")
    parser.add_argument("--bs", type=float, nargs='+', default=None, help="Multiple b values for grid search")
    parser.add_argument("--c", type=float, default=None, help="Single c value (default: 0.14)")
    parser.add_argument("--cs", type=float, nargs='+', default=None, help="Multiple c values for grid search")
    parser.add_argument("--d", type=float, default=0.21, help="Rocchio parameter for text query weight")
    parser.add_argument("--no-true-max-pooling", action="store_true", help="Disable true max-pooling")
    
    # Index paths
    parser.add_argument("--image-index", type=str, required=True, help="Path to image FAISS index directory")
    parser.add_argument("--text-index", type=str, required=True, help="Path to text FAISS index directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output directory")
    
    # Evaluation parameters
    parser.add_argument("--k-values", type=int, nargs='+', default=[10, 20, 30, 50], help="K values for recall@k")
    parser.add_argument("--map-k-values", type=int, nargs='+', default=[10, 20, 30, 50], help="K values for mAP@k")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image directory for visualization")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not FAISS_AVAILABLE:
        print("Error: faiss is not installed. Install with: pip install faiss-cpu")
        return
    
    # Determine parameter values
    alpha_values = args.alphas if args.alphas else ([args.alpha] if args.alpha else [0.5])
    m_img_values = args.m_imgs if args.m_imgs else ([args.m_img] if args.m_img else [800])
    l_pos_values = args.l_pos_values if args.l_pos_values else ([args.l_pos] if args.l_pos else [30])
    b_values = args.bs if args.bs else ([args.b] if args.b else [0.35])
    c_values = args.cs if args.cs else ([args.c] if args.c else [0.14])
    
    # Validate alphas
    for alpha in alpha_values:
        if not 0 <= alpha <= 1:
            print(f"Error: alpha {alpha} must be between 0 and 1")
            return
    
    is_grid_search = (len(alpha_values) > 1 or len(m_img_values) > 1 or 
                     len(l_pos_values) > 1 or len(b_values) > 1 or len(c_values) > 1)
    num_combinations = len(alpha_values) * len(m_img_values) * len(l_pos_values) * len(b_values) * len(c_values)
    
    # Setup paths
    queries_dir = Path(args.queries)
    image_index_dir = Path(args.image_index)
    text_index_dir = Path(args.text_index)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    # Check paths
    if not queries_dir.exists():
        print(f"Error: Queries directory not found: {queries_dir}")
        return
    if not image_index_dir.exists():
        print(f"Error: Image index not found: {image_index_dir}")
        return
    if not text_index_dir.exists():
        print(f"Error: Text index not found: {text_index_dir}")
        return
    
    # Find query folders
    query_folders = sorted([d for d in queries_dir.iterdir() if d.is_dir()])
    if len(query_folders) == 0:
        print(f"Error: No query folders found in {queries_dir}")
        return
    
    # Print header
    print(f"\nQCFR Recall Evaluation")
    print("="*60)
    if is_grid_search:
        print(f"Grid Search: {num_combinations} combinations")
    print(f"Queries: {len(query_folders)}")
    print("="*60)
    
    # Load CLIP model once
    print("\nLoading CLIP model...")
    device = args.device if args.device else get_device()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    print(f"CLIP model loaded on {device}")
    
    # Load FAISS indexes once
    print("\nLoading FAISS indexes...")
    image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
    text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
    print(f"Image index: {image_index.ntotal} vectors")
    print(f"Text index: {text_index.ntotal} vectors")
    
    # Grid search
    grid_results = {}
    param_names = ("alpha", "m_img", "l_pos", "b", "c")
    param_combinations = list(product(alpha_values, m_img_values, l_pos_values, b_values, c_values))
    
    for combo_idx, params in enumerate(param_combinations, 1):
        alpha, m_img, l_pos, b, c = params
        combo_name = f"a{alpha}_m{m_img}_l{l_pos}_b{b}_c{c}"
        
        if is_grid_search:
            print(f"\n{'='*60}")
            print(f"Combination {combo_idx}/{num_combinations}: α={alpha}, m={m_img}, l={l_pos}, b={b}, c={c}")
            print("="*60)
        
        all_results = []
        query_times = []
        
        desc = f"[{combo_idx}/{num_combinations}]" if is_grid_search else "Processing"
        for query_folder in tqdm(query_folders, desc=desc):
            start_time = time.time()
            
            result = evaluate_single_query_qcfr(
                query_folder=query_folder,
                image_index_dir=image_index_dir,
                text_index_dir=text_index_dir,
                m_img=m_img,
                m_txt=3 * m_img,
                l_pos=l_pos,
                l_neg=l_pos,
                alpha=alpha,
                a=args.a,
                b=b,
                c=c,
                d=args.d,
                k_values=args.k_values,
                map_k_values=args.map_k_values,
                true_max_pooling=not args.no_true_max_pooling,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                text_index=text_index,
                text_id_to_meta=text_id_to_meta,
                image_index=image_index,
                image_id_to_meta=image_id_to_meta,
                verbose=args.verbose
            )
            
            query_time = time.time() - start_time
            result['query_time_seconds'] = query_time
            query_times.append(query_time)
            all_results.append(result)
        
        # Compute averages
        averages = compute_average_metrics(all_results)
        
        # Store results
        grid_results[params] = {
            'all_results': all_results,
            'averages': averages,
            'query_times': query_times,
        }
        
        # Save results
        combo_output_dir = output_dir / combo_name if is_grid_search else output_dir
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "config": {"alpha": alpha, "m_img": m_img, "l_pos": l_pos, "b": b, "c": c},
            "individual_results": all_results,
            "average_metrics": averages,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(combo_output_dir / "recall_results.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        create_summary_table(all_results, averages, combo_output_dir, include_user_query=True)
        
        if is_grid_search and 'error' not in averages:
            map_50 = averages['average_metrics'].get('map@50', {}).get('mean', 0)
            print(f"  mAP@50: {map_50:.4f}")
        
        # Generate visualizations for single run (not grid search)
        if not is_grid_search and not args.no_viz:
            print("\nGenerating visualizations...")
            curves_dir = combo_output_dir / "recall_curves"
            curves_dir.mkdir(parents=True, exist_ok=True)
            
            # Individual recall curves
            for i, result in enumerate(all_results):
                if 'error' in result:
                    continue
                query_name = Path(result['query_folder']).name
                curve_file = curves_dir / f"query_{i+1:03d}_{query_name}_recall_curve.png"
                plot_recall_curve(result, curve_file, color='purple', title_prefix="QCFR")
            
            # Aggregate curve
            aggregate_file = curves_dir / "aggregate_all_queries.png"
            plot_aggregate_recall_curves(all_results, averages, aggregate_file, color='purple', title="QCFR Queries")
            print(f"Recall curves saved to: {curves_dir}")
    
    # Grid search summary
    if is_grid_search:
        print(f"\n{'='*60}")
        print("GRID SEARCH SUMMARY")
        print("="*60)
        
        # Find best
        best_params = None
        best_score = -1
        
        comparison_data = []
        for params_key, data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                row = {name: params_key[i] for i, name in enumerate(param_names)}
                for metric_name, stats in avg['average_metrics'].items():
                    row[f'{metric_name}_mean'] = stats['mean']
                comparison_data.append(row)
                
                map_50 = avg['average_metrics'].get('map@50', {}).get('mean', 0)
                if map_50 > best_score:
                    best_score = map_50
                    best_params = params_key
        
        # Save comparison
        if comparison_data:
            import csv
            fieldnames = list(param_names) + sorted([k for k in comparison_data[0].keys() if k not in param_names])
            with open(output_dir / "grid_search_comparison.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted(comparison_data, key=lambda x: tuple(x[n] for n in param_names)))
            print(f"Comparison saved to: {output_dir / 'grid_search_comparison.csv'}")
        
        if best_params:
            print(f"\nBest config: α={best_params[0]}, m={best_params[1]}, l={best_params[2]}, b={best_params[3]}, c={best_params[4]}")
            print(f"Best mAP@50: {best_score:.4f}")
            
            with open(output_dir / "best_config.json", 'w') as f:
                json.dump({f"best_{name}": best_params[i] for i, name in enumerate(param_names)}, f, indent=2)
            
            # Generate visualizations for best config
            if not args.no_viz:
                print("\nGenerating visualizations for best config...")
                best_data = grid_results[best_params]
                best_combo_name = f"a{best_params[0]}_m{best_params[1]}_l{best_params[2]}_b{best_params[3]}_c{best_params[4]}"
                best_output_dir = output_dir / best_combo_name
                curves_dir = best_output_dir / "recall_curves"
                curves_dir.mkdir(parents=True, exist_ok=True)
                
                # Individual recall curves
                for i, result in enumerate(best_data['all_results']):
                    if 'error' in result:
                        continue
                    query_name = Path(result['query_folder']).name
                    curve_file = curves_dir / f"query_{i+1:03d}_{query_name}_recall_curve.png"
                    plot_recall_curve(result, curve_file, color='purple', title_prefix="QCFR")
                
                # Aggregate curve
                aggregate_file = curves_dir / "aggregate_all_queries.png"
                plot_aggregate_recall_curves(best_data['all_results'], best_data['averages'], aggregate_file, color='purple', title="QCFR (Best Config)")
                print(f"Recall curves saved to: {curves_dir}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
