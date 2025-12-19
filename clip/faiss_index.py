"""
FAISS Index Builder for Manga Image Search

Creates a FAISS index from CLIP embeddings for fast similarity search.
Uses L2-normalized embeddings with Inner Product for cosine similarity.

Usage:
    python clip/faiss_index.py --image-dir datasets/small
    python clip/faiss_index.py --image-dir datasets/medium --use-ivf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Lazy imports - loaded on first use
_faiss = None
_clip_funcs = None


def _get_faiss():
    """Lazy load FAISS to avoid import conflicts."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_clip_funcs():
    """Lazy load CLIP functions."""
    global _clip_funcs
    if _clip_funcs is None:
        from clip import load_clip_model, get_all_images, extract_embeddings, parse_image_metadata
        _clip_funcs = (load_clip_model, get_all_images, extract_embeddings, parse_image_metadata)
    return _clip_funcs


class MangaFaissIndex:
    """
    FAISS index for manga panel similarity search.
    Uses L2-normalized embeddings with Inner Product (= cosine similarity).
    """
    
    __slots__ = ('dimension', 'index', 'id_to_meta', '_next_id')
    
    def __init__(self, dimension: int = 768, use_ivf: bool = False, nlist: int = 100):
        """
        Initialize the index.
        
        Args:
            dimension: Embedding dimension (768 for CLIP ViT-L-14)
            use_ivf: Use IVF index for faster search on large datasets
            nlist: Number of clusters for IVF index
        """
        faiss = _get_faiss()
        self.dimension = dimension
        
        if use_ivf:
            # IVF index for larger datasets - faster but requires training
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            # Flat index - exact search, good for <100k vectors
            self.index = faiss.IndexFlatIP(dimension)
        
        # Combined metadata storage: int_id -> {string_id, author, manga, ...}
        self.id_to_meta: dict[int, dict] = {}
        self._next_id = 0
    
    def add(self, embeddings: NDArray[np.float32], image_paths: list[Path]) -> None:
        """
        Add embeddings to the index with metadata.
        
        Args:
            embeddings: L2-normalized embedding vectors (N x D)
            image_paths: Corresponding image paths
        """
        if len(embeddings) == 0:
            return
        
        faiss = _get_faiss()
        _, _, _, parse_meta = _get_clip_funcs()
        
        # Ensure float32 and normalize
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        # Train IVF index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print(f"Training IVF index with {len(embeddings)} vectors...")
            self.index.train(embeddings)
        
        # Build metadata for all images at once
        start_id = self._next_id
        for i, path in enumerate(image_paths):
            meta = parse_meta(path)
            int_id = start_id + i
            self.id_to_meta[int_id] = {
                "string_id": meta["string_id"],
                "author": meta["author"],
                "manga": meta["manga"],
                "chapter": meta["chapter"],
                "chapter_num": meta["chapter_num"],
                "page": meta["page"],
                "page_num": meta["page_num"],
                "path": meta["path"],
            }
        
        self._next_id = start_id + len(image_paths)
        
        # Add to index
        self.index.add(embeddings)
        print(f"Indexed {len(embeddings)} images (total: {self.index.ntotal})")
    
    def search(self, query: NDArray[np.float32], k: int = 10) -> list[dict]:
        """
        Search for similar images.
        
        Args:
            query: Query embedding (D,) or (1, D)
            k: Number of results
        
        Returns:
            List of results with similarity scores and metadata
        """
        faiss = _get_faiss()
        
        # Prepare query
        query = np.ascontiguousarray(
            query.reshape(1, -1) if query.ndim == 1 else query,
            dtype=np.float32
        )
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Build results
        return [
            {"rank": i + 1, "similarity": float(score), **self.id_to_meta[idx]}
            for i, (score, idx) in enumerate(zip(scores[0], indices[0]))
            if idx != -1 and idx in self.id_to_meta
        ]
    
    def search_batch(self, queries: NDArray[np.float32], k: int = 10) -> list[list[dict]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Query embeddings (N, D)
            k: Number of results per query
        
        Returns:
            List of result lists
        """
        faiss = _get_faiss()
        
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        faiss.normalize_L2(queries)
        
        scores, indices = self.index.search(queries, k)
        
        results = []
        for q_scores, q_indices in zip(scores, indices):
            q_results = [
                {"rank": i + 1, "similarity": float(score), **self.id_to_meta[idx]}
                for i, (score, idx) in enumerate(zip(q_scores, q_indices))
                if idx != -1 and idx in self.id_to_meta
            ]
            results.append(q_results)
        
        return results
    
    def save(self, output_dir: Path) -> None:
        """Save index and metadata to disk."""
        faiss = _get_faiss()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        
        # Save metadata (single file with all info)
        data = {
            "dimension": self.dimension,
            "total": self.index.ntotal,
            "index_type": type(self.index).__name__,
            "metadata": {str(k): v for k, v in self.id_to_meta.items()},
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(data, f, separators=(',', ':'))  # Compact JSON
        
        print(f"Saved index ({self.index.ntotal} vectors) to {output_dir}")
    
    @classmethod
    def load(cls, index_dir: Path) -> MangaFaissIndex:
        """Load index from disk."""
        faiss = _get_faiss()
        index_dir = Path(index_dir)
        
        # Load FAISS index
        index = faiss.read_index(str(index_dir / "faiss.index"))
        
        # Load metadata
        with open(index_dir / "metadata.json") as f:
            data = json.load(f)
        
        # Create instance
        instance = cls.__new__(cls)
        instance.index = index
        instance.dimension = data.get("dimension", index.d)  # Fallback to index dimension
        
        # Handle both old and new metadata formats
        if "metadata" in data:
            # New format: {"dimension": ..., "metadata": {id: meta}}
            instance.id_to_meta = {int(k): v for k, v in data["metadata"].items()}
        else:
            # Old format: {string_id: meta} - need to rebuild mappings
            instance.id_to_meta = {}
            for i, (string_id, meta) in enumerate(data.items()):
                if isinstance(meta, dict) and "author" in meta:
                    instance.id_to_meta[i] = {"string_id": string_id, **meta}
        
        instance._next_id = max(instance.id_to_meta.keys(), default=-1) + 1
        
        print(f"Loaded index with {index.ntotal} vectors")
        return instance
    
    def get_by_id(self, int_id: int) -> dict | None:
        """Get metadata by integer ID."""
        return self.id_to_meta.get(int_id)
    
    def get_embedding(self, int_id: int) -> NDArray[np.float32] | None:
        """Reconstruct embedding for an ID."""
        if int_id not in self.id_to_meta:
            return None
        return self.index.reconstruct(int_id)
    
    def __len__(self) -> int:
        return self.index.ntotal


def build_index(
    image_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    use_ivf: bool = False,
) -> MangaFaissIndex:
    """Build FAISS index from images."""
    load_model, get_images, extract_emb, _ = _get_clip_funcs()
    
    print(f"\n{'='*50}")
    print("  FAISS Index Builder")
    print(f"{'='*50}")
    
    # Load model
    model, preprocess, device = load_model()
    
    # Find images
    print(f"\nScanning: {image_dir}")
    paths = get_images(image_dir)
    print(f"Found {len(paths)} images")
    
    if not paths:
        raise ValueError("No images found")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings, valid_paths = extract_emb(
        model, preprocess, device, paths,
        batch_size=batch_size, normalize=True
    )
    
    # Build index
    print("\nBuilding index...")
    index = MangaFaissIndex(
        dimension=embeddings.shape[1],
        use_ivf=use_ivf,
        nlist=min(100, len(valid_paths) // 10) if use_ivf else 100
    )
    index.add(embeddings, valid_paths)
    
    # Save
    index.save(output_dir)
    
    print(f"\n{'='*50}")
    print(f"  Done! {len(index)} images indexed")
    print(f"{'='*50}\n")
    
    return index


def demo_search(index_dir: Path, query_image: Path | None = None):
    """Demo search functionality."""
    print("\n=== Search Demo ===")
    
    index = MangaFaissIndex.load(index_dir)
    
    if query_image:
        load_model, _, _, _ = _get_clip_funcs()
        model, preprocess, device = load_model()
        
        from PIL import Image
        import torch
        
        img = preprocess(Image.open(query_image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()
        
        results = index.search(emb, k=5)
        print(f"\nQuery: {query_image}")
    else:
        # Use first indexed image
        first_id = min(index.id_to_meta.keys())
        meta = index.id_to_meta[first_id]
        print(f"\nQuery: {meta['author']}/{meta['manga']}/{meta['page']}")
        
        emb = index.get_embedding(first_id)
        results = index.search(emb, k=6)[1:]  # Skip self
    
    print("\nResults:")
    for r in results:
        print(f"  {r['rank']}. [{r['similarity']:.3f}] {r['author']}/{r['manga']}/{r['chapter']}/{r['page']}")


def main():
    parser = argparse.ArgumentParser(description="FAISS index for manga search")
    parser.add_argument("--image-dir", default="datasets/small", help="Image directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--use-ivf", action="store_true", help="Use IVF index (faster for large datasets)")
    parser.add_argument("--demo", action="store_true", help="Run demo search")
    parser.add_argument("--query", default=None, help="Query image path for demo")
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir / "faiss_index"
    
    index = build_index(image_dir, output_dir, args.batch_size, args.use_ivf)
    
    if args.demo:
        demo_search(output_dir, Path(args.query) if args.query else None)


if __name__ == "__main__":
    main()
