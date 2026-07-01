"""
retriever.py — Phase 1: Hybrid BM25 + Dense FAISS Retriever with RRF

Responsibilities:
- Build and hold in-memory BM25 and FAISS indexes from the clean catalog.
- Expose a single `search_shl_catalog()` function that:
    1. Runs BM25 keyword search (strong for technical terms like ".NET", "Java").
    2. Runs dense cosine-similarity FAISS search (strong for semantic queries).
    3. Merges results via Reciprocal Rank Fusion (RRF).
    4. Applies optional job_level and keys_filter post-filters.
    5. Enforces result diversification to prevent near-duplicate crowding.
- The indexes are built once at module import so per-request latency stays low.
"""

from __future__ import annotations

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import Any, Optional

from data_prep import load_catalog

# ---------------------------------------------------------------------------
# Module-level initialization (runs once at import / startup)
# ---------------------------------------------------------------------------

print("[retriever] Loading and indexing catalog — this runs once...")

# Load clean catalog
_CATALOG_LIST, _CATALOG_LOOKUP = load_catalog()
_N = len(_CATALOG_LIST)

# Tokenize doc_texts for BM25 (simple whitespace + lowercase tokenization)
_TOKENIZED_CORPUS: list[list[str]] = [
    item["doc_text"].lower().split() for item in _CATALOG_LIST
]
_BM25 = BM25Okapi(_TOKENIZED_CORPUS)

# Dense embeddings via a lightweight sentence-transformer model.
# 'all-MiniLM-L6-v2' (22M params, 80ms/batch) is a solid balance of speed & quality.
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_DOC_TEXTS: list[str] = [item["doc_text"] for item in _CATALOG_LIST]

print("[retriever] Encoding catalog documents for FAISS...")
_EMBEDDINGS: np.ndarray = _EMBED_MODEL.encode(
    _DOC_TEXTS,
    batch_size=64,
    show_progress_bar=False,
    normalize_embeddings=True,   # Required for cosine similarity via inner product
    convert_to_numpy=True,
)
_DIM: int = _EMBEDDINGS.shape[1]

# Build FAISS flat inner-product index (cosine sim on L2-normalized vectors)
_FAISS_INDEX = faiss.IndexFlatIP(_DIM)
_FAISS_INDEX.add(_EMBEDDINGS.astype(np.float32))

print(f"[retriever] Indexes ready. BM25 + FAISS ({_N} docs, dim={_DIM}).")


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    bm25_ranked: list[int],
    dense_ranked: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge two ranked lists of catalog indices via Reciprocal Rank Fusion.

    RRF score for a document d = Σ  1 / (k + rank(d, list_i))
    Higher RRF score → more relevant.

    Args:
        bm25_ranked : Ordered list of catalog indices by BM25 score.
        dense_ranked: Ordered list of catalog indices by FAISS score.
        k           : RRF smoothing constant (k=60 is the canonical default).

    Returns:
        List of (catalog_index, rrf_score) sorted descending by score.
    """
    scores: dict[int, float] = {}

    for rank, idx in enumerate(bm25_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Post-filter & Diversification helpers
# ---------------------------------------------------------------------------

def _apply_filters(
    indices: list[int],
    job_level: Optional[str],
    keys_filter: Optional[str],
) -> list[int]:
    """
    Apply soft post-filters based on user-specified constraints.

    'Soft' means: if a filter is specified but zero items pass, we return
    all indices unfiltered (graceful degradation over empty results).
    """
    if not job_level and not keys_filter:
        return indices

    filtered: list[int] = []
    for idx in indices:
        item = _CATALOG_LIST[idx]
        level_ok = (
            not job_level
            or any(job_level.lower() in lvl.lower() for lvl in item["job_levels"])
        )
        keys_ok = (
            not keys_filter
            or any(keys_filter.lower() in k.lower() for k in item["keys"])
        )
        if level_ok and keys_ok:
            filtered.append(idx)

    return filtered if filtered else indices  # graceful fallback


def _diversify(
    ranked_items: list[dict[str, Any]],
    top_k: int,
    max_same_primary_key: int = 3,
) -> list[dict[str, Any]]:
    """
    Prevent near-duplicate product families from crowding the top-K.

    Allow at most `max_same_primary_key` items sharing the same primary
    test category (e.g., no more than 3 OPQ32 variants in a single shortlist).
    ALL variants remain in the index — only the final ranking is diversified.
    """
    seen_keys: dict[str, int] = {}
    result: list[dict[str, Any]] = []

    for item in ranked_items:
        primary_key = item["keys"][0] if item.get("keys") else "Unknown"
        count = seen_keys.get(primary_key, 0)
        if count < max_same_primary_key:
            result.append(item)
            seen_keys[primary_key] = count + 1
        if len(result) >= top_k:
            break

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_shl_catalog(
    query: str,
    job_level: Optional[str] = None,
    keys_filter: Optional[str] = None,
    top_k: int = 10,
    bm25_candidates: int = 50,
    dense_candidates: int = 50,
) -> list[dict[str, Any]]:
    """
    Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion.

    Args:
        query           : User's natural language query or distilled requirement.
        job_level       : Optional seniority filter (e.g. "Manager", "Entry-Level").
        keys_filter     : Optional test category filter (e.g. "Personality & Behavior").
        top_k           : Maximum number of results to return (capped at 10 per spec).
        bm25_candidates : Number of candidates to pull from BM25 before fusion.
        dense_candidates: Number of candidates to pull from FAISS before fusion.

    Returns:
        List of up to `top_k` normalized catalog dicts, ranked by RRF score.
    """
    top_k = min(top_k, 10)  # Hard cap per assignment spec

    # --- BM25 search ---
    tokens = query.lower().split()
    bm25_scores: np.ndarray = _BM25.get_scores(tokens)
    bm25_top_indices: list[int] = np.argsort(bm25_scores)[::-1][:bm25_candidates].tolist()

    # --- Dense search ---
    query_vector = _EMBED_MODEL.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    _, dense_top_indices_arr = _FAISS_INDEX.search(query_vector, dense_candidates)
    dense_top_indices: list[int] = dense_top_indices_arr[0].tolist()

    # --- RRF Fusion ---
    fused: list[tuple[int, float]] = _rrf_fuse(bm25_top_indices, dense_top_indices)

    # --- Post-filter ---
    all_fused_indices = [idx for idx, _ in fused]
    filtered_indices = _apply_filters(all_fused_indices, job_level, keys_filter)

    # Preserve RRF order after filtering
    filtered_set = set(filtered_indices)
    ordered_indices = [idx for idx in all_fused_indices if idx in filtered_set]

    # --- Map to catalog items ---
    ranked_items = [_CATALOG_LIST[idx] for idx in ordered_indices]

    # --- Diversify to avoid crowding ---
    diversified = _diversify(ranked_items, top_k)

    return diversified


def get_catalog_lookup() -> dict[str, dict[str, Any]]:
    """Return the URL-keyed catalog lookup dict (used by validation middleware)."""
    return _CATALOG_LOOKUP


def get_catalog_list() -> list[dict[str, Any]]:
    """Return the full clean catalog list."""
    return _CATALOG_LIST


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Test 1: Java developer, mid-level ===")
    results = search_shl_catalog("Java developer programming skills", job_level="Mid-Professional")
    for r in results:
        print(f"  [{r['test_type']}] {r['name']} — {r['link']}")

    print("\n=== Test 2: Personality assessment for sales ===")
    results = search_shl_catalog("sales personality behavior traits", keys_filter="Personality")
    for r in results:
        print(f"  [{r['test_type']}] {r['name']} — {r['link']}")

    print("\n=== Test 3: Technical SQL database skills ===")
    results = search_shl_catalog("SQL database query skills")
    for r in results:
        print(f"  [{r['test_type']}] {r['name']} — {r['link']}")

    print("\n=== Test 4: No Job Solutions in results? ===")
    results = search_shl_catalog("entry level customer service phone solution")
    solution_found = any("Solution" in r["name"] for r in results)
    print(f"  Job Solution found in results: {solution_found} (should be False)")
