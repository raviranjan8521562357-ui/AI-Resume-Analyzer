"""
Embedding and similarity search using sentence-transformers.
Uses FAISS if available, otherwise falls back to numpy-based search.

All heavy imports (sentence-transformers, faiss, numpy) are LAZY
so that the FastAPI server can bind the port instantly on Render.
"""

# Lazy-loaded globals
_np = None
_faiss = None
_HAS_FAISS = None
_SentenceTransformer = None
_model = None


def _ensure_imports():
    """Lazy-load heavy dependencies on first use, not at import time."""
    global _np, _faiss, _HAS_FAISS, _SentenceTransformer

    if _np is not None:
        return  # Already imported

    import numpy as np
    _np = np

    try:
        import faiss
        _faiss = faiss
        _HAS_FAISS = True
    except ImportError:
        _HAS_FAISS = False
        print("WARNING: faiss not available — using numpy fallback for vector search")

    from sentence_transformers import SentenceTransformer
    _SentenceTransformer = SentenceTransformer


def get_model():
    """Load the embedding model (singleton pattern to avoid reloading)."""
    global _model
    _ensure_imports()
    if _model is None:
        print("Loading embedding model (first time only)...")
        _model = _SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return _model


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.

    Returns:
        Similarity score between 0 and 1
    """
    _ensure_imports()
    model = get_model()
    embeddings = model.encode([text_a, text_b])

    a = embeddings[0]
    b = embeddings[1]
    similarity = _np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b))

    return float(similarity)


# ─── Numpy fallback for FAISS ───

class _NumpyIndex:
    """Simple brute-force L2 search using numpy (FAISS fallback)."""
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query, k: int):
        diffs = self.embeddings - query
        dists = _np.sum(diffs ** 2, axis=1)
        k = min(k, len(dists))
        indices = _np.argsort(dists)[:k]
        distances = dists[indices]
        return _np.array([distances]), _np.array([indices])


def build_index(chunks: list[str]):
    """
    Build a search index from text chunks.
    Uses FAISS if available, otherwise numpy fallback.
    """
    _ensure_imports()
    model = get_model()
    embeddings = model.encode(chunks)
    embeddings = _np.array(embeddings).astype('float32')

    if _HAS_FAISS:
        dimension = embeddings.shape[1]
        index = _faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        index = _NumpyIndex(embeddings)

    return index, chunks


def search_index(index, chunks: list[str], query: str, top_k: int = 3) -> list[str]:
    """
    Search the index for chunks most similar to the query.
    """
    _ensure_imports()
    model = get_model()
    query_embedding = model.encode([query]).astype('float32')

    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results
