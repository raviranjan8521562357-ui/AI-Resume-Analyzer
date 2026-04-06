"""
Embedding and similarity search using sentence-transformers.
Uses FAISS if available, otherwise falls back to numpy-based search.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import faiss; fall back to numpy if not available (e.g. on Render)
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    print("WARNING: faiss not available — using numpy fallback for vector search")

# Singleton model instance — loaded once, reused across requests
_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model (singleton pattern to avoid reloading)."""
    global _model
    if _model is None:
        print("Loading embedding model (first time only)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return _model


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.
    
    Returns:
        Similarity score between 0 and 1
    """
    model = get_model()
    embeddings = model.encode([text_a, text_b])
    
    # Cosine similarity
    a = embeddings[0]
    b = embeddings[1]
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    return float(similarity)


# ─── Numpy fallback for FAISS ───

class _NumpyIndex:
    """Simple brute-force L2 search using numpy (FAISS fallback)."""
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def search(self, query: np.ndarray, k: int):
        # L2 distances
        diffs = self.embeddings - query  # (n, d)
        dists = np.sum(diffs ** 2, axis=1)  # (n,)
        k = min(k, len(dists))
        indices = np.argsort(dists)[:k]
        distances = dists[indices]
        return np.array([distances]), np.array([indices])


def build_index(chunks: list[str]):
    """
    Build a search index from text chunks.
    Uses FAISS if available, otherwise numpy fallback.
    
    Args:
        chunks: List of text chunks to index
    
    Returns:
        Tuple of (index, original chunks list)
    """
    model = get_model()
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    if _HAS_FAISS:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        index = _NumpyIndex(embeddings)
    
    return index, chunks


def search_index(index, chunks: list[str], query: str, top_k: int = 3) -> list[str]:
    """
    Search the index for chunks most similar to the query.
    
    Args:
        index: FAISS or numpy index
        chunks: Original text chunks
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of most relevant text chunks
    """
    model = get_model()
    query_embedding = model.encode([query]).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))
    
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results
