"""
Embedding and similarity search using sentence-transformers.
Uses FAISS if available, otherwise falls back to numpy-based search.

All heavy imports (sentence-transformers, faiss, numpy) are LAZY
so that the FastAPI server can bind the port instantly on Render.

If sentence-transformers / local model fails (e.g. OOM on free hosting),
falls back to HuggingFace Inference API for embeddings.
"""
import os
import requests as _requests
from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded globals
_np = None
_faiss = None
_HAS_FAISS = None
_SentenceTransformer = None
_model = None
_USE_API_FALLBACK = False  # True when local model can't load


def _ensure_numpy():
    """Lazy-load numpy."""
    global _np
    if _np is not None:
        return
    import numpy as np
    _np = np


def _ensure_imports():
    """Lazy-load heavy dependencies on first use, not at import time."""
    global _np, _faiss, _HAS_FAISS, _SentenceTransformer, _USE_API_FALLBACK

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

    try:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    except Exception as e:
        _SentenceTransformer = None
        _USE_API_FALLBACK = True
        print(f"WARNING: sentence-transformers failed to import: {e}")
        print("         Will use HuggingFace Inference API for embeddings.")


# ─── HuggingFace Inference API fallback for embeddings ───

_HF_EMBED_URL = "https://router.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def _get_hf_headers() -> dict:
    """Get authorization headers for HF API."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not set. Add it to your .env file.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _embed_via_api(texts: list[str]) -> list:
    """Get embeddings via HuggingFace Inference API (fallback)."""
    _ensure_numpy()
    headers = _get_hf_headers()

    # HF feature-extraction pipeline accepts a list of strings
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(3):
        try:
            response = _requests.post(
                _HF_EMBED_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                embeddings = response.json()
                return _np.array(embeddings, dtype="float32")

            if response.status_code in (503, 429):
                import time
                wait = 3 * (attempt + 1)
                print(f"[embed-api] HTTP {response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue

            # Other error
            print(f"[embed-api] HTTP {response.status_code}: {response.text[:200]}")
            break

        except _requests.exceptions.Timeout:
            import time
            wait = 3 * (attempt + 1)
            print(f"[embed-api] Timeout, retrying in {wait}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"[embed-api] Error: {e}")
            break

    raise ValueError("HuggingFace embedding API failed after retries")


def get_model():
    """Load the embedding model (singleton pattern to avoid reloading)."""
    global _model, _USE_API_FALLBACK
    _ensure_imports()

    if _USE_API_FALLBACK:
        # No local model — will use API
        return None

    if _SentenceTransformer is None:
        _USE_API_FALLBACK = True
        print("[embedder] SentenceTransformer not available, switching to API fallback")
        return None

    if _model is None:
        try:
            print("Loading embedding model (first time only)...")
            _model = _SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded!")
        except Exception as e:
            print(f"[embedder] Failed to load local model: {e}")
            print("[embedder] Switching to HuggingFace API fallback")
            _USE_API_FALLBACK = True
            return None

    return _model


def _encode(texts: list[str]):
    """Encode texts using local model or API fallback."""
    _ensure_imports()
    model = get_model()

    if model is not None:
        return model.encode(texts)
    else:
        return _embed_via_api(texts)


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.

    Returns:
        Similarity score between 0 and 1
    """
    _ensure_numpy()
    embeddings = _encode([text_a, text_b])

    a = embeddings[0]
    b = embeddings[1]
    similarity = _np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b))

    return float(max(0.0, min(1.0, similarity)))


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
    embeddings = _encode(chunks)
    embeddings = _np.array(embeddings).astype('float32')

    if _HAS_FAISS and _faiss is not None:
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
    _ensure_numpy()
    query_embedding = _encode([query])
    query_embedding = _np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results
