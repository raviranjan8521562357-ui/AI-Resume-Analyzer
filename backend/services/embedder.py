"""
Embedding and similarity search using sentence-transformers + FAISS.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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


def build_index(chunks: list[str]) -> tuple[faiss.Index, list[str]]:
    """
    Build a FAISS index from text chunks.
    
    Args:
        chunks: List of text chunks to index
    
    Returns:
        Tuple of (FAISS index, original chunks list)
    """
    model = get_model()
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index (L2 distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks


def search_index(index: faiss.Index, chunks: list[str], query: str, top_k: int = 3) -> list[str]:
    """
    Search the FAISS index for chunks most similar to the query.
    
    Args:
        index: FAISS index
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
