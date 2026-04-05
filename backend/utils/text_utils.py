"""
Text utility functions for cleaning and chunking resume text.
"""
import re


def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize text."""
    # Replace multiple newlines with single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(lines)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for RAG retrieval.
    
    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence boundary
        if end < len(text):
            # Look for the last period, newline, or sentence-ender before 'end'
            breakpoint = max(
                text.rfind('. ', start, end),
                text.rfind('\n', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end),
            )
            if breakpoint > start:
                end = breakpoint + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    # Filter out empty chunks
    return [c for c in chunks if c]
