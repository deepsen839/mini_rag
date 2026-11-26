import re

def _tokens(text: str) -> list[str]:
    """Whitespace token approximation; see README for tokenizer trade-off."""
    return re.findall(r"\S+", text)

def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    words = _tokens(text)
    if not words:
        return []
    size = max(50, size)
    overlap = min(max(0, overlap), size - 1)
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks
