# MiniRAG implementation update

## Included

- Configurable Qdrant HNSW graph settings (`m=16`, `ef_construct=100`) and search-time `hnsw_ef=128`.
- Applies HNSW settings to an existing collection without deleting vectors.
- Ensures a Qdrant KEYWORD payload index exists for the `source` field.
- Multi-file upload in the Gradio and Streamlit interfaces.
- PDF, DOCX, TXT, Markdown, and CSV extraction.
- Search across all indexed documents by default; optional exact filename filter.
- Per-source replacement on re-index, with other uploaded files preserved.
- Document inventory and per-document deletion API; Streamlit includes a document list/delete control.
- PDF page metadata captured from extracted page markers and surfaced in citations.

## Validation

- Python compilation passed for API, Gradio UI, and Streamlit UI.
- Existing test suite: 2 passed.
- Live Qdrant/HNSW operations were not exercised because this environment has no access to the user's Qdrant credentials or cluster.
