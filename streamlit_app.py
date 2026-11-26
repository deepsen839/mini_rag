import os
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Mini RAG | ProcessVenue Assessment",
    page_icon="🔎",
    layout="wide",
)

API_URL = st.sidebar.text_input(
    "Backend API URL",
    value=os.getenv("API_URL", "http://localhost:8000"),
    help="URL of the local FastAPI backend. Example: http://localhost:8000",
).rstrip("/")

REQUEST_TIMEOUT = 180


def api_post(path: str, **kwargs):
    """Send a request to the local FastAPI backend with useful error messages."""
    try:
        response = requests.post(
            f"{API_URL}{path}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to the backend at {API_URL}. "
            "Start the FastAPI server and verify the Backend API URL."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The backend request timed out. Document ingestion or answer generation "
            "may take longer; check the backend logs."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}") from exc


st.title("🔎 Mini RAG")
st.caption(
    "Upload or paste documents, retrieve relevant passages using the backend's "
    "hybrid retrieval pipeline, rerank them, and ask grounded questions with citations."
)

with st.sidebar:
    st.subheader("1 · Add knowledge")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, MD or CSV",
        type=["pdf", "txt", "md", "csv"],
    )

    if st.button(
        "Index uploaded file",
        disabled=uploaded is None,
        use_container_width=True,
    ):
        with st.spinner("Extracting, chunking, embedding and indexing…"):
            try:
                result = api_post(
                    "/api/ingest/file",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    },
                )
                st.success("Document indexed successfully.")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("2 · Retrieval settings")
    top_k = st.slider(
        "Candidate passages (top-k)",
        min_value=3,
        max_value=30,
        value=12,
        help="Controls how many candidates are considered before reranking.",
    )
    st.caption(
        "The backend should combine dense vector retrieval and BM25 lexical retrieval, "
        "then rerank the strongest candidates."
    )

    st.divider()
    st.subheader("3 · Optional document filter")
    source_filter = st.text_input(
        "Source filename (optional)",
        value="",
        placeholder="Leave blank to search all documents",
        help=(
            "Leave blank for automatic document selection. Enter the exact indexed "
            "source filename to restrict retrieval to one document. The backend must "
            "accept the optional 'source' field in /api/query."
        ),
    )

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Paste text")
    doc_title = st.text_input("Document title", "My document")
    doc_source = st.text_input("Source label", "pasted-text.txt")
    section = st.text_input("Section (optional)", "")
    pasted = st.text_area(
        "Document content",
        height=250,
        placeholder="Paste the text you want to index…",
    )

    if st.button(
        "Index pasted text",
        type="primary",
        disabled=not pasted.strip(),
    ):
        with st.spinner("Indexing pasted text…"):
            try:
                result = api_post(
                    "/api/ingest",
                    json={
                        "text": pasted,
                        "source": doc_source,
                        "title": doc_title,
                        "section": section or None,
                    },
                )
                st.success("Text indexed successfully.")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

with right:
    st.subheader("Ask a question")
    question = st.text_area(
        "Your question",
        height=130,
        placeholder="Ask something that should be answered from your indexed documents…",
    )

    ask = st.button(
        "Retrieve & answer",
        type="primary",
        disabled=not question.strip(),
        use_container_width=True,
    )

    if ask:
        request_body = {
            "question": question.strip(),
            "top_k": top_k,
        }
        if source_filter.strip():
            request_body["source"] = source_filter.strip()

        started = time.perf_counter()
        with st.spinner("Searching, reranking and generating a grounded answer…"):
            try:
                result = api_post("/api/query", json=request_body)
                result.setdefault("timings_ms", {})
                result.setdefault("usage", {})
                result.setdefault("citations", [])
                result.setdefault("pipeline", {})
                result["timings_ms"].setdefault(
                    "ui_request",
                    (time.perf_counter() - started) * 1000,
                )
                st.session_state["last_result"] = result
                st.session_state["last_question"] = question.strip()
                st.session_state["last_source_filter"] = source_filter.strip()
            except Exception as exc:
                st.error(str(exc))

result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("Answer")
    st.markdown(result.get("answer", "The backend did not return an answer."))

    timings = result.get("timings_ms") or {}
    usage = result.get("usage") or {}

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total backend latency", f"{float(timings.get('total', 0)):.0f} ms")
    col_b.metric("Prompt + completion tokens", usage.get("total_tokens", 0))
    col_c.metric(
        "Estimated LLM cost",
        f"${float(usage.get('estimated_cost_usd', 0)):.5f}",
    )

    if usage.get("cost_note"):
        st.caption(usage["cost_note"])

    st.subheader("Retrieved sources & citations")
    citations = result.get("citations") or []

    if not citations:
        st.info("No supporting passages were retrieved. Try another query or check the indexed document text.")
    else:
        for index, cite in enumerate(citations, start=1):
            title = cite.get("title", "Untitled")
            source_name = cite.get("source", "Unknown source")
            position = cite.get("position", "?")
            score = cite.get("score", "N/A")
            with st.expander(
                f"[{index}] {title} · {source_name} · chunk {position} · score {score}"
            ):
                if cite.get("section"):
                    st.caption(f"Section: {cite['section']}")
                if cite.get("page") is not None:
                    st.caption(f"Page: {cite['page']}")
                st.write(cite.get("text", ""))

    with st.expander("Pipeline diagnostics"):
        st.json(
            {
                "pipeline": result.get("pipeline", {}),
                "timings_ms": timings,
                "usage": usage,
                "query": st.session_state.get("last_question"),
                "source_filter": st.session_state.get("last_source_filter") or "Automatic / all documents",
            }
        )
