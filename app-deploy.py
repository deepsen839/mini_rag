
import asyncio
import time
import streamlit as st

from rag import service

st.set_page_config(
    page_title="Mini RAG | ProcessVenue Assessment",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Mini RAG")
st.caption(
    "Upload or paste a document, retrieve relevant passages, "
    "rerank them, and ask grounded questions with citations."
)


# Execute asynchronous RAG methods from Streamlit
def run_async(coroutine):
    return asyncio.run(coroutine)


# -----------------------------------
# Sidebar: Document upload
# -----------------------------------

with st.sidebar:
    st.subheader("1 · Add knowledge")

    uploaded = st.file_uploader(
        "Upload PDF, TXT, MD or CSV",
        type=["pdf", "txt", "md", "csv"]
    )

    if st.button(
        "Index uploaded file",
        disabled=uploaded is None,
        use_container_width=True
    ):
        with st.spinner("Extracting, chunking, embedding and indexing..."):
            try:
                import io
                from pypdf import PdfReader

                name = uploaded.name
                extension = name.rsplit(".", 1)[-1].lower()

                raw = uploaded.getvalue()

                if extension == "pdf":
                    reader = PdfReader(io.BytesIO(raw))
                    text = "\n".join(
                        page.extract_text() or ""
                        for page in reader.pages
                    )

                elif extension in ["txt", "md", "csv"]:
                    text = raw.decode("utf-8-sig")

                else:
                    raise ValueError("Unsupported file format.")

                result = run_async(
                    service.ingest(
                        text=text,
                        source=name,
                        title=name.rsplit(".", 1)[0]
                    )
                )

                st.success("Document indexed successfully!")
                st.json(result)

            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()

    st.subheader("2 · Retrieval settings")

    top_k = st.slider(
        "Initial vector search top-k",
        3, 30, 12
    )

    st.caption(
        "Retrieved passages are reranked before being sent to the LLM."
    )


# -----------------------------------
# Main interface
# -----------------------------------

left, right = st.columns([1, 1], gap="large")


with left:
    st.subheader("Paste text")

    title = st.text_input(
        "Document title",
        "My document"
    )

    source = st.text_input(
        "Source label",
        "pasted-text.txt"
    )

    section = st.text_input(
        "Section (optional)",
        ""
    )

    pasted = st.text_area(
        "Document content",
        height=250,
        placeholder="Paste the text you want to index..."
    )

    if st.button(
        "Index pasted text",
        type="primary",
        disabled=not pasted.strip()
    ):
        with st.spinner("Indexing document..."):
            try:
                result = run_async(
                    service.ingest(
                        text=pasted,
                        source=source,
                        title=title,
                        section=section or None
                    )
                )

                st.success("Document indexed successfully!")
                st.json(result)

            except Exception as e:
                st.error(f"Indexing failed: {e}")


with right:
    st.subheader("Ask a question")

    question = st.text_area(
        "Your question",
        height=130,
        placeholder="Ask something based on your indexed documents..."
    )

    ask = st.button(
        "Retrieve & answer",
        type="primary",
        disabled=not question.strip(),
        use_container_width=True
    )

    if ask:
        with st.spinner("Searching, reranking and generating..."):
            try:
                result = run_async(
                    service.query(
                        question=question,
                        top_k=top_k
                    )
                )

                st.session_state["last_result"] = result

            except Exception as e:
                st.error(f"Query failed: {e}")


# -----------------------------------
# Results
# -----------------------------------

result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("Answer")

    st.markdown(result["answer"])

    a, b, c = st.columns(3)

    a.metric(
        "Total latency",
        f"{result['timings_ms'].get('total', 0):.0f} ms"
    )

    b.metric(
        "Total tokens",
        result["usage"].get("total_tokens", 0)
    )

    c.metric(
        "Estimated LLM cost",
        f"${result['usage'].get('estimated_cost_usd', 0):.5f}"
    )

    st.caption(
        result["usage"].get("cost_note", "")
    )

    st.subheader("Retrieved sources & citations")

    if not result["citations"]:
        st.info("No supporting passages were retrieved.")

    for i, cite in enumerate(result["citations"], 1):
        with st.expander(
            f"[{i}] {cite['title']} · "
            f"{cite['source']} · "
            f"chunk {cite['position']} · "
            f"score {cite['score']}"
        ):
            if cite.get("section"):
                st.caption(f"Section: {cite['section']}")

            st.write(cite["text"])

    with st.expander("Pipeline diagnostics"):
        st.json({
            "pipeline": result["pipeline"],
            "timings_ms": result["timings_ms"],
            "usage": result["usage"]
        })
