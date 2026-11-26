
import os
from pathlib import Path

# Import spaces before Gradio and the RAG service.
import spaces
import gradio as gr
from pypdf import PdfReader
from docx import Document

from app.rag import service


# ZeroGPU registration handler.
# This is not part of the RAG pipeline.
@spaces.GPU(duration=10)
def zerogpu_check():
    return "ZeroGPU handler registered."


def read_uploaded_file(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(
            f"\n[Page {page_no}]\n{page.extract_text() or ''}"
            for page_no, page in enumerate(reader.pages, start=1)
        )

    if suffix == '.docx':
        document = Document(str(path))
        return '\n'.join(p.text for p in document.paragraphs if p.text.strip())

    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8-sig")

    raise ValueError(
        "Supported formats: PDF, DOCX, TXT, Markdown and CSV."
    )


async def index_files(file_paths):
    if not file_paths:
        return "Please upload one or more PDF, DOCX, TXT, MD or CSV files."
    results = []
    for file_path in file_paths:
        try:
            path = Path(file_path)
            text = read_uploaded_file(str(path))
            if not text.strip():
                results.append(f"Skipped {path.name}: no readable text found.")
                continue
            result = await service.ingest(text=text, source=path.name, title=path.stem)
            results.append(
                f"Indexed: {result['source']} | chunks: {result['chunks_created']} | "
                f"embedding: {result['embedding_model']} ({result['dimensions']}d)"
            )
        except Exception as exc:
            results.append(f"Failed {Path(file_path).name}: {type(exc).__name__}: {exc}")
    return "\n".join(results)


async def index_pasted_text(text, title, source, section):
    if not text or not text.strip():
        return "Please paste some document text first."

    try:
        result = await service.ingest(
            text=text.strip(),
            source=(source or "pasted-text.txt").strip(),
            title=(title or "Untitled document").strip(),
            section=section or None,
        )

        return (
            f"Indexed successfully: {result['source']}\n"
            f"Chunks created: {result['chunks_created']}\n"
            f"Embedding model: {result['embedding_model']} "
            f"({result['dimensions']} dimensions)\n"
            f"Qdrant collection: {result['collection']}"
        )

    except Exception as exc:
        return f"Indexing failed: {type(exc).__name__}: {exc}"


def get_source_choices():
    """Return exact source filenames stored in Qdrant."""
    try:
        return ["All documents"] + [
            item["source"] for item in service.list_documents()
        ]
    except Exception:
        return ["All documents"]


def refresh_source_dropdown():
    return gr.update(choices=get_source_choices(), value="All documents")


async def answer_question(message, history, source_filter):
    if not message or not message.strip():
        return "Please enter a question."

    try:
        result = await service.query(
            question=message.strip(),
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "12")),
            source=(source_filter if source_filter and source_filter != "All documents" else None),
        )

        answer = result["answer"]
        citations = result.get("citations", [])

        if citations:
            answer += "\n\n### Retrieved sources\n"

            for index, cite in enumerate(citations, 1):
                answer += (
                    f"\n**[{index}] {cite['title']}** — "
                    f"{cite['source']} "
                    f"(chunk {cite['position']}, "
                    f"score {cite['score']})\n\n"
                    f"> {cite['text'][:1200]}\n"
                )

        timings = result.get("timings_ms", {})
        usage = result.get("usage", {})
        pipeline = result.get("pipeline", [])
        if pipeline:
            answer += "\\n\\n### Retrieval diagnostics\\n"
            answer += "\\n".join(f"- {item}" for item in pipeline)

        answer += (
            "\n---\n"
            f"**Latency:** {timings.get('total', 0):.1f} ms  \n"
            f"**Tokens:** {usage.get('total_tokens', 0)}  \n"
            f"**Estimated LLM cost:** "
            f"${usage.get('estimated_cost_usd', 0):.6f}  \n"
            f"**Reranker:** {usage.get('reranker', 'unknown')}"
        )

        return answer

    except Exception as exc:
        return f"Query failed: {type(exc).__name__}: {exc}"


with gr.Blocks(
    title="Mini RAG — ProcessVenue Assessment"
) as demo:

    gr.Markdown(
        "# 🔎 Mini RAG\n"
        "Upload multiple documents or paste text, retrieve relevant passages, "
        "rerank them, and ask grounded questions with citations."
    )

    # Hidden ZeroGPU registration event.
    # Do not trigger it automatically.
    zerogpu_button = gr.Button(visible=False)
    zerogpu_output = gr.Textbox(visible=False)

    zerogpu_button.click(
        fn=zerogpu_check,
        inputs=[],
        outputs=zerogpu_output,
        api_name=False,
    )

    with gr.Tab("Upload document"):
        upload = gr.File(
            label="Upload one or more PDF, DOCX, TXT, Markdown or CSV files",
            file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
            file_count="multiple",
            type="filepath",
        )

        index_upload = gr.Button(
            "Index uploaded file",
            variant="primary",
        )

        upload_status = gr.Textbox(
            label="Indexing status",
            lines=5,
            interactive=False,
        )

        index_upload.click(
            index_files,
            inputs=upload,
            outputs=upload_status,
        )

    with gr.Tab("Paste text"):
        doc_title = gr.Textbox(
            label="Document title",
            value="My document",
        )

        source_label = gr.Textbox(
            label="Source label",
            value="pasted-text.txt",
        )

        section = gr.Textbox(
            label="Section (optional)",
        )

        pasted_text = gr.Textbox(
            label="Document content",
            lines=12,
            placeholder="Paste document text here...",
        )

        index_text = gr.Button(
            "Index pasted text",
            variant="primary",
        )

        paste_status = gr.Textbox(
            label="Indexing status",
            lines=5,
            interactive=False,
        )

        index_text.click(
            index_pasted_text,
            inputs=[
                pasted_text,
                doc_title,
                source_label,
                section,
            ],
            outputs=paste_status,
        )

    with gr.Tab("Ask questions"):
        with gr.Row():
            source_filter = gr.Dropdown(
                choices=get_source_choices(),
                value="All documents",
                label="Search scope",
                interactive=True,
            )
            refresh_sources = gr.Button("Refresh document list")
        refresh_sources.click(
            fn=refresh_source_dropdown,
            inputs=[],
            outputs=[source_filter],
        )
        gr.ChatInterface(
            fn=answer_question,
            additional_inputs=[source_filter],
            title="Chat with your indexed documents",
            description="Select an indexed document to restrict retrieval, or choose All documents for automatic source selection across the collection.",
            type="messages",
        )

    gr.Markdown(
        "---\n"
        "**Stack:** Gradio · Qdrant Cloud · "
        "Embeddings · Groq GPT-OSS · LangSmith"
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
