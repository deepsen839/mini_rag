from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from docx import Document
from io import BytesIO
from .config import settings
from .schemas import IngestRequest, QueryRequest
from .rag import service
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title='Mini RAG API', version='1.0.0', description='Cloud-vector RAG with retrieval, reranking and cited answers')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

@app.get('/health')
def health():
    return {'status': 'ok', 'qdrant_configured': bool(settings.qdrant_url and settings.qdrant_api_key),
            'embedding_model': settings.embedding_model, 'llm_model': settings.llm_model,
            'collection': settings.qdrant_collection, 'langsmith_tracing': bool(settings.langsmith_tracing and settings.langsmith_api_key)}

@app.post('/api/ingest')
async def ingest(payload: IngestRequest):
    try:
        return await service.ingest(payload.text, payload.source, payload.title, payload.section)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'Ingestion failed: {e}')

@app.post('/api/ingest/file')
async def ingest_file(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f'File exceeds {settings.max_upload_mb} MB limit.')
    name = file.filename or 'uploaded-file'
    try:
        if name.lower().endswith('.pdf') or file.content_type == 'application/pdf':
            reader = PdfReader(BytesIO(raw))
            pages = []
            for page_no, page in enumerate(reader.pages, start=1):
                content = page.extract_text() or ''
                if content.strip(): pages.append(f'\n[Page {page_no}]\n{content}')
            text = '\n'.join(pages)
            if not text.strip(): raise ValueError('No extractable text found. Scanned PDFs need OCR before upload.')
        elif name.lower().endswith(('.txt', '.md', '.csv')):
            text = raw.decode('utf-8-sig')
        elif name.lower().endswith('.docx'):
            document = Document(BytesIO(raw))
            text = '\n'.join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
        else:
            raise ValueError('Supported formats: PDF, DOCX, TXT, Markdown and CSV.')
        return await service.ingest(text, name, name.rsplit('.', 1)[0])
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not process file: {e}')


@app.get('/api/documents')
async def list_documents():
    try:
        return {'documents': service.list_documents()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'Could not list documents: {e}')

@app.delete('/api/documents')
async def delete_document(source: str):
    try:
        return service.delete_document(source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'Could not delete document: {e}')

@app.post('/api/query')
async def query(payload: QueryRequest):
    try:
        return await service.query(
            payload.question,
            payload.top_k,
            source=payload.source,
        )

    except RuntimeError as e:
        logger.exception("RAG runtime error")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        logger.exception("Unexpected error in RAG query")
        raise HTTPException(
            status_code=502,
            detail=f"Query failed: {str(e)}"
        )
