import os
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    app_name: str = 'Mini RAG'
    qdrant_url: str = ''
    qdrant_api_key: str = ''
    qdrant_collection: str = 'minirag_documents'
    embedding_provider: str = 'openai'
    embedding_model: str = 'text-embedding-3-small'
    embedding_dimensions: int = 1536
    openai_api_key: str = ''
    openai_base_url: str = 'https://api.openai.com/v1'
    llm_provider: str = 'groq'
    llm_model: str = 'openai/gpt-oss-120b'
    llm_api_key: str = ''
    cohere_api_key: str = ''
    cohere_rerank_model: str = 'rerank-v3.5'
    chunk_size_tokens: int = 900
    chunk_overlap_tokens: int = 120
    retrieval_top_k: int = 12
    rerank_top_n: int = 5
    min_relevance_score: float = 0.18
    # Qdrant HNSW: conservative defaults for small/free clusters.
    hnsw_enabled: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef: int = 128
    max_upload_mb: int = 15
    # LangSmith tracing (optional). Prefer current LANGSMITH_* names; accept legacy aliases.
    langsmith_tracing: bool = Field(default=False, validation_alias=AliasChoices('LANGSMITH_TRACING', 'LANGCHAIN_TRACING_V2'))
    langsmith_api_key: str = Field(default='', validation_alias=AliasChoices('LANGSMITH_API_KEY', 'LANGCHAIN_API_KEY'))
    langsmith_project: str = Field(default='mini-rag-assessment', validation_alias=AliasChoices('LANGSMITH_PROJECT', 'LANGCHAIN_PROJECT'))
    langsmith_endpoint: str = Field(default='https://api.smith.langchain.com', validation_alias=AliasChoices('LANGSMITH_ENDPOINT', 'LANGCHAIN_ENDPOINT'))
    app_env: str = 'development'

settings = Settings()

# LangSmith reads these process variables directly. Bridge values loaded from .env
# by Pydantic so local runs and Docker runs behave consistently.
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ.setdefault('LANGSMITH_TRACING', 'true')
    os.environ.setdefault('LANGSMITH_API_KEY', settings.langsmith_api_key)
    os.environ.setdefault('LANGSMITH_PROJECT', settings.langsmith_project)
    os.environ.setdefault('LANGSMITH_ENDPOINT', settings.langsmith_endpoint)
