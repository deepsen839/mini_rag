import re, time, uuid, hashlib, unicodedata, math
from collections import Counter
from typing import Any
import httpx
from groq import AsyncGroq
from langsmith import traceable
from qdrant_client import QdrantClient, models
from .config import settings
from .chunking import chunk_text
from sentence_transformers import SentenceTransformer
from .config import settings
class RAGService:
    
    def __init__(self):
        self.client = None
        self.llm_client = None
        self.embedding_model = None
        self._lexical_cache = None
        self._lexical_cache_at = 0.0

        if settings.embedding_provider.lower() == "local":
            self.embedding_model = SentenceTransformer(
                settings.embedding_model
            )

        if settings.qdrant_url and settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=20,
            )

            # Preserve existing documents.
            self._ensure_collection()


    def _ensure_collection(self):
        """Create/upgrade collection settings and required payload indexes.

        Existing vectors are preserved. HNSW settings are applied to the
        configured collection; source filtering remains optional at query time.
        """
        assert self.client
        names = {c.name for c in self.client.get_collections().collections}
        if settings.qdrant_collection not in names:
            self.client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=settings.hnsw_m,
                        ef_construct=settings.hnsw_ef_construct,
                    ) if settings.hnsw_enabled else None,
                ),
            )
        elif settings.hnsw_enabled:
            # Updating HNSW parameters does not delete points. Qdrant may
            # rebuild/reindex segments in the background after this change.
            self.client.update_collection(
                collection_name=settings.qdrant_collection,
                hnsw_config=models.HnswConfigDiff(
                    m=settings.hnsw_m,
                    ef_construct=settings.hnsw_ef_construct,
                ),
            )

        collection = self.client.get_collection(settings.qdrant_collection)
        payload_schema = collection.payload_schema or {}
        source_schema = payload_schema.get("source")
        if source_schema is None:
            self.client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        else:
            data_type = getattr(source_schema, "data_type", None)
            if data_type is None and isinstance(source_schema, dict):
                data_type = source_schema.get("data_type")
            if data_type is not None and str(data_type).lower() not in {"keyword", "payloadschematype.keyword"}:
                raise RuntimeError(
                    f"Qdrant payload field 'source' has type {data_type!r}; "
                    "MiniRAG requires a KEYWORD index for exact source filtering."
                )

    async def _post(self, url: str, headers: dict, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    @traceable(name="embed_documents")
    async def embed(self, texts: list[str]) -> list[list[float]]:

        provider = settings.embedding_provider.lower()

        if provider == "local":
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(
                    settings.embedding_model
                )

            vectors = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            return vectors.tolist()

        elif provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required.")

            data = await self._post(
                f'{settings.openai_base_url.rstrip("/")}/embeddings',
                {
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json"
                },
                {
                    "model": settings.embedding_model,
                    "input": texts,
                    "dimensions": settings.embedding_dimensions
                }
            )

            return [
                row["embedding"]
                for row in sorted(
                    data["data"],
                    key=lambda x: x["index"]
                )
            ]

        else:
            raise RuntimeError(
                f"Unsupported embedding provider: {provider}"
            )

    @traceable(name='ingest_document')
    async def ingest(self, text: str, source: str, title: str, section: str | None = None) -> dict:
        if not self.client:
            raise RuntimeError('Qdrant Cloud is not configured. Set QDRANT_URL and QDRANT_API_KEY.')
        source = (source or 'uploaded-document').strip()
        title = (title or source).strip()
        chunks = chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
        if not chunks:
            raise ValueError('No readable text was found in the document.')
        vectors = await self.embed(chunks)

        # Same filename means replace that document; different filenames coexist.
        self.client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source))
                ])
            ),
            wait=True,
        )

        document_id = hashlib.sha256(source.encode('utf-8')).hexdigest()
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = hashlib.sha256(f"{document_id}:{i}:{chunk}".encode("utf-8")).hexdigest()
            page_match = re.search(r"\[Page\s+(\d+)\]", chunk, flags=re.IGNORECASE)
            page = int(page_match.group(1)) if page_match else None
            payload = {
                'text': chunk, 'source': source, 'document_id': document_id,
                'title': title, 'section': section, 'page': page,
                'position': i, 'chunk_id': chunk_id,
            }
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector=vector,
                payload=payload,
            ))
        self.client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
        self._lexical_cache = None  # refresh lexical index after ingestion
        return {
            'source': source, 'document_id': document_id,
            'chunks_created': len(chunks), 'collection': settings.qdrant_collection,
            'embedding_model': settings.embedding_model,
            'dimensions': settings.embedding_dimensions,
        }

    def list_documents(self) -> list[dict]:
        """Return distinct indexed sources using a paginated Qdrant scroll."""
        if not self.client:
            raise RuntimeError('Qdrant Cloud is not configured.')
        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=['source', 'title', 'document_id'],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                source = payload.get('source')
                if source:
                    item = docs.setdefault(source, {
                        'source': source,
                        'title': payload.get('title') or source,
                        'document_id': payload.get('document_id'),
                        'chunks': 0,
                    })
                    item['chunks'] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda item: item['source'].lower())

    def delete_document(self, source: str) -> dict:
        if not self.client:
            raise RuntimeError('Qdrant Cloud is not configured.')
        source = (source or '').strip()
        if not source:
            raise ValueError('A source filename is required.')
        self.client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key='source', match=models.MatchValue(value=source))
            ])),
            wait=True,
        )
        return {'deleted_source': source}

    @staticmethod
    def _source_key(value: str) -> str:
        """Normalize punctuation, whitespace and case in a filename."""
        name = unicodedata.normalize("NFKC", str(value or "")).replace("\\\\", "/").split("/")[-1]
        return re.sub(r"[^a-z0-9]+", "", name.casefold())

    def resolve_source(self, requested_source: str | None) -> str | None:
        """Resolve a user filename to the exact source string stored in Qdrant."""
        if not requested_source or not requested_source.strip():
            return None
        requested = requested_source.strip()
        documents = self.list_documents()
        exact = [d["source"] for d in documents if d["source"] == requested]
        if exact:
            return exact[0]
        key = self._source_key(requested)
        matches = [d["source"] for d in documents if self._source_key(d["source"]) == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("The filename matches multiple indexed documents: " + ", ".join(matches))
        names = [d["source"] for d in documents]
        available = "; ".join(names[:20]) if names else "(no documents indexed)"
        raise ValueError(f"No indexed document matches '{requested}'. Available sources: {available}")


    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        """Tokenize English and Unicode technical terms without external dependencies."""
        return re.findall(r"[a-zA-Z0-9_+#.-]+|[^\W\d_]+", (text or "").casefold(), flags=re.UNICODE)

    def _load_lexical_documents(self) -> list[dict]:
        """Load and cache chunk payloads for BM25-style lexical retrieval."""
        now = time.time()
        if self._lexical_cache is not None and now - self._lexical_cache_at < 300:
            return self._lexical_cache
        rows = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.qdrant_collection, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for point in points:
                p = point.payload or {}
                text = p.get("text") or ""
                if text.strip():
                    rows.append({
                        "id": str(point.id), "text": text,
                        "source": p.get("source", "unknown"),
                        "title": p.get("title", "Untitled"),
                        "section": p.get("section"),
                        "position": int(p.get("position", 0)),
                        "page": p.get("page"),
                        "_tokens": self._lexical_tokens(text),
                    })
            if offset is None:
                break
        self._lexical_cache = rows
        self._lexical_cache_at = now
        return rows

    @staticmethod
    def _bm25_rank(question: str, documents: list[dict], limit: int) -> list[dict]:
        """Small in-process BM25 implementation for exact technical-term recovery."""
        query_terms = RAGService._lexical_tokens(question)
        if not query_terms or not documents:
            return []
        tokenized = [d.get("_tokens") or RAGService._lexical_tokens(d.get("text", "")) for d in documents]
        n_docs = len(tokenized)
        avg_len = sum(map(len, tokenized)) / max(1, n_docs)
        doc_freq = Counter()
        term_freqs, lengths = [], []
        for tokens in tokenized:
            tf = Counter(tokens)
            term_freqs.append(tf)
            lengths.append(len(tokens))
            doc_freq.update(set(tf))
        k1, b = 1.5, 0.75
        scored = []
        for index, doc in enumerate(documents):
            score, tf, doc_len = 0.0, term_freqs[index], lengths[index]
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                denom = freq + k1 * (1.0 - b + b * doc_len / max(1.0, avg_len))
                score += idf * (freq * (k1 + 1.0)) / denom
            if any(term in tf for term in set(query_terms) if len(term) >= 4):
                score *= 1.15
            if score > 0:
                item = dict(doc)
                item.pop("_tokens", None)
                item["lexical_score"] = float(score)
                scored.append(item)
        return sorted(scored, key=lambda item: item["lexical_score"], reverse=True)[:limit]

    @traceable(name='retrieve_documents')
    async def retrieve(self, question: str, top_k: int, source: str | None = None) -> list[dict[str, Any]]:
        if not self.client:
            raise RuntimeError('Qdrant Cloud is not configured. Set QDRANT_URL and QDRANT_API_KEY.')
        resolved_source = self.resolve_source(source) if source else None
        query_filter = None
        if resolved_source:
            query_filter = models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=resolved_source))
            ])
        vector = (await self.embed([question]))[0]
        dense_limit = max(top_k * 4, 40)
        result = self.client.query_points(
            collection_name=settings.qdrant_collection, query=vector,
            query_filter=query_filter,
            search_params=models.SearchParams(hnsw_ef=settings.hnsw_ef, exact=False),
            limit=dense_limit, with_payload=True,
        )
        dense_docs = []
        for p in result.points:
            payload = p.payload or {}
            dense_docs.append({
                'id': str(p.id), 'text': payload.get('text', ''),
                'source': payload.get('source', 'unknown'),
                'title': payload.get('title', 'Untitled'),
                'section': payload.get('section'), 'position': int(payload.get('position', 0)),
                'page': payload.get('page'),
                'score': float(p.score if p.score is not None else 0.0),
                'retrieval_method': 'dense',
            })
        lexical_docs = self._load_lexical_documents()
        if resolved_source:
            lexical_docs = [d for d in lexical_docs if d["source"] == resolved_source]
        lexical_ranked = self._bm25_rank(question, lexical_docs, dense_limit)

        fused = {}
        rrf_k = 60
        for rank, doc in enumerate(dense_docs, start=1):
            entry = fused.setdefault(doc["id"], dict(doc))
            entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (rrf_k + rank)
            entry["dense_rank"] = rank
        for rank, doc in enumerate(lexical_ranked, start=1):
            entry = fused.setdefault(doc["id"], dict(doc))
            entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (rrf_k + rank)
            entry["lexical_score"] = doc["lexical_score"]
            entry["lexical_rank"] = rank
            if "score" not in entry:
                entry["score"] = 0.0
                entry["retrieval_method"] = "lexical"
        return sorted(fused.values(), key=lambda d: d.get("fusion_score", 0.0), reverse=True)[:max(top_k * 3, 20)]

    
    @traceable(name="rerank_documents")
    async def rerank(
        self,
        question: str,
        docs: list[dict],
        top_n: int
    ) -> tuple[list[dict], str]:

        if not docs:
            return [], "none"

        # Use Cohere when configured
        if settings.cohere_api_key:
            data = await self._post(
                "https://api.cohere.com/v2/rerank",
                {
                    "Authorization": f"Bearer {settings.cohere_api_key}",
                    "Content-Type": "application/json"
                },
                {
                    "model": settings.cohere_rerank_model,
                    "query": question,
                    "documents": [d["text"] for d in docs],
                    "top_n": min(top_n, len(docs))
                }
            )

            ranked = []

            for row in data.get("results", []):
                d = dict(docs[row["index"]])
                d["rerank_score"] = float(row["relevance_score"])
                ranked.append(d)

            return ranked, f"Cohere {settings.cohere_rerank_model}"

        # Unicode-aware lexical fallback
        q = set(
            re.findall(r"\w+", question.lower(), flags=re.UNICODE)
        )

        for d in docs:
            words = set(
                re.findall(
                    r"\w+",
                    d["text"].lower(),
                    flags=re.UNICODE
                )
            )

            lexical_score = len(q & words) / max(1, len(q))

            vector_score = max(0.0, min(1.0, d["score"]))

            # Combine lexical and semantic relevance
            d["rerank_score"] = (
                0.35 * lexical_score +
                0.65 * vector_score
            )

        ranked = sorted(
            docs,
            key=lambda d: d["rerank_score"],
            reverse=True
        )[:top_n]

        return ranked, "combined lexical + vector similarity"

    
    @traceable(name='generate_grounded_answer')
    async def generate(self, question: str, docs: list[dict]) -> tuple[str, dict]:
        if settings.llm_provider.lower() != 'groq':
            raise RuntimeError(
                f"Unsupported LLM_PROVIDER: {settings.llm_provider}"
            )

        if not settings.llm_api_key:
            raise RuntimeError('LLM_API_KEY is required.')

        if self.llm_client is None:
            self.llm_client = AsyncGroq(
                api_key=settings.llm_api_key,
                timeout=60.0,
                max_retries=2
            )

        if not docs:
            return (
                "I couldn't find relevant information in the indexed documents.",
                {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            )

        # Limit total context, not just individual chunks.
        MAX_CONTEXT_CHARS = 10000
        MAX_CHUNK_CHARS = 3000

        context_parts = []
        total_chars = 0

        for i, d in enumerate(docs):
            remaining = MAX_CONTEXT_CHARS - total_chars

            if remaining <= 0:
                break

            chunk_text = d.get('text', '')[:MAX_CHUNK_CHARS]

            metadata = (
                f"[{i+1}] Source: {d['source']} | "
                f"Title: {d['title']} | "
                f"Section: {d.get('section') or 'N/A'} | "
                f"Chunk: {d['position']}\n"
            )

            available = remaining - len(metadata)

            if available <= 0:
                break

            chunk_text = chunk_text[:available]

            part = metadata + chunk_text
            context_parts.append(part)
            total_chars += len(part)

        context = '\n\n'.join(context_parts)

        system = (
            'You are a careful retrieval-grounded assistant. '
            'Answer only from the supplied context. '
            'Cite every factual claim with inline references '
            'like [1] or [2]. Never invent citations. '
            'If the context does not contain the answer, '
            'say so explicitly. Keep the answer concise.'
        )

        completion = await self.llm_client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.1,
            max_tokens=800,
            messages=[
                {'role': 'system', 'content': system},
                {
                    'role': 'user',
                    'content': (
                        f'Question: {question}\n\n'
                        f'Context:\n{context}'
                    )
                },
            ],
        )

        usage = completion.usage

        return completion.choices[0].message.content or '', {
            'prompt_tokens': int(usage.prompt_tokens if usage else 0),
            'completion_tokens': int(usage.completion_tokens if usage else 0),
            'total_tokens': int(usage.total_tokens if usage else 0)
        }


    @traceable(name='rag_query')
    async def query(self, question: str, top_k: int, source: str | None = None) -> dict:
        timings = {}; started = time.perf_counter()
        t = time.perf_counter()

        # A manually selected source always takes precedence. Otherwise, perform
        # a wider global search and infer the strongest source candidates from
        # the actual Qdrant results. Keep global candidates as a safety net.
        if source and source.strip():
            resolved_source = self.resolve_source(source)
            retrieved = await self.retrieve(question, top_k, source=resolved_source)
            selected_sources = [resolved_source]
            routing_mode = "manual source selection"
            global_count = len(retrieved)
        else:
            candidate_k = max(top_k * 4, 40)
            global_candidates = await self.retrieve(question, candidate_k, source=None)
            global_count = len(global_candidates)

            source_scores: dict[str, list[float]] = {}
            for item in global_candidates:
                source_scores.setdefault(item["source"], []).append(float(item["score"]))

            # Rank sources using their strongest matching passage, with the
            # number of supporting passages as a tie-breaker.
            selected_sources = sorted(
                source_scores,
                key=lambda name: (
                    max(source_scores[name]),
                    len(source_scores[name]),
                    sum(source_scores[name]) / max(1, len(source_scores[name])),
                ),
                reverse=True,
            )[:2]

            if selected_sources:
                routed = []
                for selected_source in selected_sources:
                    routed.extend(
                        await self.retrieve(question, top_k, source=selected_source)
                    )
                by_id = {item["id"]: item for item in global_candidates}
                by_id.update({item["id"]: item for item in routed})
                retrieved = sorted(
                    by_id.values(), key=lambda item: float(item["score"]), reverse=True
                )[:max(top_k * 2, 20)]
                routing_mode = "automatic source selection"
            else:
                retrieved = global_candidates[:top_k]
                routing_mode = "global search fallback"

        timings['retrieval'] = (time.perf_counter()-t)*1000
        raw_retrieved_count = len(retrieved)
        raw_top_scores = [
            {"source": d.get("source"), "chunk": d.get("position"),
             "dense": round(float(d.get("score", 0.0)), 10),
             "bm25": round(float(d.get("lexical_score", 0.0)), 4),
             "rrf": round(float(d.get("fusion_score", 0.0)), 6)}
            for d in retrieved[:8]
        ]
        t = time.perf_counter(); ranked, reranker = await self.rerank(question, retrieved, settings.rerank_top_n)
        timings['reranking'] = (time.perf_counter()-t)*1000

        # Reject weak matches before sending unrelated passages to the LLM.
        min_score = max(0.0, min(1.0, settings.min_relevance_score))
        reranked_count = len(ranked)
        ranked = [d for d in ranked if d.get('rerank_score', 0.0) >= min_score]
        rejected_count = reranked_count - len(ranked)

        t = time.perf_counter(); answer, usage = await self.generate(question, ranked)
        timings['generation'] = (time.perf_counter()-t)*1000
        timings['total'] = (time.perf_counter()-started)*1000
        citations = [{'id': d['id'], 'source': d['source'], 'title': d['title'], 'section': d['section'],
                      'position': d['position'], 'page': d.get('page'), 'text': d['text'], 'score': round(d['score'], 4)} for d in ranked]
        # Rough, explicitly labelled estimate. Rates can be changed as provider pricing changes.
        usage['estimated_cost_usd'] = round((usage['prompt_tokens']*0.00000059 + usage['completion_tokens']*0.00000079), 6)
        usage['cost_note'] = 'Rough Groq-hosted GPT-OSS estimate; excludes embedding/rerank costs and may differ by model/plan.'
        usage['reranker'] = reranker
        return {'answer': answer, 'citations': citations, 'timings_ms': {k: round(v, 1) for k,v in timings.items()},
                'usage': usage, 'pipeline': [
                    'query embedding',
                    f'hybrid retrieval: dense top {max(top_k * 4, 40)} + BM25 lexical scan, RRF fusion',
                    f'source routing: {routing_mode}',
                    f'selected source(s): {selected_sources if selected_sources else "none"}',
                    f'global candidate results: {global_count}',
                    f'candidate results after routing: {raw_retrieved_count}',
                    f'top hybrid retrieval evidence (dense/BM25/RRF): {raw_top_scores}',
                    f'reranker: {reranker}',
                    f'reranked candidates: {reranked_count}',
                    f'candidates rejected by threshold: {rejected_count}',
                    f'final context chunks: {len(ranked)}',
                    f'min relevance score: {min_score}',
                    f'LLM: {settings.llm_model}',
                ]}

service = RAGService()
