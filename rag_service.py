
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_PATH = ROOT / "chroma_db_multi"

COLLECTION_NAME = "advanced_rag_documents"
LEGACY_COLLECTION_NAMES = ("langchain",)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "qwen2.5:3b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_BATCH_SIZE = 500

RELEVANCE_THRESHOLD = 1.35
RERANK_SCORE_GAP = 3.5


# ============================================================
# LAZY VECTOR STORE
# ============================================================

class LazyVectorStore:
    """Expose vector-store operations through the RAG service."""

    def __init__(self, service: "RAGService"):
        self._service = service

    def _collection(self):
        self._service.ensure_runtime()
        return self._service.collection

    def get(self, **kwargs: Any) -> dict[str, Any]:
        return self._collection().get(**kwargs)

    def add_documents(
        self,
        documents: Iterable[Document],
        **kwargs: Any,
    ):
        self._service.ensure_runtime()
        return self._service.add_documents(list(documents), **kwargs)

    def delete(self, **kwargs: Any) -> None:
        return self._collection().delete(**kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ):
        return self._service.similarity_search_with_score(
            query, k=k, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 6,
        **kwargs: Any,
    ):
        return self._service.max_marginal_relevance_search(
            query, k=k, **kwargs
        )


# ============================================================
# RAG SERVICE
# ============================================================

class RAGService:

    def __init__(self) -> None:
        self.collection: Any | None = None
        self.embedding_model: SentenceTransformer | None = None
        self.llm: ChatOllama | None = None
        self.reranker: CrossEncoder | None = None

        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None
        self.conversation_history: list[dict[str, str]] = []

        self._runtime_ready = False

    # ========================================================
    # INITIALIZE RUNTIME
    # ========================================================

    def ensure_runtime(self) -> None:
        if self._runtime_ready:
            return

        self._initialize()

    def _initialize(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        if self.collection is None:
            client = chromadb.PersistentClient(
                path=str(CHROMA_PATH)
            )

            collection_names = {
                collection.name
                for collection in client.list_collections()
            }

            if COLLECTION_NAME in collection_names:
                self.collection = client.get_collection(
                    name=COLLECTION_NAME
                )
            else:
                target = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "l2"},
                )

                legacy_name = next(
                    (
                        name
                        for name in LEGACY_COLLECTION_NAMES
                        if name in collection_names
                    ),
                    None,
                )

                if legacy_name is not None and target.count() == 0:
                    legacy_collection = client.get_collection(
                        name=legacy_name
                    )

                    legacy_data = legacy_collection.get(
                        include=[
                            "documents",
                            "metadatas",
                            "embeddings",
                        ]
                    )

                    ids = legacy_data.get("ids", [])
                    texts = legacy_data.get("documents", [])
                    metadatas = legacy_data.get("metadatas", [])
                    embeddings = legacy_data.get("embeddings")

                    if ids and embeddings is not None:
                        for start in range(0, len(ids), MAX_BATCH_SIZE):
                            end = start + MAX_BATCH_SIZE

                            target.add(
                                ids=ids[start:end],
                                embeddings=embeddings[start:end],
                                documents=texts[start:end],
                                metadatas=metadatas[start:end],
                            )

                self.collection = target

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL
            )

        if self.llm is None:
            self.llm = ChatOllama(
                model=LLM_MODEL,
                temperature=0,
                num_predict=1024,
            )

        if self.reranker is None:
            self.reranker = CrossEncoder(RERANKER_MODEL)

        self._build_bm25_index()
        self._runtime_ready = True

    # ========================================================
    # REBUILD BM25 INDEX
    # ========================================================

    def _build_bm25_index(self) -> None:
        raw = self.collection.get(
            include=["documents", "metadatas"]
        )

        texts = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        self.documents = [
            Document(
                page_content=text,
                metadata=metadata or {},
            )
            for text, metadata in zip(texts, metadatas)
            if text
        ]

        if self.documents:
            tokenized_documents = [
                re.findall(
                    r"\w+",
                    doc.page_content.lower(),
                )
                for doc in self.documents
            ]

            self.bm25 = BM25Okapi(tokenized_documents)
        else:
            self.bm25 = None

    # ========================================================
    # ADD DOCUMENTS TO CHROMADB
    # ========================================================

    def add_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        self.ensure_runtime()

        if not documents:
            return []

        texts = [
            doc.page_content
            for doc in documents
        ]

        metadatas = [
            doc.metadata
            for doc in documents
        ]

        ids = [
            str(uuid.uuid4())
            for _ in documents
        ]

        # Embed before writing anything to ChromaDB.
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            convert_to_tensor=False,
            show_progress_bar=False,
        )

        # ChromaDB can accept only a limited number of items
        # per request, so write in batches.
        for start in range(0, len(ids), MAX_BATCH_SIZE):
            end = start + MAX_BATCH_SIZE

            self.collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )

        return ids

    # ========================================================
    # FIND CHUNKS BELONGING TO ONE PDF
    # ========================================================

    def _find_ids_for_file(
        self,
        file_name: str,
    ) -> list[str]:
        """
        Find matching chunks only in the active collection.

        ChromaDB returns IDs automatically. Do not include
        'ids' in the include argument.
        """
        self.ensure_runtime()

        target_name = Path(file_name).name

        existing = self.collection.get(
            include=["metadatas"]
        )

        matching_ids = []

        for doc_id, metadata in zip(
            existing.get("ids", []),
            existing.get("metadatas", []),
        ):
            if not metadata:
                continue

            source = str(metadata.get("source", ""))

            if Path(source).name == target_name:
                matching_ids.append(doc_id)

        return matching_ids

    def _delete_matches_for_file(
        self,
        file_name: str,
    ) -> int:
        """Delete all matching chunks from the active collection in safe batches."""
        ids_to_delete = self._find_ids_for_file(file_name)

        if ids_to_delete:
            for start in range(0, len(ids_to_delete), MAX_BATCH_SIZE):
                batch_ids = ids_to_delete[start:start + MAX_BATCH_SIZE]
                self.collection.delete(ids=batch_ids)

        return len(ids_to_delete)

    # ========================================================
    # LOAD PDF
    # ========================================================

    def load_pdf(
        self,
        pdf_path: str | Path,
    ) -> list[Document]:
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files can be processed."
            )

        documents: list[Document] = []

        try:
            import pymupdf

            with pymupdf.open(str(path)) as pdf:
                for page_number in range(pdf.page_count):
                    page = pdf[page_number]
                    text = page.get_text("text") or ""

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text.strip(),
                                metadata={
                                    "source": path.name,
                                    "page": page_number,
                                    "page_number": page_number + 1,
                                },
                            )
                        )

        except ImportError:
            from pypdf import PdfReader

            reader = PdfReader(str(path))

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "source": path.name,
                                "page": page_number - 1,
                                "page_number": page_number,
                            },
                        )
                    )

        return documents

    # ========================================================
    # SPLIT PDF PAGES INTO CHUNKS
    # ========================================================

    def _split_documents(
        self,
        pages: list[Document],
        source: str,
    ) -> list[Document]:
        chunks: list[Document] = []

        for page in pages:
            text = page.page_content
            start = 0

            while start < len(text):
                chunk_text = text[
                    start:start + CHUNK_SIZE
                ]

                if not chunk_text:
                    break

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **page.metadata,
                            "source": source,
                        },
                    )
                )

                if start + CHUNK_SIZE >= len(text):
                    break

                start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    # ========================================================
    # ADD OR REPLACE A PDF
    # ========================================================

    def add_pdf(
        self,
        pdf_path: str | Path,
    ) -> int:
        """
        Index a PDF.

        First read, split, and add the new chunks. Only after
        the new chunks have been added do we remove the previous
        chunks with the same filename.
        """
        self.ensure_runtime()

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        source = path.name

        # Process the PDF before modifying existing indexed data.
        pages = self.load_pdf(path)

        chunks = self._split_documents(
            pages,
            source,
        )

        if not chunks:
            raise ValueError(
                f"No extractable text was found in {source}. "
                "The PDF may be scanned or empty."
            )

        # Capture the OLD IDs before adding new chunks.
        # Otherwise, a later filename-based deletion could
        # accidentally delete the new chunks too.
        old_ids = self._find_ids_for_file(source)

        # If embedding or adding fails, the previous chunks
        # remain untouched.
        self.add_documents(chunks)

        # Remove the old version only after adding succeeds.
        if old_ids:
            for start in range(0, len(old_ids), MAX_BATCH_SIZE):
                batch_ids = old_ids[start:start + MAX_BATCH_SIZE]
                self.collection.delete(ids=batch_ids)

        # Refresh semantic/BM25 document state.
        self._build_bm25_index()

        return len(chunks)

    # ========================================================
    # DELETE A PDF
    # ========================================================

    def delete_pdf(
        self,
        file_name: str,
    ) -> int:
        """Remove a PDF's indexed chunks and its local file."""
        self.ensure_runtime()

        # Only accept a filename, not an arbitrary filesystem path.
        safe_name = Path(file_name).name

        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Invalid PDF filename.")

        file_path = DATA_DIR / safe_name

        # First remove chunks from the active searchable collection.
        chunks_removed = self._delete_matches_for_file(
            safe_name
        )

        # Rebuild BM25 after the ChromaDB deletion.
        self._build_bm25_index()

        # Remove the local PDF, if it exists.
        if file_path.exists():
            file_path.unlink()

        return chunks_removed

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ):
        self.ensure_runtime()

        if self.collection.count() == 0:
            return []

        query_vector = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(k, self.collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        if not documents or not documents[0]:
            return []

        return [
            (
                Document(
                    page_content=doc,
                    metadata=metadata or {},
                ),
                float(distance),
            )
            for doc, metadata, distance in zip(
                documents[0],
                metadatas[0],
                distances[0],
            )
        ]

    # ========================================================
    # MAX MARGINAL RELEVANCE COMPATIBILITY
    # ========================================================

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
    ):
        # Preserve the existing behavior of this project.
        return self.similarity_search_with_score(
            query,
            k=fetch_k,
        )[:k]

    # ========================================================
    # SCORE NORMALIZATION
    # ========================================================

    def normalize_scores(
        self,
        scores,
    ) -> list[float]:
        if scores is None or len(scores) == 0:
            return []

        scores = [float(score) for score in scores]

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0 for _ in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    # ========================================================
    # CONVERSATION QUERY REWRITING
    # ========================================================

    def get_previous_subject(
        self,
        previous_question: str,
    ) -> str | None:
        if not previous_question:
            return None
        question = previous_question.strip().rstrip("?.!")

        patterns = [
            r"^what\s+(?:is|are)\s+(?:the\s+)?(?:advantages?|disadvantages?|features?|types?|applications?|uses?|benefits?|differences?|examples?)\s+of\s+(.+)$",
            r"^what\s+(?:is|are)\s+(.+?)(?:'s)?\s+(?:advantages?|disadvantages?|features?|types?|applications?|uses?|benefits?|differences?|examples?)$",
            r"^(?:the\s+)?(?:advantages?|disadvantages?|features?|types?|applications?|uses?|benefits?|differences?|examples?)\s+of\s+(.+)$",
            r"^what\s+(?:is|are)\s+(.+)$",
            r"^explain\s+(?:the\s+)?(?:concept\s+of\s+)?(.+)$",
            r"^tell\s+me\s+about\s+(.+)$",
            r"^describe\s+(.+)$",
            r"^define\s+(.+)$",
            r"^how\s+does\s+(.+?)\s+work$",
        ]

        for pattern in patterns:
            match = re.match(pattern, question, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if not re.match(r"^(it|its|their|this|that|these|those)$", candidate, re.IGNORECASE):
                    candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE).strip()
                    if candidate:
                        return candidate

        return None

    def rewrite_query(
        self,
        question: str,
    ) -> str:
        if not self.conversation_history:
            return question

        pronouns_regex = r"\b(it|its|they|them|their|this|that|these|those)\b"

        if not re.search(pronouns_regex, question, re.IGNORECASE):
            return question

        # Search backward through conversation history to find the real subject
        subject = None
        for turn in reversed(self.conversation_history):
            prev_q = turn.get("question", "")
            subject = self.get_previous_subject(prev_q)
            if subject:
                break

        if not subject:
            return question

        rewritten = question
        rewritten = re.sub(r"\b(its|their)\b", f"{subject}'s", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\b(it|they|them|this|that|these|those)\b", subject, rewritten, flags=re.IGNORECASE)

        return rewritten

    # ========================================================
    # HYBRID SEARCH: SEMANTIC + BM25
    # ========================================================

    def hybrid_search(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        self.ensure_runtime()

        semantic_results = self.similarity_search_with_score(
            question,
            k=10,
        )

        results_map: dict[
            tuple[str, int | None, str],
            dict[str, Any],
        ] = {}

        for document, distance in semantic_results:
            key = self.document_key(document)

            results_map[key] = {
                "document": document,
                "semantic_score": 1.0 / (1.0 + distance),
                "keyword_score": 0.0,
            }

        if self.bm25 is not None:
            tokens = re.findall(
                r"\w+",
                question.lower(),
            )

            scores = self.bm25.get_scores(tokens)
            normalized_scores = self.normalize_scores(
                scores.tolist()
            )

            top_indices = sorted(
                enumerate(normalized_scores),
                key=lambda item: item[1],
                reverse=True,
            )[:10]

            for index, score in top_indices:
                document = self.documents[index]
                key = self.document_key(document)

                results_map.setdefault(
                    key,
                    {
                        "document": document,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                    },
                )

                results_map[key]["keyword_score"] = score

        results = []

        for item in results_map.values():
            item["score"] = (
                0.6 * item["semantic_score"]
                + 0.4 * item["keyword_score"]
            )
            results.append(item)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:8]

    # ========================================================
    # CROSSENCODER RERANKING
    # ========================================================

    def rerank_results(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.ensure_runtime()

        if not results:
            return []

        pairs = [
            (
                question,
                result["document"].page_content,
            )
            for result in results
        ]

        scores = self.reranker.predict(pairs)

        reranked = [
            {
                **result,
                "score": float(score),
            }
            for result, score in zip(results, scores)
        ]

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        if reranked:
            best_score = reranked[0]["score"]

            reranked = [
                item
                for item in reranked
                if item["score"]
                >= best_score - RERANK_SCORE_GAP
            ]

        return reranked[:4]

    # ========================================================
    # CREATE PDF CONTEXT
    # ========================================================

    def create_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        parts = []

        for result in results:
            document = result["document"]
            metadata = document.metadata

            page = metadata.get(
                "page_number",
                metadata.get("page", "Unknown"),
            )

            parts.append(
                f"SOURCE: {metadata.get('source', 'Unknown')}\n"
                f"PAGE: {page}\n\n"
                f"{document.page_content.strip()}"
            )

        return "\n---\n".join(parts)

    # ========================================================
    # FILTER CONTEXT BY QUESTION TOPIC
    # ========================================================

    def filter_context_by_topic(
        self,
        results: list[dict[str, Any]],
        question: str,
    ):
        if not results:
            return results

        search_text = self.rewrite_query(question) or question
        subject = self.get_previous_subject(search_text) or self.get_previous_subject(question)

        # If a specific subject is identified, prioritize chunks mentioning it
        if subject:
            subject_clean = subject.lower().strip()
            subject_words = [w for w in re.findall(r"[a-z0-9]+", subject_clean) if len(w) > 1]
            if subject_words:
                filtered = [
                    result
                    for result in results
                    if all(w in result["document"].page_content.lower() for w in subject_words)
                ]
                if filtered:
                    return filtered

        stop_words = {
            "the", "a", "an", "it", "its", "they",
            "them", "their", "this", "that", "these",
            "those", "what", "are", "is", "which",
            "who", "when", "where", "why", "how",
            "do", "does", "did", "have", "has", "had",
            "of", "in", "to", "for", "and", "or",
        }

        topic_words = [
            word
            for word in re.findall(r"[a-z0-9]+", search_text.lower())
            if word not in stop_words and len(word) > 2
        ]

        if not topic_words:
            return results

        filtered = [
            result
            for result in results
            if any(
                word in result["document"].page_content.lower()
                for word in topic_words
            )
        ]

        return filtered or results

    # ========================================================
    # DOCUMENT IDENTITY
    # ========================================================

    @staticmethod
    def document_key(
        doc: Document,
    ) -> tuple[str, int | None, str]:
        return (
            str(doc.metadata.get("source", "")),
            doc.metadata.get(
                "page",
                doc.metadata.get("page_number"),
            ),
            doc.page_content.strip(),
        )

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    @staticmethod
    def system_prompt(
        question: str,
        context: str,
    ) -> str:
        return f"""You are an expert document-based question answering assistant.

Answer the user's question directly, accurately, and concisely using ONLY the information provided in the PDF CONTEXT below.
Do not use any outside knowledge, assumptions, or hallucinations.

Strict Rules:
1. Answer specifically about the main subject of the question: "{question}".
2. If the user asks about Python, answer only about Python. If the user asks about Pandas, answer only about Pandas. If the user asks about NumPy, answer only about NumPy. If the user asks about Statistical Learning or Supervised Learning, answer specifically about that topic.
3. If the user asks for advantages, features, types, or applications, present them clearly with bullet points.
4. If the PDF CONTEXT does not contain enough relevant information to answer the question, answer with exactly:
I could not find enough information in the provided PDF documents.
5. Do not include introductory phrases like "Based on the provided context" or "According to the PDF". State the facts directly.

Question:
{question}

PDF CONTEXT:
{context}

Answer:"""

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        self.ensure_runtime()

        prompt = self.system_prompt(
            question,
            context,
        )

        response = self.llm.invoke(prompt)
        return response.content

    # ========================================================
    # STREAM ANSWER
    # ========================================================

    def generate_answer_stream(
        self,
        question: str,
        context: str,
    ):
        self.ensure_runtime()

        prompt = self.system_prompt(
            question,
            context,
        )

        for chunk in self.llm.stream(prompt):
            if chunk.content:
                yield chunk.content


# ============================================================
# SHARED SERVICE AND VECTOR STORE
# ============================================================

service = RAGService()
vectorstore = LazyVectorStore(service)


# ============================================================
# PUBLIC FUNCTIONS USED BY APP.PY
# ============================================================

def add_pdf(
    pdf_path: str | Path,
) -> int:
    return service.add_pdf(pdf_path)


def delete_pdf(
    file_name: str,
) -> int:
    return service.delete_pdf(file_name)


def rebuild_bm25_index() -> None:
    service.ensure_runtime()
    service._build_bm25_index()


def rewrite_query(
    question: str,
) -> str:
    return service.rewrite_query(question)


def hybrid_search(
    question: str,
) -> list[dict[str, Any]]:
    return service.hybrid_search(question)


def rerank_results(
    question: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return service.rerank_results(
        question,
        results,
    )


def create_context(
    results: list[dict[str, Any]],
) -> str:
    return service.create_context(results)


def generate_answer_stream(
    question: str,
    context: str,
    results: list[dict[str, Any]] | None = None,
):
    answer_question = question

    if results is not None:
        answer_question = service.rewrite_query(question) or question

        filtered_results = service.filter_context_by_topic(
            results,
            answer_question,
        )

        context = service.create_context(
            filtered_results
        )

    yield from service.generate_answer_stream(
        answer_question,
        context,
    )


def generate_answer(
    question: str,
    context: str,
    results: list[dict[str, Any]] | None = None,
) -> str:
    answer_question = question

    if results is not None:
        answer_question = service.rewrite_query(question) or question

        filtered_results = service.filter_context_by_topic(
            results,
            answer_question,
        )

        context = service.create_context(
            filtered_results
        )

    return service.generate_answer(
        answer_question,
        context,
    )


def clean_answer(
    answer: str,
) -> str:
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()

    return answer