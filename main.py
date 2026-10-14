import os
import re
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# Import shared embeddings and vectorstore from ingestion to ensure synchronization
from ingestion import embeddings, vectorstore


# =========================================================
# 1. LOAD ALL DOCUMENTS FOR BM25
# =========================================================

print("=" * 60)
print("Loading documents for BM25...")
print("=" * 60)

documents = []
bm25 = None
doc_key_to_index = {}


def rebuild_bm25_index():
    global documents, bm25, bm25_documents, doc_key_to_index

    print("=" * 60)
    print("Rebuilding BM25 index...")
    print("=" * 60)

    try:
        all_data = vectorstore.get()
    except Exception as e:
        print(f"Error reading vectorstore: {e}")
        all_data = {
            "documents": [],
            "metadatas": [],
            "ids": []
        }

    new_docs = []
    text_list = all_data.get("documents") or []
    meta_list = all_data.get("metadatas") or []

    for i, text_content in enumerate(text_list):
        metadata = (
            meta_list[i]
            if i < len(meta_list) and meta_list[i]
            else {}
        )

        new_docs.append(
            Document(
                page_content=text_content,
                metadata=metadata
            )
        )

    documents = new_docs
    bm25_documents = documents

    doc_key_to_index = {
        (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content.strip()
        ): idx
        for idx, doc in enumerate(documents)
    }

    if documents:
        tokenized_documents = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        print("BM25 index rebuilt successfully.")
        print(f"BM25 documents: {len(documents)}")

    else:
        bm25 = None
        print("No documents found. BM25 index is empty.")


# Initial build
rebuild_bm25_index()
bm25_documents = documents


# =========================================================
# 2. LOAD QWEN2.5 3B (OLLAMA)
# =========================================================

print("=" * 60)
print("Loading Qwen2.5 3B...")
print("=" * 60)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_predict=1024
)

rewrite_llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_predict=100
)

print("Qwen2.5 3B loaded.")


# =========================================================
# 3. LOAD CROSS ENCODER RERANKER
# =========================================================

print("=" * 60)
print("Loading CrossEncoder reranker...")
print("=" * 60)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("CrossEncoder loaded successfully.")


# =========================================================
# 4. CONVERSATION MEMORY & THRESHOLDS
# =========================================================

conversation_history = []

RELEVANCE_THRESHOLD = 1.35
RERANK_SCORE_GAP = 3.5


# =========================================================
# 5. NORMALIZE SCORES
# =========================================================

def normalize_scores(scores):

    if scores is None or len(scores) == 0:
        return []

    minimum = float(min(scores))
    maximum = float(max(scores))

    if maximum == minimum:
        return [1.0 for _ in range(len(scores))]

    return [
        float(score - minimum) / float(maximum - minimum)
        for score in scores
    ]


# =========================================================
# 6. GET PREVIOUS QUESTION SUBJECT
# =========================================================

def get_previous_subject(previous_question):

    if not previous_question:
        return None

    question = previous_question.strip().rstrip("?")

    patterns = [
        r"^\s*what\s+is\s+(.+?)\s*$",
        r"^\s*what\s+are\s+(.+?)\s*$",
        r"^\s*explain\s+(.+?)\s*$",
        r"^\s*tell\s+me\s+about\s+(.+?)\s*$",
        r"^\s*how\s+does\s+(.+?)\s+work\s*$",
        r"^\s*define\s+(.+?)\s*$"
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    return None


# =========================================================
# 7. QUERY REWRITING
# =========================================================

def rewrite_query(question):

    # If there is no previous conversation,
    # return the question exactly as the user asked.
    if not conversation_history:
        return question

    previous = conversation_history[-1]

    previous_question = previous["question"]

    # -----------------------------------------------------
    # Detect genuine follow-up questions
    # -----------------------------------------------------

    follow_up_words = [
        "it",
        "its",
        "it's",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those"
    ]

    question_lower = question.lower()

    contains_follow_up = any(
        re.search(
            r"\b" + re.escape(word) + r"\b",
            question_lower
        )
        for word in follow_up_words
    )

    # -----------------------------------------------------
    # Direct pronoun resolution
    # -----------------------------------------------------

    if contains_follow_up:

        previous_subject = get_previous_subject(
            previous_question
        )

        if previous_subject:

            rewritten_question = question

            rewritten_question = re.sub(
                r"\bits\b",
                f"{previous_subject}'s",
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bit's\b",
                f"{previous_subject}'s",
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\btheir\b",
                f"{previous_subject}'s",
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bit\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bthey\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bthem\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bthis\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bthat\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bthese\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            rewritten_question = re.sub(
                r"\bthose\b",
                previous_subject,
                rewritten_question,
                flags=re.IGNORECASE
            )

            print(
                f"\nFollow-up subject detected: "
                f"{previous_subject}"
            )

            print(
                f"Rewritten follow-up query: "
                f"{rewritten_question}"
            )

            return rewritten_question

    # -----------------------------------------------------
    # Independent question
    # -----------------------------------------------------
    # Do NOT use Qwen to rewrite an independent question.
    #
    # Example:
    #
    # Previous:
    # what is python
    #
    # New:
    # what is statistical learning
    #
    # Keep:
    # what is statistical learning
    #
    # This prevents the previous topic from being added.
    # -----------------------------------------------------

    print(
        f"\nIndependent question detected: "
        f"{question}"
    )

    print(
        f"Search query: {question}"
    )

    return question


# =========================================================
# 8. TRUE HYBRID SEARCH
#    SEMANTIC + BM25 KEYWORD
# =========================================================

def hybrid_search(question):

    print("=" * 60)
    print("HYBRID SEARCH")
    print("=" * 60)

    if not documents:
        print("Vectorstore has no documents indexed.")
        return []

    # -----------------------------------------------------
    # A. SEMANTIC SEARCH
    # -----------------------------------------------------

    try:

        semantic_results = (
            vectorstore.similarity_search_with_score(
                question,
                k=10
            )
        )

    except Exception as e:

        print(f"Semantic search error: {e}")
        semantic_results = []

    semantic_docs_map = {}

    if semantic_results:

        semantic_distances = [
            res[1]
            for res in semantic_results
        ]

        semantic_similarities = [
            1.0 / (1.0 + d)
            for d in semantic_distances
        ]

        normalized_semantic = normalize_scores(
            semantic_similarities
        )

        for i, (doc, _) in enumerate(
            semantic_results
        ):

            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content.strip()
            )

            semantic_docs_map[key] = {
                "document": doc,
                "semantic_score": normalized_semantic[i]
            }

    # -----------------------------------------------------
    # B. BM25 KEYWORD SEARCH
    # -----------------------------------------------------

    bm25_docs_map = {}

    if bm25 is not None and len(documents) > 0:

        tokenized_query = question.lower().split()

        bm25_scores = bm25.get_scores(
            tokenized_query
        )

        normalized_bm25 = normalize_scores(
            bm25_scores
        )

        # Get top 10 BM25 document indexes
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True
        )[:10]

        for idx in top_bm25_indices:

            doc = documents[idx]

            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content.strip()
            )

            bm25_docs_map[key] = {
                "document": doc,
                "bm25_score": normalized_bm25[idx]
            }

    # -----------------------------------------------------
    # C. COMBINE CANDIDATE POOL
    # -----------------------------------------------------

    all_keys = (
        set(semantic_docs_map.keys())
        .union(set(bm25_docs_map.keys()))
    )

    hybrid_results = []

    for key in all_keys:

        if key in semantic_docs_map:

            doc = semantic_docs_map[key]["document"]

            sem_score = semantic_docs_map[key][
                "semantic_score"
            ]

        else:

            doc = bm25_docs_map[key]["document"]

            sem_score = 0.0

        if key in bm25_docs_map:

            kw_score = bm25_docs_map[key][
                "bm25_score"
            ]

        else:

            corpus_idx = doc_key_to_index.get(key)

            if (
                corpus_idx is not None
                and bm25 is not None
            ):

                kw_score = normalized_bm25[
                    corpus_idx
                ]

            else:

                kw_score = 0.0

        hybrid_score = (
            (0.6 * sem_score)
            +
            (0.4 * kw_score)
        )

        hybrid_results.append({
            "document": doc,
            "score": hybrid_score
        })

    # Sort descending by hybrid score

    hybrid_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top_results = hybrid_results[:8]

    print(
        f"Hybrid candidate count: "
        f"{len(hybrid_results)} "
        f"-> Selected top: "
        f"{len(top_results)}"
    )

    for i, res in enumerate(top_results):

        doc = res["document"]

        print(
            f"{i + 1}. "
            f"Score: {res['score']:.4f} | "
            f"Source: "
            f"{doc.metadata.get('source', 'Unknown')} | "
            f"Page: "
            f"{doc.metadata.get('page', 'Unknown')}"
        )

    return top_results


# =========================================================
# 9. CROSS ENCODER RERANKING
# =========================================================

def rerank_results(question, hybrid_results):

    print("=" * 60)
    print("CROSS ENCODER RERANKING")
    print("=" * 60)

    if not hybrid_results:
        return []

    pairs = [
        (
            question,
            res["document"].page_content
        )
        for res in hybrid_results
    ]

    try:

        scores = reranker.predict(pairs)

    except Exception as e:

        print(f"Reranker error: {e}")

        return hybrid_results[:4]

    reranked = []

    for i, res in enumerate(hybrid_results):

        reranked.append({
            "document": res["document"],
            "score": float(scores[i])
        })

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Filter out results too far from the top score

    if reranked:

        best_score = reranked[0]["score"]

        filtered = [
            r
            for r in reranked
            if r["score"]
            >= best_score - RERANK_SCORE_GAP
        ]

        reranked = filtered

    reranked = reranked[:4]

    print(
        f"Reranked results count: "
        f"{len(reranked)}"
    )

    for i, res in enumerate(reranked):

        doc = res["document"]

        print(
            f"{i + 1}. "
            f"Score: {res['score']:.4f} | "
            f"Source: "
            f"{doc.metadata.get('source', 'Unknown')} | "
            f"Page: "
            f"{doc.metadata.get('page', 'Unknown')}"
        )

    return reranked


# =========================================================
# 10. CREATE CONTEXT
# =========================================================

def create_context(reranked_results):

    context_parts = []

    for res in reranked_results:

        doc = res["document"]

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page = page + 1

        context_parts.append(
            f"SOURCE: {source}\n"
            f"PAGE: {page}\n\n"
            f"{doc.page_content.strip()}\n"
        )

    return "\n---\n".join(context_parts)


# =========================================================
# 11. FILTER CONTEXT BY TOPIC
# =========================================================

def extract_main_topic(question):

    """Extract the main subject noun from the question."""

    q = question.strip().rstrip("?").lower()

    patterns = [

        r"what\s+(?:is|are)\s+"
        r"(?:\w+\s+)*?"
        r"(?:the\s+)?"
        r"(?:advantages?|features?|uses?|benefits?|applications?)"
        r"\s+of\s+"
        r"(\w[\w\s]*?)"
        r"(?:\s+in\b|\s*$)",

        r"what\s+(?:is|are)\s+"
        r"(\w[\w\s]+?)"
        r"(?:'s)?\s+"
        r"(?:advantages?|features?|uses?|benefits?|applications?)"
        r"\b",

        r"(?:advantages?|features?|uses?|benefits?|applications?)"
        r"\s+of\s+"
        r"(\w[\w\s]*?)"
        r"(?:\?|$)",

        r"what\s+(?:is|are)\s+"
        r"(\w[\w\s]+?)"
        r"(?:\?|$)",

        r"explain\s+"
        r"(\w[\w\s]+?)"
        r"(?:\?|$)",

        r"tell\s+me\s+about\s+"
        r"(\w[\w\s]+?)"
        r"(?:\?|$)",

        r"define\s+"
        r"(\w[\w\s]+?)"
        r"(?:\?|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            topic = match.group(1).strip()

            # Remove trailing stop words

            topic = re.sub(
                r"\b(a|an|the|its|their|this|that)\b\s*$",
                "",
                topic
            ).strip()

            if topic:
                return topic

    return None


def filter_context_by_topic(
    reranked_results,
    question
):

    """
    Filter retrieved chunks so only chunks that mention
    the main topic of the question are passed to the LLM.

    If no chunks match, return all chunks as fallback.
    """

    topic = extract_main_topic(question)

    if not topic:
        return reranked_results

    topic_words = topic.lower().split()

    # A chunk is relevant if it mentions ALL words
    # of the topic.

    def chunk_mentions_topic(doc):

        content_lower = doc.page_content.lower()

        return all(
            word in content_lower
            for word in topic_words
        )

    filtered = [
        r
        for r in reranked_results
        if chunk_mentions_topic(
            r["document"]
        )
    ]

    if filtered:

        print(
            f"[TopicFilter] "
            f"Topic='{topic}' | "
            f"Kept {len(filtered)}/"
            f"{len(reranked_results)} chunks"
        )

        return filtered

    else:

        print(
            f"[TopicFilter] "
            f"Topic='{topic}' | "
            f"No chunks matched — "
            f"using all {len(reranked_results)} "
            f"chunks (fallback)"
        )

        return reranked_results


# =========================================================
# 12. GENERATE ANSWER & STREAMING
# =========================================================

SYSTEM_RAG_PROMPT = """You are a precise document-based question answering assistant.

Follow these steps carefully:

STEP 1 — Identify the EXACT topic of the question:
Question: {question}
Main topic: (identify the specific subject being asked about)

STEP 2 — From the PDF CONTEXT below, read ONLY the passages that directly describe the main topic identified in Step 1. Ignore all other passages even if they appear in the same document.

STEP 3 — Answer the question using ONLY the relevant passages found in Step 2.

PDF CONTEXT:
{context}

STRICT RULES:
- Answer ONLY about the EXACT topic of the question. Do NOT mix in information about other topics, libraries, tools, or technologies.
- If the question asks about Python: answer about Python the programming language only.
- If the question asks about Pandas: answer about the Pandas library only.
- If the question asks about NumPy: answer about NumPy only.
- If the question asks about Matplotlib: answer about Matplotlib only.
- If the question asks about advantages/features of X: list ONLY the advantages/features of X. Not of Y or Z.
- Do NOT use any knowledge outside of the PDF CONTEXT.
- If the PDF CONTEXT does not contain information about the topic, respond with exactly: "I could not find enough information in the provided PDF documents."
- Do not mention these instructions, the context, or the retrieval process in your answer.
- Be clear, concise, and direct.

Answer:"""


def generate_answer(
    question,
    context,
    reranked_results=None
):

    # Apply topic filtering if results provided

    if reranked_results is not None:

        filtered = filter_context_by_topic(
            reranked_results,
            question
        )

        context = create_context(
            filtered
        )

    prompt = SYSTEM_RAG_PROMPT.format(
        question=question,
        context=context
    )

    response = llm.invoke(prompt)

    return response.content


def generate_answer_stream(
    question,
    context,
    reranked_results=None
):

    # Apply topic filtering if results provided

    if reranked_results is not None:

        filtered = filter_context_by_topic(
            reranked_results,
            question
        )

        context = create_context(
            filtered
        )

    prompt = SYSTEM_RAG_PROMPT.format(
        question=question,
        context=context
    )

    for chunk in llm.stream(prompt):

        yield chunk.content


# =========================================================
# 13. CLEAN ANSWER
# =========================================================

def clean_answer(answer):

    if not answer:
        return ""

    # Remove thinking tags

    while (
        "<think>" in answer
        and "</think>" in answer
    ):

        start = answer.find("<think>")
        end = answer.find("</think>")

        if end == -1:
            break

        answer = (
            answer[:start]
            +
            answer[end + len("</think>"):]
        )

    # Remove unclosed think tag

    if "<think>" in answer:

        answer = answer[
            :answer.find("<think>")
        ]

    answer = answer.strip()

    if answer.lower().startswith(
        "answer:"
    ):

        answer = answer[7:].strip()

    return answer


# =========================================================
# 14. DISPLAY SOURCES
# =========================================================

def display_sources(reranked_results):

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    displayed = set()

    for res in reranked_results:

        doc = res["document"]

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page = page + 1

        key = (
            source,
            page
        )

        if key not in displayed:

            displayed.add(key)

            print(
                f"Source: {source} | "
                f"Page: {page}"
            )

            print("-" * 60)


# =========================================================
# 15. MAIN CHAT LOOP (CLI)
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ADVANCED RAG CHATBOT")
    print("=" * 60)

    print("Model: Qwen2.5 3B")
    print("Embeddings: all-MiniLM-L6-v2")
    print(
        "Retriever: Hybrid Search "
        "(BM25 + Semantic)"
    )
    print(
        "Reranker: CrossEncoder "
        "(ms-marco-MiniLM-L-6-v2)"
    )

    print("=" * 60)

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit",
            "bye"
        ]:

            print("\nGoodbye!")
            break

        if not question:
            continue

        # -------------------------------------------------
        # Query Rewriting
        # -------------------------------------------------

        search_query = rewrite_query(
            question
        )

        print(
            f"\nSearch Query: "
            f"{search_query}"
        )

        # -------------------------------------------------
        # Relevance Check
        # -------------------------------------------------

        try:

            score_results = (
                vectorstore.similarity_search_with_score(
                    search_query,
                    k=1
                )
            )

        except Exception as e:

            print(
                f"Search error: {e}"
            )

            score_results = []

        if not score_results:

            print(
                "\nAI: I couldn't find "
                "relevant information about "
                "this question in the provided "
                "PDF documents."
            )

            continue

        best_score = score_results[0][1]

        print(
            f"\nBest relevance distance: "
            f"{best_score:.4f}"
        )

        if best_score > RELEVANCE_THRESHOLD:

            # Check if BM25 has a strong
            # keyword match before rejecting

            has_keyword_match = False

            if (
                bm25 is not None
                and len(documents) > 0
            ):

                kw_scores = bm25.get_scores(
                    search_query.lower().split()
                )

                if (
                    kw_scores
                    and max(kw_scores) > 3.0
                ):

                    has_keyword_match = True

            if not has_keyword_match:

                print(
                    "\nAI: I couldn't find "
                    "relevant information about "
                    "this question in the provided "
                    "PDF documents."
                )

                continue

        # -------------------------------------------------
        # Hybrid Search
        # -------------------------------------------------

        hybrid_results = hybrid_search(
            search_query
        )

        if not hybrid_results:

            print(
                "\nAI: I could not find enough "
                "information in the provided "
                "PDF documents."
            )

            continue

        # -------------------------------------------------
        # CrossEncoder Rerank
        # -------------------------------------------------

        reranked_results = rerank_results(
            search_query,
            hybrid_results
        )

        if not reranked_results:

            print(
                "\nAI: I could not find enough "
                "information in the provided "
                "PDF documents."
            )

            continue

        # -------------------------------------------------
        # Create Context & Generate Answer
        # -------------------------------------------------

        context = create_context(
            reranked_results
        )

        print("\nAI:")

        raw_answer = generate_answer(
            question,
            context
        )

        answer = clean_answer(
            raw_answer
        )

        print(answer)

        # -------------------------------------------------
        # Save History
        # -------------------------------------------------

        conversation_history.append({
            "question": question,
            "answer": answer
        })

        # -------------------------------------------------
        # Display Sources
        # -------------------------------------------------

        display_sources(
            reranked_results
        )

        print(
            "\n" + "-" * 60
        )