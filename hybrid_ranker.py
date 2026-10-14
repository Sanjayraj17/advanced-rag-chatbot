import os
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from ingestion import embeddings, vectorstore

# =========================================================
# 1. LOAD DOCUMENTS
# =========================================================

print("\nLoading documents...")

data = vectorstore.get()
text_list = data.get("documents") or []
meta_list = data.get("metadatas") or []

documents = []
for i, text in enumerate(text_list):
    meta = meta_list[i] if i < len(meta_list) and meta_list[i] else {}
    documents.append(
        Document(
            page_content=text,
            metadata=meta
        )
    )

print(f"Total documents: {len(documents)}")

# =========================================================
# 2. CREATE BM25 INDEX
# =========================================================

print("\nCreating BM25 index...")

if documents:
    tokenized_documents = [
        doc.page_content.lower().split()
        for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_documents)
    print("BM25 index created.")
else:
    bm25 = None
    print("No documents available.")

# =========================================================
# 3. NORMALIZE SCORES
# =========================================================

def normalize_scores(scores):
    if scores is None or len(scores) == 0:
        return []
    minimum = float(min(scores))
    maximum = float(max(scores))

    if maximum == minimum:
        return [1.0 for _ in range(len(scores))]

    return [float(score - minimum) / float(maximum - minimum) for score in scores]

# =========================================================
# 4. HYBRID SEARCH
# =========================================================

def hybrid_search(question):
    if not documents:
        return []

    # Semantic search
    semantic_results = vectorstore.similarity_search_with_score(question, k=10)
    semantic_distances = [res[1] for res in semantic_results]
    semantic_similarities = [1.0 / (1.0 + d) for d in semantic_distances]
    normalized_semantic = normalize_scores(semantic_similarities)

    results = {}
    for i, (doc, _) in enumerate(semantic_results):
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content.strip()
        )
        results[key] = {
            "document": doc,
            "semantic_score": normalized_semantic[i],
            "keyword_score": 0.0
        }

    # BM25 search
    if bm25 is not None:
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        normalized_bm25 = normalize_scores(bm25_scores)

        for index, score in enumerate(normalized_bm25):
            doc = documents[index]
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content.strip()
            )
            if key in results:
                results[key]["keyword_score"] = score
            elif score > 0.3:
                results[key] = {
                    "document": doc,
                    "semantic_score": 0.0,
                    "keyword_score": score
                }

    for item in results.values():
        item["hybrid_score"] = (
            0.6 * item["semantic_score"] + 0.4 * item["keyword_score"]
        )

    ranked_results = sorted(
        results.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return ranked_results[:8]

# =========================================================
# 5. INTERACTIVE TEST LOOP
# =========================================================

if __name__ == "__main__":
    while True:
        question = input("\nEnter search query (or 'exit'): ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nHybrid ranker stopped.")
            break

        if not question:
            continue

        print("\nRunning hybrid search...")
        results = hybrid_search(question)
        print(f"\nTotal ranked results: {len(results)}")

        for i, item in enumerate(results, start=1):
            doc = item["document"]
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            display_page = page + 1 if isinstance(page, int) else "N/A"

            print(f"\nResult {i}")
            print(f"Hybrid score: {item['hybrid_score']:.4f}")
            print(f"Semantic score: {item['semantic_score']:.4f}")
            print(f"Keyword score: {item['keyword_score']:.4f}")
            print(f"Source: {source}")
            print(f"Page: {display_page}")
            print("Content:", doc.page_content[:300])
            print("-" * 60)