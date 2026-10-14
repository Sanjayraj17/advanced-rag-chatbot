import os
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from ingestion import embeddings, vectorstore

# =========================================================
# 1. LOAD DOCUMENTS FOR BM25
# =========================================================

print("\nLoading documents for BM25...")

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
# 2. CREATE BM25 KEYWORD INDEX
# =========================================================

print("\nCreating BM25 keyword index...")

if documents:
    tokenized_documents = [
        doc.page_content.lower().split()
        for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_documents)
    print("BM25 index created.")
else:
    bm25 = None
    print("No documents found for BM25.")

# =========================================================
# 3. HYBRID SEARCH FUNCTION
# =========================================================

def hybrid_search(question):
    if not documents:
        return []

    # Semantic MMR search
    semantic_results = vectorstore.max_marginal_relevance_search(
        question,
        k=6,
        fetch_k=20,
        lambda_mult=0.7
    )

    # BM25 Keyword Search
    combined_results = []
    seen_text = set()

    for doc in semantic_results:
        text = doc.page_content.strip()
        if text not in seen_text:
            combined_results.append(doc)
            seen_text.add(text)

    if bm25 is not None:
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:6]

        for idx in top_indices:
            doc = documents[idx]
            text = doc.page_content.strip()
            if text not in seen_text:
                combined_results.append(doc)
                seen_text.add(text)

    return combined_results[:8]

# =========================================================
# 4. INTERACTIVE TEST LOOP
# =========================================================

if __name__ == "__main__":
    while True:
        question = input("\nEnter search query (or 'exit'): ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nHybrid search stopped.")
            break

        if not question:
            continue

        print("\nRunning hybrid search...")
        results = hybrid_search(question)
        print(f"\nTotal combined results: {len(results)}")

        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            display_page = page + 1 if isinstance(page, int) else "N/A"

            print(f"\nResult {i}")
            print(f"Source: {source}")
            print(f"Page: {display_page}")
            print("Content:", doc.page_content[:300])
            print("-" * 60)