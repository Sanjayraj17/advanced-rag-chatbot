from ingestion import vectorstore

# =========================================================
# TEST MULTI-RETRIEVAL WITH MMR
# =========================================================

print("\nConnecting to multi-PDF ChromaDB...")

if __name__ == "__main__":
    while True:
        question = input("\nEnter your question (or 'exit'): ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nTest stopped.")
            break

        if not question:
            continue

        print("\nSearching using MMR...")
        results = vectorstore.max_marginal_relevance_search(
            question,
            k=6,
            fetch_k=20,
            lambda_mult=0.7
        )

        print(f"\nRetrieved results ({len(results)}):")
        for i, result in enumerate(results, start=1):
            source = result.metadata.get("source", "Unknown")
            page = result.metadata.get("page", None)
            display_page = page + 1 if isinstance(page, int) else "N/A"

            print("\n" + "=" * 60)
            print(f"Result {i} | Source: {source} | Page: {display_page}")
            print("=" * 60)
            print(result.page_content[:400])