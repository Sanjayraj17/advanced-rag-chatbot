from ingestion import vectorstore

query = "What is Python?"

print(f"Testing retrieval with query: '{query}'")
results = vectorstore.similarity_search(
    query,
    k=3
)

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

if not results:
    print("No results found in ChromaDB.")
else:
    for i, document in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source:", document.metadata.get("source", "Unknown"))
        page = document.metadata.get("page", None)
        display_page = page + 1 if isinstance(page, int) else "N/A"
        print("Page:", display_page)
        print("\nContent:")
        print(document.page_content[:400])
        print("-" * 60)