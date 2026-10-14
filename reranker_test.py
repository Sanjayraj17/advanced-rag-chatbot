from sentence_transformers import CrossEncoder


# =========================================================
# 1. LOAD RERANKER MODEL
# =========================================================

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker model loaded.")


# =========================================================
# 2. TEST QUESTION
# =========================================================

question = "What is Python?"


# =========================================================
# 3. TEST RETRIEVED DOCUMENTS
# =========================================================

documents = [

    "Python is an easy to learn, powerful programming language.",

    "Python Software Foundation copyright information.",

    "Python licenses and acknowledgements.",

    "Python enables programs to be written compactly and readably.",

    "The Python community includes many volunteers and contributors."

]


# =========================================================
# 4. CREATE QUESTION-DOCUMENT PAIRS
# =========================================================

pairs = []

for document in documents:

    pairs.append(
        [
            question,
            document
        ]
    )


# =========================================================
# 5. CALCULATE RERANKING SCORES
# =========================================================

print("\nCalculating reranking scores...")

scores = reranker.predict(
    pairs
)


# =========================================================
# 6. COMBINE DOCUMENTS AND SCORES
# =========================================================

results = []

for i in range(len(documents)):

    results.append(
        {
            "document": documents[i],
            "score": float(scores[i])
        }
    )


# =========================================================
# 7. SORT BY SCORE
# =========================================================

results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# =========================================================
# 8. DISPLAY RESULTS
# =========================================================

print("\nReranked results:")

for i, result in enumerate(results):

    print(
        f"\nResult {i + 1}"
    )

    print(
        "Score:",
        round(result["score"], 4)
    )

    print(
        "Text:",
        result["document"]
    )

    print(
        "-" * 60
    )