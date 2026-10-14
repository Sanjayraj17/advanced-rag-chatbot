import json
import traceback
from pathlib import Path

import rag_service as rag

output_path = Path(__file__).with_name("verification_results.json")
error_path = Path(__file__).with_name("verification_error.txt")
try:
    rag.service.ensure_runtime()
    question = "What is Python release 3.7.0 about?"
    results = rag.hybrid_search(question)
    semantic = rag.service.similarity_search_with_score(question, k=3)
    output = {
        "collection": rag.service.collection.name,
        "document_count": len(rag.service.documents),
        "bm25_ready": rag.service.bm25 is not None,
        "hybrid_count": len(results),
        "top_sources": [
            {
                "source": item["document"].metadata.get("source"),
                "page": item["document"].metadata.get("page_number"),
                "score": round(item["score"], 4),
            }
            for item in results[:5]
        ],
        "semantic_count": len(semantic),
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
except Exception:
    error_path.write_text(traceback.format_exc(), encoding="utf-8")
    raise
