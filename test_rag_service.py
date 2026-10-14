import rag_service
import chromadb


def test_rewrite_query_resolves_followup_subject():
    svc = rag_service.service
    svc.conversation_history = [{"question": "what is python", "answer": "Python is a language."}]
    rewritten = svc.rewrite_query("what are its advantages")
    assert rewritten.lower() == "what are python advantages"


def test_hybrid_search_returns_pandas_relevant_passages():
    svc = rag_service.service
    results = svc.hybrid_search("what are pandas advantages")
    assert results
    assert any("pandas" in r["document"].page_content.lower() for r in results)


def test_followup_generation_uses_rewritten_subject():
    svc = rag_service.service
    svc.conversation_history = [{"question": "what is python", "answer": "Python is a language."}]
    question = "what are its advantages"
    rewritten = svc.rewrite_query(question)
    results = svc.hybrid_search(rewritten)
    filtered = svc.filter_context_by_topic(results, question)
    context = svc.create_context(filtered[:4])
    answer = rag_service.generate_answer(question, context, filtered[:4])
    text = answer.lower()
    assert "python" in text
    assert "advantage" in text


def test_service_avoids_legacy_langchain_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_service, "CHROMA_PATH", tmp_path)
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.get_or_create_collection(name="langchain")

    svc = rag_service.RAGService()
    svc._initialize()

    assert svc.collection.name == "advanced_rag_documents"
