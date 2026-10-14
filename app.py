import streamlit as st
import os
import json
from datetime import datetime
import rag_service as rag

add_pdf = rag.add_pdf
delete_pdf = rag.delete_pdf
rebuild_bm25_index = rag.rebuild_bm25_index
rewrite_query = rag.rewrite_query
hybrid_search = rag.hybrid_search
rerank_results = rag.rerank_results
create_context = rag.create_context
generate_answer_stream = rag.generate_answer_stream
clean_answer = rag.clean_answer
conversation_history = rag.service.conversation_history


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
        color: #1E3A8A;
    }

    .subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 20px;
    }

    .source-box {
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 6px;
        background-color: rgba(241, 245, 249, 0.6);
        font-size: 13px;
    }

    .export-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px dashed #CBD5E1;
        background-color: #F8FAFC;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "saved_conversations" not in st.session_state:
    st.session_state.saved_conversations = {}

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = "chat_1"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# ============================================================
# CREATE FIRST CONVERSATION
# ============================================================

if st.session_state.active_conversation_id not in st.session_state.saved_conversations:
    st.session_state.saved_conversations[st.session_state.active_conversation_id] = {
        "title": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }


# ============================================================
# HELPER FUNCTIONS — CONVERSATION & EXPORT
# ============================================================

def create_new_conversation():
    conversations = st.session_state.saved_conversations
    number = len(conversations) + 1
    conversation_id = f"chat_{number}"

    while conversation_id in conversations:
        number += 1
        conversation_id = f"chat_{number}"

    conversations[conversation_id] = {
        "title": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }

    st.session_state.active_conversation_id = conversation_id
    conversation_history.clear()


def export_chat_markdown(conversation: dict) -> str:
    title = conversation.get("title", "Chat Transcript")
    created_at = conversation.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines = [
        f"# {title}",
        f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Created: {created_at})*",
        "",
        "---",
        ""
    ]
    for idx, msg in enumerate(conversation.get("messages", []), start=1):
        lines.append(f"### Q{idx}: {msg['question']}")
        lines.append("")
        lines.append(msg["answer"])
        lines.append("")
        if msg.get("sources"):
            lines.append("**Sources:**")
            for s in msg["sources"]:
                lines.append(f"- 📄 {s}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def export_chat_json(conversation: dict) -> str:
    return json.dumps(conversation, indent=2, ensure_ascii=False)


def export_chat_text(conversation: dict) -> str:
    title = conversation.get("title", "Chat Transcript")
    lines = [f"=== {title.upper()} ===", ""]
    for idx, msg in enumerate(conversation.get("messages", []), start=1):
        lines.append(f"Q{idx}: {msg['question']}")
        lines.append(f"A{idx}: {msg['answer']}")
        if msg.get("sources"):
            lines.append("Sources: " + ", ".join(msg["sources"]))
        lines.append("-" * 50)
    return "\n".join(lines)


def export_all_chats_json(conversations: dict) -> str:
    return json.dumps(conversations, indent=2, ensure_ascii=False)


# ============================================================
# GET ACTIVE CONVERSATION
# ============================================================

active_conversation = st.session_state.saved_conversations[
    st.session_state.active_conversation_id
]


# ============================================================
# SYNC BACKEND CONVERSATION HISTORY
# ============================================================

conversation_history.clear()
for message in active_conversation["messages"]:
    conversation_history.append(
        {
            "question": message["question"],
            "answer": message["answer"]
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ RAG System")

    # --------------------------------------------------------
    # CONVERSATION MANAGER
    # --------------------------------------------------------
    st.subheader("💬 Conversations")

    if st.button("🆕 New Chat", use_container_width=True):
        create_new_conversation()
        st.rerun()

    conversations = st.session_state.saved_conversations
    if conversations:
        for conversation_id, conversation in conversations.items():
            title = conversation["title"]
            message_count = len(conversation["messages"])
            icon = "🟢" if conversation_id == st.session_state.active_conversation_id else "💬"

            if st.button(
                f"{icon} {title}",
                key=f"select_{conversation_id}",
                use_container_width=True
            ):
                st.session_state.active_conversation_id = conversation_id
                conversation_history.clear()
                for message in conversation["messages"]:
                    conversation_history.append(
                        {
                            "question": message["question"],
                            "answer": message["answer"]
                        }
                    )
                st.rerun()

            st.caption(f"{message_count} message(s)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if len(conversations) > 1:
            if st.button("🗑️ Delete Chat", use_container_width=True):
                current_id = st.session_state.active_conversation_id
                del conversations[current_id]
                remaining_ids = list(conversations.keys())
                new_active_id = remaining_ids[0]
                st.session_state.active_conversation_id = new_active_id
                conversation_history.clear()
                for message in conversations[new_active_id]["messages"]:
                    conversation_history.append(
                        {
                            "question": message["question"],
                            "answer": message["answer"]
                        }
                    )
                st.rerun()
    with col_btn2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            current_id = st.session_state.active_conversation_id
            conversations[current_id]["messages"] = []
            conversations[current_id]["title"] = "New Chat"
            conversation_history.clear()
            st.rerun()

    # --------------------------------------------------------
    # EXPORT & SHARE CHAT
    # --------------------------------------------------------
    st.divider()
    st.subheader("📤 Export & Share")

    has_messages = len(active_conversation["messages"]) > 0

    if has_messages:
        md_data = export_chat_markdown(active_conversation)
        json_data = export_chat_json(active_conversation)
        txt_data = export_chat_text(active_conversation)

        safe_title = "".join(c for c in active_conversation["title"] if c.isalnum() or c in (' ', '_', '-')).strip()
        filename_base = f"chat_{safe_title or 'transcript'}"

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "📝 Markdown",
                data=md_data,
                file_name=f"{filename_base}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_exp2:
            st.download_button(
                "📊 JSON",
                data=json_data,
                file_name=f"{filename_base}.json",
                mime="application/json",
                use_container_width=True
            )

        st.download_button(
            "📄 Plain Text (.txt)",
            data=txt_data,
            file_name=f"{filename_base}.txt",
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("📋 Share Transcript"):
            st.text_area(
                "Copy full transcript:",
                value=md_data,
                height=180,
                key="share_transcript_area"
            )
    else:
        st.info("Start chatting to enable Export and Share options.")

    if len(conversations) > 0:
        all_chats_json = export_all_chats_json(conversations)
        st.download_button(
            "💾 Backup All Chats (.json)",
            data=all_chats_json,
            file_name=f"all_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------
    st.divider()
    st.subheader("📚 Knowledge Base")

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    pdf_files = sorted([
        file for file in os.listdir(data_folder)
        if file.lower().endswith(".pdf")
    ])

    if pdf_files:
        for pdf_file in pdf_files:
            file_size_kb = os.path.getsize(os.path.join(data_folder, pdf_file)) / 1024
            st.write(f"📄 **{pdf_file}** ({file_size_kb:.0f} KB)")
    else:
        st.write("No PDF documents found in data folder.")

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------
    st.divider()
    st.subheader("📊 System Status")

    rag.service.ensure_runtime()
    indexed_chunks = len(rag.service.documents)

    st.metric("📚 PDF Documents", len(pdf_files))
    st.metric("📦 Indexed Chunks", indexed_chunks)
    st.success("🟢 RAG Engine Online (ChromaDB + BM25 + Qwen2.5 3B)")

    # --------------------------------------------------------
    # ADD NEW PDF
    # --------------------------------------------------------
    st.divider()
    st.subheader("➕ Add New PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        file_path = os.path.join(data_folder, uploaded_file.name)
        st.caption(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("📥 Index Uploaded PDF", use_container_width=True):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Indexing PDF into ChromaDB and building BM25..."):
                try:
                    chunks_added = add_pdf(file_path)
                    rebuild_bm25_index()
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.session_state.uploader_key += 1
                    st.success(f"Successfully indexed {chunks_added} chunks from {uploaded_file.name}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while indexing PDF: {e}")

    # --------------------------------------------------------
    # DELETE PDF
    # --------------------------------------------------------
    st.divider()
    st.subheader("🗑️ Delete PDF")

    if pdf_files:
        pdf_to_delete = st.selectbox(
            "Select a PDF to delete",
            pdf_files,
            key="delete_select"
        )

        if st.button("🗑️ Delete Selected PDF", use_container_width=True):
            try:
                with st.spinner("Removing PDF chunks from ChromaDB and BM25..."):
                    chunks_removed = delete_pdf(pdf_to_delete)
                    rebuild_bm25_index()

                file_path = os.path.join(data_folder, pdf_to_delete)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass

                st.session_state.processed_files.discard(pdf_to_delete)
                st.session_state.uploader_key += 1
                st.success(f"Removed {pdf_to_delete} ({chunks_removed} chunks deleted)!")
                st.rerun()
            except Exception as e:
                st.error(f"Error while deleting PDF: {e}")
    else:
        st.info("No PDFs available to delete.")


# ============================================================
# MAIN CONTENT AREA
# ============================================================

st.markdown('<div class="main-title">🤖 Advanced RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Hybrid Search (BM25 + Dense Embeddings) · CrossEncoder Reranker · Qwen2.5 3B'
    '</div>',
    unsafe_allow_html=True
)

current_chat_title = active_conversation.get("title", "New Chat")
st.subheader(f"💬 {current_chat_title}")


# ============================================================
# DISPLAY CONVERSATION MESSAGES
# ============================================================

for message in active_conversation["messages"]:
    with st.chat_message("user", avatar="👤"):
        st.write(message["question"])

    with st.chat_message("assistant", avatar="🤖"):
        st.write(message["answer"])
        if message.get("sources"):
            with st.expander("📚 Sources Cited", expanded=False):
                for s in message["sources"]:
                    st.markdown(f"- 📄 `{s}`")


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input("Ask a question about your PDF documents...")


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    # Update conversation title from first question
    if len(active_conversation["messages"]) == 0 and active_conversation["title"] == "New Chat":
        if len(question) > 35:
            active_conversation["title"] = question[:35] + "..."
        else:
            active_conversation["title"] = question

    # Render User Message
    with st.chat_message("user", avatar="👤"):
        st.write(question)

    # Render Assistant Message
    with st.chat_message("assistant", avatar="🤖"):
        try:
            # -------------------------------------------------
            # STEP 1 — CONVERSATIONAL QUERY REWRITING
            # -------------------------------------------------
            with st.status("🧠 Understanding question & resolving context...", expanded=False) as status:
                retrieval_query = rewrite_query(question)
                status.update(label=f"🧠 Search Query: {retrieval_query}", state="complete")

            # -------------------------------------------------
            # STEP 2 — RELEVANCE CHECK
            # -------------------------------------------------
            with st.status("🔍 Checking document relevance...", expanded=False) as status:
                try:
                    best_results = rag.vectorstore.similarity_search_with_score(
                        retrieval_query,
                        k=1
                    )
                except Exception:
                    best_results = []

                is_relevant = bool(best_results)

                if best_results:
                    best_score = best_results[0][1]
                    if best_score > rag.RELEVANCE_THRESHOLD:
                        is_relevant = False

                # Keyword check fallback via BM25
                if not is_relevant and rag.service.bm25 is not None and rag.service.documents:
                    kw_scores = rag.service.bm25.get_scores(retrieval_query.lower().split())
                    if len(kw_scores) > 0 and max(kw_scores) > 3.0:
                        is_relevant = True

                if is_relevant:
                    status.update(label="🔍 Relevant documents found in knowledge base", state="complete")
                else:
                    status.update(label="🔍 No relevant documents found", state="error")

            if not is_relevant:
                final_answer = "I could not find enough information in the provided PDF documents."
                st.write(final_answer)

                msg = {
                    "question": question,
                    "answer": final_answer,
                    "sources": []
                }
                active_conversation["messages"].append(msg)
                conversation_history.append({"question": question, "answer": final_answer})
                st.stop()

            # -------------------------------------------------
            # STEP 3 — HYBRID RETRIEVAL (SEMANTIC + BM25)
            # -------------------------------------------------
            with st.status("🔎 Searching knowledge base (Hybrid Search)...", expanded=False) as status:
                hybrid_results = hybrid_search(retrieval_query)
                status.update(
                    label=f"🔎 Retrieved {len(hybrid_results)} candidates via Hybrid Search",
                    state="complete"
                )

            if not hybrid_results:
                final_answer = "I could not find enough information in the provided PDF documents."
                st.write(final_answer)

                msg = {
                    "question": question,
                    "answer": final_answer,
                    "sources": []
                }
                active_conversation["messages"].append(msg)
                conversation_history.append({"question": question, "answer": final_answer})
                st.stop()

            # -------------------------------------------------
            # STEP 4 — CROSSENCODER RERANKING
            # -------------------------------------------------
            with st.status("🎯 Reranking retrieved passages...", expanded=False) as status:
                reranked_results = rerank_results(retrieval_query, hybrid_results)
                status.update(
                    label=f"🎯 Reranked top {len(reranked_results)} most relevant passages",
                    state="complete"
                )

            if not reranked_results:
                final_answer = "I could not find enough information in the provided PDF documents."
                st.write(final_answer)

                msg = {
                    "question": question,
                    "answer": final_answer,
                    "sources": []
                }
                active_conversation["messages"].append(msg)
                conversation_history.append({"question": question, "answer": final_answer})
                st.stop()

            # -------------------------------------------------
            # STEP 5 — CREATE CONTEXT
            # -------------------------------------------------
            context = create_context(reranked_results)

            with st.expander("🔍 View Retrieval & Context Details", expanded=False):
                st.write("**Rewritten Search Query:**", retrieval_query)
                st.write("### Reranked Passages:")
                for i, r in enumerate(reranked_results, start=1):
                    doc = r["document"]
                    src = doc.metadata.get("source", "Unknown")
                    p = doc.metadata.get("page_number", doc.metadata.get("page", 0))
                    st.write(f"**#{i}** | `{src}` (p.{p}) | Score: `{r['score']:.4f}`")
                    st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    st.divider()

            # -------------------------------------------------
            # STEP 6 — STREAM LLM ANSWER (QWEN2.5 3B)
            # -------------------------------------------------
            answer_placeholder = st.empty()
            streamed_answer = ""

            with st.spinner("🤖 Qwen2.5 3B is generating the answer..."):
                for chunk in generate_answer_stream(
                    question,
                    context,
                    reranked_results
                ):
                    streamed_answer += chunk
                    answer_placeholder.markdown(streamed_answer)

            # -------------------------------------------------
            # STEP 7 — CLEAN ANSWER
            # -------------------------------------------------
            final_answer = clean_answer(streamed_answer)
            answer_placeholder.markdown(final_answer)

            # -------------------------------------------------
            # STEP 8 — DISPLAY SOURCES
            # -------------------------------------------------
            st.write("")
            st.write("### 📚 Sources")

            displayed_sources = set()
            sources_summary = []

            for result in reranked_results:
                document = result["document"]
                source = document.metadata.get("source", "Unknown")
                page = document.metadata.get("page_number", document.metadata.get("page", 0))

                source_key = (source, page)
                if source_key not in displayed_sources:
                    displayed_sources.add(source_key)
                    source_str = f"{source} — Page {page}"
                    sources_summary.append(source_str)

                    st.markdown(
                        f"""
                        <div class="source-box">
                            📄 <strong>{source}</strong> — Page {page}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # -------------------------------------------------
            # STEP 9 — SAVE CONVERSATION WITH SOURCES
            # -------------------------------------------------
            new_message = {
                "question": question,
                "answer": final_answer,
                "sources": sources_summary
            }

            active_conversation["messages"].append(new_message)
            conversation_history.append({
                "question": question,
                "answer": final_answer
            })

        except Exception as e:
            st.error("An error occurred while processing your question.")
            st.exception(e)