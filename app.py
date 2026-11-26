import csv
import io
import time
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

import assistant

st.set_page_config(
    page_title="AI Assistant with Model Benchmarking",
    page_icon="🤖",
    layout="wide"
)

MODELS = assistant.DEFAULT_MODELS

# ---------------------------
# Session state
# ---------------------------
DEFAULT_STATE = {
    "conversations": [],
    "active_chat_id": None,
    "documents": [],
    "benchmark_results": None,
    "benchmark_question": "",
    "benchmark_csv": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or __import__("os").environ.get("GROQ_API_KEY", "")


def get_chat(chat_id):
    return next((c for c in st.session_state.conversations if c["id"] == chat_id), None)


def create_chat():
    chat = {
        "id": "chat_" + uuid.uuid4().hex[:10],
        "title": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": []
    }
    st.session_state.conversations.append(chat)
    st.session_state.active_chat_id = chat["id"]
    return chat


def open_chat(chat_id):
    if get_chat(chat_id):
        st.session_state.active_chat_id = chat_id


def delete_chat(chat_id):
    st.session_state.conversations[:] = [
        c for c in st.session_state.conversations if c["id"] != chat_id
    ]
    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = None


def add_message(chat, role, content):
    chat["messages"].append({"role": role, "content": content})
    if role == "user" and len([m for m in chat["messages"] if m["role"] == "user"]) == 1:
        title = content.strip().replace("\n", " ")
        chat["title"] = title[:45] + ("..." if len(title) > 45 else "")


def save_benchmark_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Model",
        "Average Response Time (seconds)",
        "Peak RAM (MB)",
        "Quality Rating"
    ])
    for result in results:
        if not result.get("failed"):
            # Cloud provider RAM is not exposed by the API. Keep the original
            # CSV column for compatibility, but leave it blank rather than inventing a value.
            writer.writerow([
                result["model"],
                round(result["response_time"], 2),
                "",
                result["quality"]
            ])
    return output.getvalue().encode("utf-8-sig")


def run_model_benchmark(question, models, api_key):
    results = []
    for model_name in models:
        start = time.perf_counter()
        try:
            answer = assistant.generate_answer(
                question,
                model_name,
                api_key,
                history=[]
            )
            elapsed = time.perf_counter() - start
            results.append({
                "model": model_name,
                "answer": answer,
                "response_time": elapsed,
                "quality": 3,
                "failed": False
            })
        except Exception as error:
            elapsed = time.perf_counter() - start
            results.append({
                "model": model_name,
                "answer": f"Error: {error}",
                "response_time": elapsed,
                "quality": 0,
                "failed": True
            })
    return results


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🤖 MY AI ASSISTANT")
st.sidebar.caption("AI Chat | Document Learning | MCQ Practice")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["AI Assistant", "Document Manager", "Chat History", "Benchmark Dashboard"]
)

st.sidebar.divider()
model = st.sidebar.selectbox("Choose AI Model", MODELS)
st.sidebar.caption("Cloud inference powered by Groq")
st.sidebar.caption("This public demo is not the offline Ollama version.")

st.sidebar.divider()
st.sidebar.subheader("💬 Your Conversations")
if st.session_state.conversations:
    ordered = list(reversed(st.session_state.conversations))
    chat_ids = [c["id"] for c in ordered]
    titles = {c["id"]: c.get("title", "Untitled Chat") for c in ordered}
    selected_chat = st.sidebar.selectbox(
        "Select a conversation",
        options=chat_ids,
        format_func=lambda chat_id: titles[chat_id],
        key="sidebar_chat_choice"
    )
    if st.sidebar.button("Open Selected Chat", use_container_width=True):
        open_chat(selected_chat)
        st.session_state.page_notice = "Conversation selected."
        st.rerun()
else:
    st.sidebar.caption("No conversations in this browser session yet.")

st.sidebar.divider()
st.sidebar.caption("Built with Python, Streamlit, Groq and TF-IDF document retrieval")

api_key = get_api_key()
if not api_key:
    st.sidebar.error("Groq API key is not configured.")
    st.sidebar.caption("Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets.")


# ---------------------------
# AI Assistant
# ---------------------------
if page == "AI Assistant":
    st.title("🤖 My AI Assistant")
    st.write("Chat with AI, learn from your documents, create study notes and practise MCQs.")
    st.info(
        "Cloud edition: requests are processed by Groq. "
        "The original local Ollama project remains separate."
    )

    if st.session_state.get("page_notice"):
        st.info(st.session_state.pop("page_notice"))

    col_new, col_space = st.columns([1, 5])
    with col_new:
        if st.button("＋ New Chat"):
            create_chat()
            st.rerun()

    if not st.session_state.active_chat_id or not get_chat(st.session_state.active_chat_id):
        create_chat()

    active_chat = get_chat(st.session_state.active_chat_id)

    st.caption(f"Current conversation: {active_chat['title']}")
    st.divider()

    mode = st.selectbox(
        "Choose what you want to do:",
        ["Normal Chat", "Ask PDF", "Summarize Notes", "Generate MCQs"]
    )
    st.caption(
        "Document features support PDF, DOCX, TXT, MD, PPTX, PNG, JPG and JPEG uploads. "
        "Images are processed using OCR."
    )
    st.divider()

    if mode == "Normal Chat":
        st.subheader("⚖️ Compare Both AI Models")
        st.write("Both hosted models receive the same question. Compare their answers and speed.")

        benchmark_question = st.text_area(
            "Enter your benchmark question:",
            key="benchmark_question_input",
            placeholder="Example: Explain Artificial Intelligence in simple words."
        )
        if st.button(
            "🚀 Benchmark Both Models",
            type="primary",
            disabled=not benchmark_question.strip() or not api_key
        ):
            st.session_state.benchmark_question = benchmark_question.strip()
            st.session_state.benchmark_results = None
            st.session_state.benchmark_csv = None

            progress = st.progress(0)
            status = st.empty()
            results = []
            for index, model_name in enumerate(MODELS):
                status.info(f"Benchmarking {model_name}...")
                results.extend(run_model_benchmark(
                    benchmark_question.strip(), [model_name], api_key
                ))
                progress.progress((index + 1) / len(MODELS))

            st.session_state.benchmark_results = results
            status.success("Benchmarking finished.")

        if st.session_state.benchmark_results:
            results = st.session_state.benchmark_results
            st.divider()
            st.subheader("📊 Benchmark Comparison")
            st.caption("Question: " + st.session_state.benchmark_question)
            successful = [r for r in results if not r.get("failed")]

            for result in results:
                with st.expander(f"🤖 {result['model']} — Model Answer", expanded=True):
                    if result.get("failed"):
                        st.error(result["answer"])
                    else:
                        st.markdown(result["answer"])
                        st.metric("Response Time", f"{result['response_time']:.2f} seconds")
                        result["quality"] = st.slider(
                            f"Rate answer quality (1 = Poor, 5 = Excellent) — {result['model']}",
                            min_value=1,
                            max_value=5,
                            value=int(result["quality"]),
                            key=f"quality_{result['model']}"
                        )

            if successful:
                st.subheader("Performance Summary")
                comparison = pd.DataFrame([
                    {
                        "Model": r["model"],
                        "Response Time (seconds)": round(r["response_time"], 2),
                        "Server RAM": "Not exposed by API",
                        "Quality Rating": r["quality"]
                    }
                    for r in successful
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("⭐ Selected Model's Answer")
                selected_result = next((r for r in successful if r["model"] == model), None)
                if selected_result:
                    st.success(f"Showing the previously generated answer from {model}.")
                    st.markdown(selected_result["answer"])
                    st.caption("Changing the sidebar model does not run the benchmark again.")
                else:
                    st.info("The selected model did not complete this benchmark.")

                if st.button("💾 Prepare Benchmark CSV"):
                    st.session_state.benchmark_csv = save_benchmark_csv(successful)
                if st.session_state.benchmark_csv:
                    st.download_button(
                        "⬇️ Download Benchmark Results",
                        data=st.session_state.benchmark_csv,
                        file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.caption("Peak RAM is blank because Groq does not expose server RAM to this app.")

        st.divider()

    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Type your question here...", disabled=not api_key)
    if question:
        previous_history = active_chat["messages"].copy()
        add_message(active_chat, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    if mode == "Normal Chat":
                        answer = assistant.normal_chat(
                            question, model, api_key, previous_history,
                            st.session_state.conversations, active_chat["id"]
                        )
                    elif mode == "Ask PDF":
                        answer = assistant.answer_from_pdf(
                            question, model, api_key, st.session_state.documents
                        )
                    elif mode == "Summarize Notes":
                        answer = assistant.summarize_notes(
                            question, model, api_key, st.session_state.documents
                        )
                    elif mode == "Generate MCQs":
                        answer = assistant.generate_mcqs(
                            question, model, api_key, st.session_state.documents
                        )
                    else:
                        answer = "Please select a valid mode."

                    st.markdown(answer)
                    add_message(active_chat, "assistant", answer)
                    st.rerun()
                except Exception as error:
                    st.error(f"Something went wrong: {error}")


# ---------------------------
# Document Manager
# ---------------------------
elif page == "Document Manager":
    st.title("📁 Document Manager")
    st.write("Upload and index files for document questions, summaries and MCQ generation.")
    st.info(
        "Supported: PDF, DOCX, TXT, MD, PPTX, PNG, JPG and JPEG. "
        "Image text is extracted using OCR."
    )

    st.subheader("1. Upload individual or multiple files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=sorted(assistant.SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="individual_uploads"
    )
    if st.button("Upload and Index Selected Files", type="primary", disabled=not uploaded_files):
        progress = st.progress(0)
        outcomes = []
        for index, uploaded in enumerate(uploaded_files):
            try:
                outcomes.append(assistant.save_and_index_upload(uploaded, st.session_state.documents))
            except Exception as error:
                outcomes.append({"name": uploaded.name, "status": f"Failed: {error}", "chunks": 0})
            progress.progress((index + 1) / len(uploaded_files))

        for item in outcomes:
            if item["status"] in ("Uploaded and indexed", "Already uploaded"):
                st.success(f"{item['name']} — {item['status']} ({item['chunks']} chunks)")
            else:
                st.error(f"{item['name']} — {item['status']}")
        st.rerun()

    st.divider()
    st.subheader("2. Upload a complete folder")
    st.caption("Choose a folder in the file picker. Supported files inside it are uploaded together.")
    folder_files = st.file_uploader(
        "Choose a folder",
        type=sorted(assistant.SUPPORTED_EXTENSIONS),
        accept_multiple_files="directory",
        key="folder_uploads"
    )
    if st.button("Upload and Index Folder", disabled=not folder_files):
        progress = st.progress(0)
        outcomes = []
        for index, uploaded in enumerate(folder_files):
            try:
                outcomes.append(assistant.save_and_index_upload(uploaded, st.session_state.documents))
            except Exception as error:
                outcomes.append({"name": uploaded.name, "status": f"Failed: {error}", "chunks": 0})
            progress.progress((index + 1) / len(folder_files))
        for item in outcomes:
            if item["status"] in ("Uploaded and indexed", "Already uploaded"):
                st.success(f"{item['name']} — {item['status']} ({item['chunks']} chunks)")
            else:
                st.error(f"{item['name']} — {item['status']}")
        st.rerun()

    st.divider()
    st.subheader("3. Your document library")
    library = assistant.get_document_library(st.session_state.documents)
    st.metric("Indexed documents", len(library))
    if library:
        st.dataframe(
            pd.DataFrame([{"Document": x["name"], "Indexed text chunks": x["chunks"]} for x in library]),
            use_container_width=True,
            hide_index=True
        )
        selected_document = st.selectbox(
            "Select a document to remove",
            [x["name"] for x in library],
            key="delete_document_choice"
        )
        if st.button("Delete Selected Document"):
            removed = assistant.delete_document(selected_document, st.session_state.documents)
            st.success(f"Removed '{selected_document}' ({removed} document entry).")
            st.rerun()
    else:
        st.warning("No documents uploaded yet. Add a file above to get started.")

    st.divider()
    st.subheader("4. Clear the complete document library")
    st.warning("This clears the documents uploaded in this browser session. It does not delete chat history.")
    confirm_clear = st.checkbox("I understand. Delete all documents and their index.")
    if st.button("Clear All Documents", disabled=not confirm_clear):
        removed = assistant.clear_document_library(st.session_state.documents)
        st.success(f"Document library cleared. Removed {removed} documents.")
        st.rerun()


# ---------------------------
# Chat History
# ---------------------------
elif page == "Chat History":
    st.title("📚 Chat History")
    st.write("Open, continue, or delete conversations from this browser session.")
    st.divider()
    conversations = list(reversed(st.session_state.conversations))
    if conversations:
        st.caption(f"Total conversations: {len(conversations)}")
        for conversation in conversations:
            chat_id = conversation["id"]
            with st.expander(f"💬 {conversation['title']} | {conversation['created_at']}"):
                st.caption(f"{len(conversation['messages'])} messages")
                for message in conversation["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Continue Chat", key=f"continue_{chat_id}", use_container_width=True):
                        open_chat(chat_id)
                        st.session_state.page_notice = "Previous conversation loaded."
                        st.rerun()
                with col2:
                    if st.button("Delete Chat", key=f"delete_{chat_id}", use_container_width=True):
                        delete_chat(chat_id)
                        st.rerun()
    else:
        st.info("No conversations saved in this browser session yet.")


# ---------------------------
# Benchmark Dashboard
# ---------------------------
elif page == "Benchmark Dashboard":
    st.title("📊 Model Benchmark Dashboard")
    st.write("Review the latest model comparison performed in this browser session.")
    results = st.session_state.benchmark_results
    if results:
        successful = [r for r in results if not r.get("failed")]
        if successful:
            df = pd.DataFrame([
                {
                    "Model": r["model"],
                    "Average Response Time (seconds)": round(r["response_time"], 2),
                    "Peak RAM (MB)": None,
                    "Quality Rating": r["quality"]
                }
                for r in successful
            ])
            st.success("Latest benchmark report")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Peak RAM is unavailable for cloud-hosted models; the API does not expose server memory usage.")
            st.subheader("⏱️ Response Time")
            st.bar_chart(df.set_index("Model")["Average Response Time (seconds)"])
            st.subheader("⭐ Quality Rating")
            st.bar_chart(df.set_index("Model")["Quality Rating"])
        else:
            st.warning("The latest benchmark did not complete successfully for either model.")
    else:
        st.warning("No benchmark report found. Run Benchmark Both Models from AI Assistant.")
