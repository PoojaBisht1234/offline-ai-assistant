import hashlib
import io
import re
from datetime import datetime
from pathlib import Path

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SYSTEM_PROMPT = """
You are a friendly and intelligent AI assistant.
Explain difficult concepts in simple language.
Give step-by-step explanations when useful.
Be polite, patient, organized, and honest when you do not know something.
Use examples whenever useful.
"""

# Groq production model IDs. Check Groq's live model catalogue if a model is
# retired or unavailable: https://console.groq.com/docs/models
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]

SUPPORTED_EXTENSIONS = {
    "pdf", "docx", "txt", "md", "pptx", "png", "jpg", "jpeg"
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_client(api_key):
    if not api_key:
        raise RuntimeError(
            "Groq API key not configured. Add GROQ_API_KEY in "
            "Streamlit Cloud → App → Settings → Secrets."
        )
    return Groq(api_key=api_key)


def generate_answer(prompt, model, api_key, history=None, system_prompt=None):
    client = get_client(api_key)
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
    ]

    for item in (history or [])[-10:]:
        if item.get("role") in ("user", "assistant"):
            messages.append({
                "role": item["role"],
                "content": str(item.get("content", ""))[:12000]
            })

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content or ""


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_text(filename, content):
    extension = Path(filename).suffix.lower().lstrip(".")
    stream = io.BytesIO(content)

    if extension == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(stream)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if extension == "docx":
        from docx import Document
        doc = Document(stream)
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if extension in {"txt", "md"}:
        return content.decode("utf-8", errors="ignore")

    if extension == "pptx":
        from pptx import Presentation
        presentation = Presentation(stream)
        parts = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)
        return "\n".join(parts)

    if extension in {"png", "jpg", "jpeg"}:
        from PIL import Image
        import pytesseract
        image = Image.open(stream)
        return pytesseract.image_to_string(image)

    raise ValueError(f"Unsupported file type: .{extension}")


def save_and_index_upload(uploaded_file, document_library):
    filename = Path(uploaded_file.name).name
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {filename}")

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError(f"{filename} is empty.")

    digest = hashlib.sha256(content).hexdigest()
    if any(item["file_id"] == digest for item in document_library):
        existing = next(item for item in document_library if item["file_id"] == digest)
        return {
            "name": filename,
            "status": "Already uploaded",
            "chunks": len(existing["chunks"])
        }

    extracted = extract_text(filename, content)
    chunks = _chunk_text(extracted)
    if not chunks:
        raise ValueError("No readable text was found in this file.")

    document_library.append({
        "name": filename,
        "file_id": digest,
        "chunks": chunks,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    return {
        "name": filename,
        "status": "Uploaded and indexed",
        "chunks": len(chunks)
    }


def get_document_library(document_library):
    return [
        {"name": item["name"], "chunks": len(item["chunks"]), "file_id": item["file_id"]}
        for item in document_library
    ]


def delete_document(source_name, document_library):
    before = len(document_library)
    document_library[:] = [
        item for item in document_library if item["name"] != source_name
    ]
    return before - len(document_library)


def clear_document_library(document_library):
    removed = len(document_library)
    document_library.clear()
    return removed


def search_document_chunks(question, document_library, limit=5):
    corpus = []
    metadata = []
    for item in document_library:
        for index, chunk in enumerate(item["chunks"]):
            corpus.append(chunk)
            metadata.append({
                "source": item["name"],
                "chunk_index": index
            })

    if not corpus:
        return [], []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        return [], []

    ranked = scores.argsort()[::-1][:limit]
    documents = [corpus[i] for i in ranked if scores[i] > 0]
    metadatas = [metadata[i] for i in ranked if scores[i] > 0]
    return documents, metadatas


def answer_from_pdf(question, model, api_key, document_library):
    if not document_library:
        return "Your document library is empty. Open Document Manager and upload files first."

    documents, metadatas = search_document_chunks(question, document_library, 5)
    if not documents:
        return "No relevant information was found in your uploaded documents."

    context = "\n\n".join(
        f"Source: {meta['source']}\nContent: {chunk}"
        for chunk, meta in zip(documents, metadatas)
    )
    prompt = f"""
Answer using only the document content below.
If the answer is not present, say so clearly.
Give a simple, clear answer.

DOCUMENT CONTENT:
{context}

QUESTION:
{question}
"""
    return generate_answer(prompt, model, api_key)


def summarize_notes(topic, model, api_key, document_library):
    if not document_library:
        return "Your document library is empty. Upload documents in Document Manager first."

    documents, metadatas = search_document_chunks(topic, document_library, 10)
    if not documents:
        return f"No relevant notes found for: {topic}"

    source_context = "\n\n".join(
        f"Source: {meta['source']}\n{chunk}"
        for chunk, meta in zip(documents, metadatas)
    )
    prompt = f"""
Create useful, well-organized study notes about: {topic}
Use simple language, headings, definitions, explanations, examples and bullet points.
Avoid unrelated content. Base the notes on the provided material.

MATERIAL:
{source_context}
"""
    return generate_answer(prompt, model, api_key)


def generate_mcqs(topic, model, api_key, document_library):
    if not document_library:
        return "Your document library is empty. Upload documents in Document Manager first."

    documents, _ = search_document_chunks(topic, document_library, 8)
    if not documents:
        return "No relevant notes found for this topic."

    context = "\n\n".join(documents)
    prompt = f"""
Generate 10 multiple-choice questions from the study material below.

TOPIC:
{topic}

STUDY MATERIAL:
{context}

For every question:
- Give four options: A, B, C, D.
- Mention the correct answer.
- Give a short explanation.
Use a format suitable for university exams.
"""
    return generate_answer(prompt, model, api_key)


def normal_chat(question, model, api_key, history, all_conversations, current_chat_id):
    # Retrieve a few relevant user/assistant exchanges from other chats using TF-IDF.
    memory_texts = []
    memory_labels = []
    for chat in all_conversations:
        if chat.get("id") == current_chat_id:
            continue
        messages = chat.get("messages", [])
        for i in range(len(messages) - 1):
            if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                memory_texts.append(
                    f"User: {messages[i].get('content', '')}\n"
                    f"Assistant: {messages[i + 1].get('content', '')}"
                )
                memory_labels.append(chat.get("title", "Previous conversation"))

    memory_context = ""
    if memory_texts:
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(memory_texts + [question])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            selected = scores.argsort()[::-1][:2]
            useful = [
                f"Previous chat ({memory_labels[i]}):\n{memory_texts[i]}"
                for i in selected if scores[i] > 0
            ]
            if useful:
                memory_context = "\n\nRelevant context from previous chats:\n" + "\n\n".join(useful)
        except ValueError:
            pass

    system = SYSTEM_PROMPT + memory_context
    return generate_answer(question, model, api_key, history=history, system_prompt=system)
