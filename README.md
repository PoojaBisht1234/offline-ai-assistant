# Offline AI Assistant

🚀 Live demo: https://offline-ai-assistant-9w5kjgtjkxpm63gasceynu.streamlit.app/

# AI Assistant with Model Benchmarking (Cloud Edition)

A public Streamlit demo adapted from an offline Ollama project. This cloud edition uses Groq-hosted language models so it can run on Streamlit Community Cloud.

## Features

- AI chat with conversation history during the current browser session
- Document upload and retrieval for PDF, DOCX, TXT, Markdown, PowerPoint and images
- OCR for PNG/JPG/JPEG files
- Study-note summarization from uploaded documents
- 10 multiple-choice questions generated from uploaded documents
- Compare two hosted models using the same question
- Response-time and user-rated answer-quality comparison
- CSV benchmark download and dashboard

## Important: Cloud edition vs offline edition

This repository is the **cloud edition**. It does not run Ollama locally. AI inference is sent to Groq's hosted API.

The original offline edition should remain in its own folder/repository and continue using Ollama.

Because Groq does not expose the server's RAM usage to this app, the cloud benchmark reports response time and manual quality rating. The `Peak RAM (MB)` field is left blank in downloaded CSV files for compatibility with the original report format.

Chat history and uploaded documents are kept in Streamlit session state for the active browser session. They are not a permanent multi-user database and may be cleared when the session ends or the app restarts. Do not upload confidential documents to a public demo.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "paste-your-new-key-here"
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

Never commit `.streamlit/secrets.toml` or publish your API key.

## Deploy on Streamlit Community Cloud

1. Push all files in this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose this repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.
6. In the deployed app's settings, open **Secrets** and add:

   ```toml
   GROQ_API_KEY = "your-new-groq-api-key"
   ```

7. Save and reboot the app.

## Models

The app compares these Groq model IDs by default:

- `openai/gpt-oss-20b`
- `llama-3.3-70b-versatile`

Model availability, access, quotas and pricing can change. See the live [Groq supported models list](https://console.groq.com/docs/models).

## Tech stack

Python · Streamlit · Groq API · TF-IDF retrieval · ChromaDB-free in-memory document indexing · PyPDF · python-docx · python-pptx · Tesseract OCR
