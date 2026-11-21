# Document Q&A with Streamlit

Upload PDF, DOCX, TXT, or Markdown files and ask questions about their contents. The app uses retrieval-augmented generation (RAG) with Chroma for semantic search and Groq for answer generation. Answers include source references, and recent chat history helps resolve follow-up questions.

## Run locally on Windows

Open PowerShell in the project folder. The example uses Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Keep the key in that local file, which Git ignores. Optionally set `GROQ_MODEL` to choose a different model available to your Groq account. The default is `openai/gpt-oss-20b`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, upload a document, and ask a question. For example, ask "What are the main points?" and follow up with "Can you explain the second point?"

## How it works

1. The app extracts text from uploaded files and records source names and PDF page numbers.
2. It splits the text into overlapping chunks. Chroma embeds and indexes those chunks in memory. The default embedding model can download on first use.
3. Short uploads fit entirely in the prompt. For larger uploads, Chroma retrieves relevant chunks using the question and recent chat context.
4. Groq receives the selected excerpts, the question, and recent chat history. The app displays the answer and its retrieved sources.

Scanned PDFs need a searchable text layer. Starting a new session requires uploading the documents again.

## Project files

| File | Responsibility |
| --- | --- |
| `streamlit_app.py` | Upload controls, chat UI, and session state |
| `document_loader.py` | Text extraction and source labels |
| `vector_store.py` | Text chunking, Chroma indexing, and retrieval |
| `rag_pipeline.py` | Prompt assembly and Groq requests |
| `requirements.txt` | Python dependencies |
| `.env.example` | Local configuration template |
