"""Streamlit entry point for document Q&A."""
import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import APIError
from streamlit.errors import StreamlitAPIException

from document_loader import extract
from rag_pipeline import answer_question
from vector_store import DocumentIndex


AVATAR_DIR = Path(__file__).with_name("assets")


def show_message(message):
    role = message["role"]
    with st.chat_message(role, avatar=str(AVATAR_DIR / f"{role}.png")):
        st.write(message["content"])



def show_retrieved_sources(messages):
    answers = {
        index: message["sources"]
        for index, message in enumerate(messages)
        if message["role"] == "assistant" and message.get("sources")
    }
    if not answers:
        return

    def answer_label(index):
        if index == "latest":
            return "Latest answer"
        question = messages[index - 1]["content"]
        return question if len(question) <= 65 else f"{question[:62]}..."

    with st.sidebar, st.expander("Retrieved sources"):
        selected = st.selectbox(
            "Answer",
            options=["latest", *reversed(answers)],
            format_func=answer_label,
        )
        sources = answers[next(reversed(answers))] if selected == "latest" else answers[selected]
        for number, label in enumerate(sources, 1):
            st.text(f"[{number}] {label}")


def show_file_preview(uploads, previews):
    if not uploads:
        return

    st.subheader("File preview")
    selected = st.selectbox(
        "Choose an uploaded file",
        options=range(len(uploads)),
        format_func=lambda index: uploads[index][0],
    )
    name, data = uploads[selected]
    if Path(name).suffix.lower() == ".pdf":
        try:
            st.pdf(data, height=480, alt=f"Preview of {name}")
        except StreamlitAPIException:
            pass
        return

    sections = previews[selected]
    if not sections:
        st.caption("No text is available for this file.")
    elif len(sections) == 1:
        with st.container(height=320, border=True):
            st.text(sections[0].text)
    else:
        selected_page = st.selectbox(
            "Page",
            options=range(len(sections)),
            format_func=lambda index: sections[index].source,
        )
        with st.container(height=320, border=True):
            st.text(sections[selected_page].text)


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    st.set_page_config(page_title="Document Q&A")
    state = st.session_state
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader(
            "Upload PDF, Word, text, or Markdown files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        uploads = [(file.name, file.getvalue()) for file in files]
        signature = [(name, hashlib.sha256(data).digest()) for name, data in uploads]
        if signature != state.get("uploads") or "previews" not in state:
            sections, warnings, previews = [], [], []
            for name, data in uploads:
                extracted = []
                try:
                    extracted = [section for section in extract(name, data) if section.text.strip()]
                    if not extracted:
                        warnings.append(f"{name}: No readable text was found.")
                    sections.extend(extracted)
                except Exception:
                    warnings.append(f"{name}: The file could not be read.")
                previews.append(extracted)
            previous_index = state.get("index")
            if hasattr(previous_index, "close"):
                previous_index.close()
            index = None
            if sections:
                try:
                    with st.spinner("Building the Chroma index?"):
                        index = DocumentIndex(sections)
                except Exception:
                    warnings.append(
                        "Chroma could not index these files. Check that its embedding model "
                        "can download, then remove and re-upload the files to retry."
                    )
            state.update(
                uploads=signature, messages=[], warnings=warnings, index=index, previews=previews,
            )
        for warning in state.warnings:
            st.warning(warning)
        show_file_preview(uploads, state.previews)

    st.title("Document Q&A")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.info("Set GROQ_API_KEY in .env to enable chat.")
    elif state.index is None:
        st.info("Upload a document with readable text to start.")
    for message in state.messages:
        show_message(message)
    question = st.chat_input("Ask about your documents", disabled=not api_key or state.index is None)
    if question and question.strip():
        history = [
            {"role": message["role"], "content": message["content"][:1000]}
            for message in state.messages[-6:]
        ]
        user_message = {"role": "user", "content": question.strip()}
        show_message(user_message)
        try:
            with st.spinner("Reading your documents…"):
                answer, sources = answer_question(state.index, question.strip(), history, api_key)
        except APIError as error:
            status = getattr(error, "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            st.error(f"Groq could not answer{detail}. Check your API key and model access, then retry.")
        else:
            reply = {"role": "assistant", "content": answer, "sources": sources}
            state.messages.extend([user_message, reply])
            show_message(reply)

    show_retrieved_sources(state.messages)


if __name__ == "__main__":
    main()
