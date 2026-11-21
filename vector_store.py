"""Index uploaded document chunks and retrieve excerpts with Chroma."""
from uuid import uuid4

import chromadb

from document_loader import Excerpt

CONTEXT_LIMIT = 12000


class DocumentIndex:
    def __init__(self, sections):
        self.sections = sections
        self.client = chromadb.EphemeralClient()
        self.name = f"documents-{uuid4().hex}"
        collection = self.client.create_collection(self.name)
        try:
            chunks = []
            for section in sections:
                # Split by characters to bound even very long unbroken text.
                for start in range(0, len(section.text), 800):
                    chunk = section.text[start:start + 1000].strip()
                    if chunk:
                        chunks.append(Excerpt(section.source, chunk))
                    if start + 1000 >= len(section.text):
                        break
            for start in range(0, len(chunks), 128):
                batch = chunks[start:start + 128]
                collection.add(
                    ids=[str(number) for number in range(start, start + len(batch))],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[{"source": chunk.source} for chunk in batch],
                )
        except Exception:
            self.close()
            raise
        self.collection = collection
        self.chunk_count = len(chunks)

    def close(self):
        self.client.delete_collection(self.name)

    @staticmethod
    def fits(sections):
        return sum(len(section.text) + len(section.source) + 20 for section in sections) <= CONTEXT_LIMIT

    def select(self, question, history):
        if self.fits(self.sections):
            return self.sections
        # Put the current question first so prior turns add context to follow-ups.
        previous_turn = history[-2:]
        query = "\n".join([question, *(turn["content"][:200] for turn in previous_turn)])
        results = self.collection.query(
            query_texts=[query],
            n_results=min(8, self.chunk_count),
            include=["documents", "metadatas"],
        )
        selected = []
        for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunk = Excerpt(metadata["source"], document)
            if self.fits([*selected, chunk]):
                selected.append(chunk)
        return selected
