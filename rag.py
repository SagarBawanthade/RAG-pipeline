from pathlib import Path

import numpy as np
import requests
import sys
import chromadb

# Imports for the RAG

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

DOCS_FOLDER = "docs"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks
TOP_K = 3             # how many chunks to retrieve per question
MIN_SCORE = 0.55     # chunks scoring below this are ignored
CHROMA_PATH = "chroma_db"       # folder where Chroma saves its data
COLLECTION_NAME = "k8s_docs"    # like a table name



def load_documents(folder):
    docs = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix in {".txt", ".md"}:
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs




def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap   # step back a little, that is the overlap
    return chunks

def chunk_by_section(text, max_size=1000):
    chunks = []
    for section in text.split("\n\n"):
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_size:
            chunks.append(section)           # keep the whole section as one chunk
        else:
            chunks.extend(chunk_text(section))  # too long, split it the old way
    return chunks


def embed(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"])



def get_collection(rebuild=False):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass   # nothing to delete the first time

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine similarity, like before
    )

    if collection.count() == 0:
        print("Collection is empty, chunking and embedding documents...")
        ids, texts, metas = [], [], []
        for doc in load_documents(DOCS_FOLDER):
            for i, piece in enumerate(chunk_by_section(doc["text"])):
                ids.append(f"{doc['source']}-{i}")
                texts.append(piece)
                metas.append({"source": doc["source"]})
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embed(texts).tolist(),
            metadatas=metas,
        )
    return collection


def retrieve(question, collection, k=TOP_K):
    q = embed([question])[0].tolist()
    result = collection.query(query_embeddings=[q], n_results=k)

    hits = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        score = 1 - dist   # Chroma returns distance, we want similarity
        hits.append((score, {"source": meta["source"], "text": text}))
    return hits



def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{i + 1}] (source: {chunk['source']})\n{chunk['text']}"
        for i, (score, chunk) in enumerate(hits)
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:"""


def generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    rebuild = "--rebuild" in sys.argv
    collection = get_collection(rebuild)
    print(f"Ready. {collection.count()} chunks in the database.\n")

    while True:
        question = input("Ask a question (or type 'quit'): ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        hits = retrieve(question, collection) 

        print("\n--- Retrieved chunks ---")
        for score, chunk in hits:
            status = "USED   " if score >= MIN_SCORE else "IGNORED"
            print(f"{status} score={score:.2f} | {chunk['source']}")
            print(chunk["text"][:200].replace("\n", " ") + "...\n")

        good_hits = [(s, c) for s, c in hits if s >= MIN_SCORE]

        if not good_hits:
            print("--- Answer ---")
            print("I don't know based on the documents.\n")
            continue

        answer = generate(build_prompt(question, good_hits))
        print("--- Answer ---")
        print(answer, "\n")


if __name__ == "__main__":
    main()
