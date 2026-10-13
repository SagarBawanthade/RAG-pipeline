# RAG from Scratch

A beginner-friendly Retrieval-Augmented Generation (RAG) project built with Python, Ollama, and ChromaDB.

## Tech Stack

- Python
- Ollama
- ChromaDB
- nomic-embed-text — embeddings
- llama3.2:3b — answer generation

## Pipeline

1. Load documents from the docs directory.
2. Split documents into chunks.
3. Generate embeddings using Ollama.
4. Store chunks and embeddings in ChromaDB.
5. Retrieve relevant chunks using semantic search.
6. Filter retrieved chunks by similarity score.
7. Generate answers using the retrieved context.

## Project Structure

```text
rag-from-scratch/
├── docs/
│   └── k8s_notes.txt
├── rag.py
├── show_chunks.py
├── requirements.txt
├── .gitignore
└── README.md

## Setup
1. Install Python and Docker.
2. Run Ollama and download the required models:
   docker exec -it ollama ollama pull nomic-embed-text
   docker exec -it ollama ollama pull llama3.2:3b
3. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
4. Install dependencies:
   pip install -r requirements.txt
5. Run the application:
   python rag.py
