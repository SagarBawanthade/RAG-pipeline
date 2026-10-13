import sys
from rag import load_documents, chunk_text, chunk_by_section, DOCS_FOLDER

chunker = chunk_by_section if "section" in sys.argv else chunk_text

for doc in load_documents(DOCS_FOLDER):
    for i, piece in enumerate(chunker(doc["text"]), start=1):
        print(f"===== CHUNK {i} ({len(piece)} chars) =====")
        print(piece)
        print()