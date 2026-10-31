"""Build the two persistent Chroma indexes using local SentenceTransformers embeddings."""
from __future__ import annotations
import json
from .rag import build_chroma_indexes

if __name__ == "__main__":
    print(json.dumps({"indexed_collections":build_chroma_indexes(),"embedding_model":"sentence-transformers/all-MiniLM-L6-v2","network":"disabled; local_files_only=True"},indent=2))
