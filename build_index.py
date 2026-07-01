import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data_prep import load_catalog

INDEX_FILE = "shl_faiss.index"

def build():
    print("[build_index] Loading catalog...")
    catalog_list, _ = load_catalog()
    doc_texts = [item["doc_text"] for item in catalog_list]
    
    print("[build_index] Loading SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("[build_index] Encoding documents...")
    embeddings = model.encode(
        doc_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    dim = embeddings.shape[1]
    
    print(f"[build_index] Building FAISS index (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, INDEX_FILE)
    print(f"[build_index] Index saved to {INDEX_FILE}.")

if __name__ == "__main__":
    build()
