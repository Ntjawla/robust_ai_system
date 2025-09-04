import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Get project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load knowledge base
kb_file = os.path.join(PROJECT_ROOT, "data", "knowledge_base.txt")
with open(kb_file, "r", encoding="utf-8") as f:
    kb = f.read().splitlines()

# Create embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kb_embeddings = embedder.encode(kb)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(np.array(kb_embeddings))

# Retrieve top-k relevant knowledge
def retrieve_context(query, top_k=2):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [kb[i] for i in I[0]]
