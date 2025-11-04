from __future__ import annotations

from typing import List, Dict, Any

from src.embeddings.embedder import EmbeddingClient, load_app_config
from src.embeddings.vector_store import search


def retrieve_relevant_chunks(dataset_name: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Embed a query and return top-k similar chunks with metadata."""
    cfg = load_app_config()
    ec = EmbeddingClient(cfg.embeddings_provider, cfg.embeddings_model)
    q_vec = ec.embed_texts([query])[0]
    results = search(dataset_name, q_vec, k=k, vector_dir=cfg.vector_dir)
    return results
