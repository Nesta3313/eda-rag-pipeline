from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
import numpy as np
import faiss

from .embedder import load_app_config, embed_chunks
from ..chunking.chunker import build_chunks

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_faiss_index(dataset_name: str, vectors: np.ndarray, records: List[Dict[str, Any]], out_dir: str):
    out_path = Path(out_dir) / dataset_name
    _ensure_dir(out_path)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine ~ normalized dot; we can L2-normalize first
    # normalize to unit length for cosine similarity behavior
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    normed = vectors / norms

    index.add(normed.astype(np.float32))

    faiss.write_index(index, str(out_path / "index.faiss"))
    with open(out_path / "meta.pkl", "wb") as f:
        pickle.dump({"records": records}, f)

def build_index(artifacts_dir: str) -> Tuple[str, int]:
    """
    Build an index for a dataset artifacts directory like:
    data/artifacts/<dataset_name>/
    Returns (dataset_name, n_chunks)
    """
    cfg = load_app_config()
    ds_name = Path(artifacts_dir).name

    # 1) chunk
    chunks = build_chunks(dataset_name=ds_name, artifacts_dir=artifacts_dir,
                          chunk_size=800, chunk_overlap=120)

    # 2) embed
    emb = embed_chunks(chunks, cfg.embeddings_provider, cfg.embeddings_model)

    # 3) persist FAISS
    save_faiss_index(ds_name, emb.vectors, emb.records, cfg.vector_dir)

    return ds_name, len(chunks)

def load_index(dataset_name: str, vector_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    p = Path(vector_dir) / dataset_name
    index = faiss.read_index(str(p / "index.faiss"))
    with open(p / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta["records"]

def search(dataset_name: str, query_vec: np.ndarray, k: int = 5, vector_dir: str = "storage/vectors"):
    index, records = load_index(dataset_name, vector_dir)
    # normalize query for cosine-like similarity
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    D, I = index.search(q.reshape(1, -1).astype(np.float32), k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        rec = records[idx]
        hits.append({"rank": rank+1, "score": float(D[0][rank]), "record": rec})
    return hits