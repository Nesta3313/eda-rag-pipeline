from __future__ import annotations
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as np
from pydantic import BaseModel
import yaml

load_dotenv("config/.env")

class AppConfig(BaseModel):
    embeddings_provider: str
    embeddings_model: str
    vector_dir: str

def load_app_config(path: str = "config/config.yaml") -> AppConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return AppConfig(
        embeddings_provider=cfg.get("embeddings", {}).get("provider", "openai"),
        embeddings_model=cfg.get("embeddings", {}).get("model", "text-embedding-3-small"),
        vector_dir=cfg.get("paths", {}).get("vector_dir", "storage/vectors")
    )

@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    records: List[Dict[str, Any]]

class EmbeddingClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.provider == "openai":
            resp = self.client.embeddings.create(model=self.model, input=texts)
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

def embed_chunks(chunks: List[Dict[str, Any]], provider: str, model: str) -> EmbeddingResult:
    ec = EmbeddingClient(provider, model)
    texts = [c["text"] for c in chunks]
    vectors = ec.embed_texts(texts)
    return EmbeddingResult(vectors=vectors, records=chunks)
