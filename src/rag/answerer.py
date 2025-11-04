from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.rag.retriever import retrieve_relevant_chunks

load_dotenv("config/.env")

def answer(question: str, dataset_name: str = "medical_insurance", k: int = 5) -> str:
    """Retrieve top-k EDA chunks and use LLM to answer grounded on them."""
    retrieved = retrieve_relevant_chunks(dataset_name, question, k=k)

    context_blocks = []
    for hit in retrieved:
        meta = hit["record"]["metadata"]
        context_blocks.append(f"[Section: {meta['section']}] {hit['record']['text']}")
    context_text = "\n\n---\n\n".join(context_blocks)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are a data analyst assistant.
You are helping interpret an EDA report for the dataset '{dataset_name}'.
Use ONLY the following context to answer the user's question.
If the answer is not found in context, say you don't have enough data.

Context:
{context_text}

Question:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
