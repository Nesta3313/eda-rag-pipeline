from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import re

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120

def load_artifacts(dataset_artifacts_dir: str) -> Dict[str, Path]:
    """Return paths to the JSON + Markdown artifacts for a dataset."""
    p = Path(dataset_artifacts_dir)
    assert p.exists(), f"Artifacts dir not found: {p}"
    json_path = p / "eda_summary.json"
    md_path = p / "eda_summary.md"
    assert json_path.exists(), f"Missing {json_path}"
    assert md_path.exists(), f"Missing {md_path}"
    return {"json": json_path, "md": md_path}

def _split_markdown_sections(md_text: str) -> List[Dict[str, str]]:
    """
    Coarse split by Markdown headings. Each section keeps its title.
    Return list of dicts: {'title': '## Schema', 'text': '...'}
    """
    lines = md_text.splitlines()
    sections = []
    current_title = "Document"
    current_buf: List[str] = []

    for ln in lines:
        if re.match(r"^#{1,6}\s", ln):
            # flush previous
            if current_buf:
                sections.append({"title": current_title, "text": "\n".join(current_buf).strip()})
                current_buf = []
            current_title = ln.strip()
        else:
            current_buf.append(ln)
    if current_buf:
        sections.append({"title": current_title, "text": "\n".join(current_buf).strip()})
    return sections

def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Simple char-based chunker with overlap.
    """
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return [c.strip() for c in chunks if c.strip()]

def build_chunks(dataset_name: str, artifacts_dir: str,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_OVERLAP) -> List[Dict[str, Any]]:
    """
    Produce retrievable chunks with metadata.
    Each chunk: {'id','text','metadata':{'dataset','section','source','path'}}
    """
    paths = load_artifacts(artifacts_dir)
    md_text = Path(paths["md"]).read_text(encoding="utf-8")
    json_data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    # 1) Markdown sections → chunk
    sections = _split_markdown_sections(md_text)
    out: List[Dict[str, Any]] = []
    counter = 0
    for sec in sections:
        sec_chunks = _chunk_text(sec["text"], chunk_size, chunk_overlap)
        for idx, ch in enumerate(sec_chunks):
            counter += 1
            out.append({
                "id": f"{dataset_name}::md::{counter}",
                "text": f"{sec['title']}\n\n{ch}",
                "metadata": {
                    "dataset": dataset_name,
                    "section": sec["title"].lstrip("# ").strip(),
                    "source": "eda_summary.md",
                    "path": str(paths["md"])
                }
            })

    # 2) JSON key areas → chunk (schema, missingness, correlations, warnings)
    important_keys = ["overview", "schema", "missingness", "descriptives", "correlations",
                      "validators", "leakage_checks", "warnings"]
    for k in important_keys:
        if k not in json_data:
            continue
        text_block = json.dumps(json_data[k], ensure_ascii=False, indent=2)
        for ch in _chunk_text(text_block, chunk_size, chunk_overlap):
            counter += 1
            out.append({
                "id": f"{dataset_name}::json::{counter}",
                "text": f"{k.upper()}:\n{ch}",
                "metadata": {
                    "dataset": dataset_name,
                    "section": k,
                    "source": "eda_summary.json",
                    "path": str(paths["json"])
                }
            })
    return out