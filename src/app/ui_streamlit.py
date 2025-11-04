import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- Make "src/" importable when run via Streamlit ---
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.eda.summarize import run_eda
from src.embeddings.vector_store import build_index
from src.rag.answerer import answer
from src.rag.retriever import retrieve_relevant_chunks
from src.embeddings.embedder import load_app_config

# ---------- Config & constants ----------
load_dotenv("config/.env")

st.set_page_config(page_title="EDA RAG Assistant", layout="wide")
st.title("🧠 EDA RAG Assistant")

cfg = load_app_config()

# Server directories (override with env vars if desired)
BASE_DATA_DIR = Path(os.getenv("BASE_DATA_DIR", "data/raw")).resolve()       # curated server CSVs
UPLOADS_DIR   = Path(os.getenv("UPLOADS_DIR", "data/uploads")).resolve()     # user uploads
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_dataset_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", name).strip("-_").lower()
    return slug or "dataset"

def list_csvs(base: Path) -> list[Path]:
    try:
        return sorted(p for p in base.rglob("*.csv") if p.is_file())
    except Exception:
        return []

def save_uploaded_csv(file, dataset_name_hint: str | None = None) -> Path:
    """
    Save uploaded CSV to UPLOADS_DIR/<dataset_name>/<original_name>.csv
    Returns saved file path.
    """
    ensure_dir(UPLOADS_DIR)
    hinted = dataset_name_hint or Path(file.name).stem
    ds = sanitize_dataset_name(hinted)
    ds_dir = (UPLOADS_DIR / ds).resolve()

    # stay inside uploads dir
    if not str(ds_dir).startswith(str(UPLOADS_DIR)):
        raise ValueError("Invalid dataset path.")
    ensure_dir(ds_dir)

    # size check
    file_bytes = file.read()
    file.seek(0)
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"File too large: {size_mb:.1f} MB (limit {MAX_UPLOAD_MB} MB).")

    # extension check
    if not file.name.lower().endswith(".csv"):
        raise ValueError("Only .csv files are supported.")

    out_path = ds_dir / Path(file.name).name
    with open(out_path, "wb") as f:
        f.write(file_bytes)
    return out_path

# ---------- Sidebar: environment checks ----------
with st.sidebar:
    st.header("Settings")
    st.write(f"Vector store: `{cfg.vector_dir}`")
    st.write(f"Server CSV base: `{BASE_DATA_DIR}`")
    st.write(f"Uploads dir: `{UPLOADS_DIR}`")
    api_present = bool(os.getenv("OPENAI_API_KEY"))
    st.write(f"OpenAI key loaded: {'✅' if api_present else '❌'}")
    if not api_present:
        st.warning("No OPENAI_API_KEY found in config/.env")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["1) Run EDA", "2) Build Index", "3) Ask Questions"])

# =============== Tab 1: Run EDA ===============
with tab1:
    st.subheader("Run EDA")

    mode = st.radio("Choose data source", ["Upload CSV", "Pick server CSV"], horizontal=True)
    target = st.text_input("Optional: Target column (for leakage checks)", value="")

    if mode == "Upload CSV":
        st.caption(f"Uploads are saved under: `{UPLOADS_DIR}`")
        up = st.file_uploader("Upload a CSV", type=["csv"], accept_multiple_files=False)
        ds_hint = st.text_input("Dataset name (folder to save under)", value="", placeholder="e.g., heart_failure_v2")
        build_now = st.toggle("Build index now for Q&A", value=True)

        if st.button("Run EDA on uploaded file", type="primary", disabled=up is None):
            try:
                if up is None:
                    st.error("Please upload a CSV.")
                else:
                    saved_path = save_uploaded_csv(up, dataset_name_hint=ds_hint or Path(up.name).stem)
                    st.success(f"Uploaded to: {saved_path}")

                    # Run EDA
                    summary = run_eda(str(saved_path), target=target or None)
                    ds = Path(saved_path).stem
                    art_dir = Path("data/artifacts") / ds

                    st.info(f"Artifacts written to: {art_dir}")
                    st.caption("Quick overview")
                    st.json(summary.get("overview", {}))
                    if summary.get("warnings"):
                        st.caption("Warnings")
                        for w in summary["warnings"]:
                            st.warning(w)

                    # ⬇️ Download buttons for MD / HTML
                    md_path = art_dir / "eda_summary.md"
                    html_path = art_dir / "profile.html"
                    st.divider()
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        if md_path.exists():
                            st.download_button(
                                label="⬇️ Download EDA Markdown",
                                data=md_path.read_text(encoding="utf-8"),
                                file_name=f"{ds}_eda_summary.md",
                                mime="text/markdown",
                            )
                    with col_dl2:
                        if html_path.exists():
                            st.download_button(
                                label="⬇️ Download Profile HTML",
                                data=html_path.read_bytes(),
                                file_name=f"{ds}_profile.html",
                                mime="text/html",
                            )

                    # Optional: build index immediately
                    if build_now:
                        ds_name, n = build_index(str(art_dir))
                        st.success(f"Index built for **{ds_name}** with **{n}** chunks.")
                        st.caption(f"Next: Tab 3 → set Dataset name = **{ds_name}**")
            except Exception as e:
                st.exception(e)

    else:  # Pick server CSV
        st.caption(f"Server CSV base: `{BASE_DATA_DIR}`")
        csv_paths = list_csvs(BASE_DATA_DIR)
        if not csv_paths:
            st.warning("No CSV files found on server.")
        else:
            rel_options = [str(p.relative_to(BASE_DATA_DIR)) for p in csv_paths]
            choice = st.selectbox("Choose dataset on server", rel_options, index=0)

            if st.button("Run EDA on server CSV", type="primary"):
                try:
                    chosen = (BASE_DATA_DIR / choice).resolve()
                    if not str(chosen).startswith(str(BASE_DATA_DIR)):
                        st.error("Invalid path.")
                    else:
                        summary = run_eda(str(chosen), target=target or None)
                        ds = chosen.stem
                        art_dir = Path("data/artifacts") / ds
                        st.success(f"EDA complete for **{ds}**. Artifacts: {art_dir}")
                        st.json(summary.get("overview", {}))
                        if summary.get("warnings"):
                            st.caption("Warnings")
                            for w in summary["warnings"]:
                                st.warning(w)

                        # ⬇️ Download buttons for MD / HTML
                        md_path = art_dir / "eda_summary.md"
                        html_path = art_dir / "profile.html"
                        st.divider()
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            if md_path.exists():
                                st.download_button(
                                    label="⬇️ Download EDA Markdown",
                                    data=md_path.read_text(encoding="utf-8"),
                                    file_name=f"{ds}_eda_summary.md",
                                    mime="text/markdown",
                                )
                        with col_dl2:
                            if html_path.exists():
                                st.download_button(
                                    label="⬇️ Download Profile HTML",
                                    data=html_path.read_bytes(),
                                    file_name=f"{ds}_profile.html",
                                    mime="text/html",
                                )

                except AssertionError as e:
                    st.error(str(e))
                except Exception as e:
                    st.exception(e)

# =============== Tab 2: Build Index ===============
with tab2:
    st.subheader("Build a vector index from EDA artifacts")
    artifacts_dir = st.text_input(
        "Artifacts directory (e.g., data/artifacts/medical_insurance)",
        value="data/artifacts/medical_insurance"
    )
    if st.button("Build Index"):
        try:
            ds, n = build_index(artifacts_dir)
            st.success(f"Index built for **{ds}** with **{n}** chunks.")
            st.caption(f"Saved under: {Path(cfg.vector_dir) / ds}")
        except AssertionError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)

# =============== Tab 3: Ask Questions ===============
with tab3:
    st.subheader("Ask grounded questions about your dataset")
    dataset_name = st.text_input("Dataset name (folder under storage/vectors/)", value="medical_insurance")
    q = st.text_input("Your question", value="Which features have high missing values?")
    k = st.slider("Top-k chunks", min_value=2, max_value=10, value=5)

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Preview retrieved chunks"):
            try:
                hits = retrieve_relevant_chunks(dataset_name, q, k=k)
                if not hits:
                    st.info("No hits. Did you build the index for this dataset?")
                for h in hits:
                    meta = h["record"]["metadata"]
                    st.markdown(f"**Score:** {h['score']:.3f} — **Section:** {meta.get('section','')}")
                    with st.expander(meta.get('source', 'source') + " — snippet"):
                        st.write(h["record"]["text"])
            except Exception as e:
                st.exception(e)

    with colB:
        if st.button("Answer", type="primary"):
            try:
                ans = answer(q, dataset_name=dataset_name, k=k)
                st.markdown("### Answer")
                st.markdown(ans)
            except Exception as e:
                st.exception(e)