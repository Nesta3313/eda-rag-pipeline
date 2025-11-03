from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    # optional, controlled by config
    from ydata_profiling import ProfileReport
    HAS_PROFILING = True
except Exception:
    HAS_PROFILING = False

from .validators import (
    find_id_like_columns,
    find_constant_columns,
    find_high_cardinality_cols,
    find_mixed_type_columns,
    leakage_heuristics,
)

# ---------------------------
# Config loading
# ---------------------------

@dataclass
class EDAConfig:
    raw_data_dir: str
    artifacts_dir: str
    correlation_method: str = "spearman"
    profile_html: bool = False
    max_rows_preview: int = 200
    leakage_check: bool = True

def load_config(path: str = "config/config.yaml") -> EDAConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    eda = cfg.get("eda", {})

    return EDAConfig(
        raw_data_dir=paths.get("raw_data_dir", "data/raw"),
        artifacts_dir=paths.get("artifacts_dir", "data/artifacts"),
        correlation_method=eda.get("correlation_method", "spearman"),
        profile_html=eda.get("profile_html", True),
        max_rows_preview=eda.get("max_rows_preview", 200),
        leakage_check=eda.get("leakage_check", True),
    )

# ---------------------------
# Core EDA helpers
# ---------------------------

def _infer_roles(df: pd.DataFrame) -> Dict[str, str]:
    """
    Assign simple roles: numeric, categorical, datetime.
    (Why: downstream transforms depend on coarse roles, not pandas dtypes alone.)
    """
    roles = {}
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_numeric_dtype(dt):
            roles[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dt):
            roles[col] = "datetime"
        else:
            roles[col] = "categorical"
    return roles

def _basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    mem = df.memory_usage(deep=True).sum()
    duplicates = int(df.duplicated().sum())
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "memory_bytes": int(mem),
        "memory_mb": round(mem / (1024 * 1024), 3),
        "duplicate_rows": duplicates,
        "duplicate_rows_pct": round(duplicates / max(1, df.shape[0]) * 100, 3),
        "columns": df.columns.tolist(),
    }

def _schema(df: pd.DataFrame, roles: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    For each column: dtype, role, example values, nunique, missing%.
    (Why: compact, highly usable in RAG answers.)
    """
    schema = []
    for col in df.columns:
        col_data = df[col]
        missing_pct = float(col_data.isna().mean() * 100)
        nunique = int(col_data.nunique(dropna=True))
        ex_vals = col_data.dropna().unique()[:5]
        ex_vals = [str(v) for v in ex_vals]
        schema.append({
            "name": col,
            "dtype": str(col_data.dtype),
            "role": roles[col],
            "n_unique": nunique,
            "missing_pct": round(missing_pct, 3),
            "examples": ex_vals,
        })
    return schema

def _missingness(df: pd.DataFrame) -> Dict[str, Any]:
    missing_by_col = df.isna().mean().sort_values(ascending=False) * 100
    rows_with_any_missing = int(df.isna().any(axis=1).sum())
    return {
        "rows_with_any_missing": rows_with_any_missing,
        "rows_with_any_missing_pct": round(rows_with_any_missing / max(1, len(df)) * 100, 3),
        "missing_by_column_pct": missing_by_col.round(3).to_dict(),
        "columns_with_missing": [c for c, v in missing_by_col.items() if v > 0],
    }

def _numeric_descriptives(df: pd.DataFrame) -> Dict[str, Any]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return {"numeric_columns": [], "stats": {}}
    desc = num_df.describe().to_dict()
    # add skew, kurtosis
    skew = num_df.skew(numeric_only=True).to_dict()
    kurt = num_df.kurtosis(numeric_only=True).to_dict()
    for col in num_df.columns:
        desc.setdefault(col, {})
        desc[col]["skew"] = float(skew.get(col, np.nan))
        desc[col]["kurtosis"] = float(kurt.get(col, np.nan))
    return {
        "numeric_columns": num_df.columns.tolist(),
        "stats": {col: {k: (float(v) if pd.notna(v) else None) for k, v in stats.items()}
                  for col, stats in desc.items()}
    }

def _categorical_descriptives(df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
    cat_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, UTC]"]).columns
    out: Dict[str, Any] = {"categorical_columns": list(cat_cols), "value_counts_topk": {}}
    for col in cat_cols:
        vc = df[col].astype("string").value_counts(dropna=True).head(top_k)
        out["value_counts_topk"][col] = vc.to_dict()
    return out

def _correlations(df: pd.DataFrame, method: str) -> Dict[str, Any]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {"method": method, "pairs": []}
    corr = num_df.corr(method=method)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c = corr.iloc[i, j]
            if pd.isna(c):
                continue
            pairs.append({"col_x": cols[i], "col_y": cols[j], "corr": float(c)})
    # sort strongest absolute
    pairs.sort(key=lambda d: abs(d["corr"]), reverse=True)
    return {"method": method, "pairs": pairs[:100]}  # cap for brevity

# ---------------------------
# Public API
# ---------------------------

def run_eda(csv_path: str, target: Optional[str] = None, config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Read CSV, compute compact EDA summaries, write JSON + Markdown (+ optional HTML profile).
    Returns the in-memory summary dict for convenience.

    Why this signature:
    - csv_path: explicit per-run selection (no hidden magic).
    - target: optional; if provided we do sharper leakage checks.
    - config_path: lets you swap configs in tests.
    """
    cfg = load_config(config_path)
    csv_path = Path(csv_path)
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    # try to parse datetimes quickly
    # Try to coerce likely date/time columns safely (no deprecated args)
    LIKELY_DT = ("date", "time", "timestamp", "datetime")

    for col in df.columns:
        name = col.lower()
        if any(k in name for k in LIKELY_DT):
            s = df[col]
            # Only attempt parse on text-like columns to avoid clobbering true numerics
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                # First: generic parse, coerce bad values to NaT (safer than silent ignore)
                parsed = pd.to_datetime(s, utc=False, errors="coerce")

                # If nothing parsed, try a few explicit common formats
                if parsed.isna().all():
                    for fmt in ("%Y-%m-%d",
                                "%m/%d/%Y",
                                "%Y-%m-%d %H:%M:%S",
                                "%m/%d/%Y %H:%M",
                                "%d-%b-%Y"):
                        try:
                            parsed = pd.to_datetime(s, format=fmt, utc=False, errors="coerce")
                            if not parsed.isna().all():
                                break
                        except Exception:
                            pass

                # Adopt parsed series if at least some values succeeded
                if not parsed.isna().all():
                    df[col] = parsed

    roles = _infer_roles(df)

    # Collect facts
    overview = _basic_overview(df)
    schema = _schema(df, roles)
    missing = _missingness(df)
    num_desc = _numeric_descriptives(df)
    cat_desc = _categorical_descriptives(df)
    corrs = _correlations(df, method=cfg.correlation_method)

    # Warnings/validators (why: these are gold in practice)
    constant_cols = find_constant_columns(df)
    id_like_cols = find_id_like_columns(df)
    high_card_cols = find_high_cardinality_cols(df)
    mixed_type_cols = find_mixed_type_columns(df)

    warnings: List[str] = []
    if constant_cols:
        warnings.append(f"Constant columns (drop candidates): {constant_cols}")
    if high_card_cols:
        warnings.append(f"High-cardinality categorical columns (consider encoding or hashing): {high_card_cols}")
    if mixed_type_cols:
        warnings.append(f"Columns with mixed types (clean/standardize): {mixed_type_cols}")
    if id_like_cols:
        warnings.append(f"ID-like columns (avoid as features): {id_like_cols}")

    leakage: Dict[str, Any] = {}
    if cfg.leakage_check:
        leakage = leakage_heuristics(df, target)

    # Assemble summary
    dataset_name = csv_path.stem
    out_dir = Path(cfg.artifacts_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "meta": {
            "dataset_name": dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),

    "source_csv": str(csv_path),
            "n_rows": overview["n_rows"],
            "n_columns": overview["n_columns"],
        },
        "overview": overview,
        "schema": schema,
        "missingness": missing,
        "descriptives": {
            "numeric": num_desc,
            "categorical": cat_desc
        },
        "correlations": corrs,
        "validators": {
            "constant_columns": constant_cols,
            "id_like_columns": id_like_cols,
            "high_cardinality_columns": high_card_cols,
            "mixed_type_columns": mixed_type_cols,
        },
        "leakage_checks": leakage,
        "warnings": warnings,
        "config_used": {
            "correlation_method": cfg.correlation_method,
            "profile_html": cfg.profile_html,
            "max_rows_preview": cfg.max_rows_preview,
            "leakage_check": cfg.leakage_check
        }
    }

    # Write JSON
    json_path = out_dir / "eda_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write Markdown (human-readable)
    md_path = out_dir / "eda_summary.md"
    _write_markdown(summary, md_path, preview_rows=cfg.max_rows_preview, df=df)

    # Optional heavy HTML profile
    if cfg.profile_html and HAS_PROFILING:
        try:
            prof = ProfileReport(df, title=f"Profile: {dataset_name}", explorative=True, minimal=True)
            prof.to_file(out_dir / "profile.html")
        except Exception as e:
            warnings.append(f"Profile generation failed: {e}")

    return summary

def _write_markdown(summary: Dict[str, Any], md_path: Path, preview_rows: int, df: pd.DataFrame) -> None:
    lines: List[str] = []
    meta = summary["meta"]
    lines.append(f"# EDA Summary — {meta['dataset_name']}")
    lines.append("")
    lines.append(f"- Generated at: {meta['created_at']}")
    lines.append(f"- Rows: {meta['n_rows']}, Columns: {meta['n_columns']}")
    lines.append(f"- Source: `{meta['source_csv']}`")
    lines.append("")

    # Warnings
    if summary.get("warnings"):
        lines.append("## ⚠️ Warnings")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    # Schema
    lines.append("## Schema")
    lines.append("| Column | Role | Dtype | Unique | Missing % | Examples |")
    lines.append("|---|---|---|---:|---:|---|")
    for col in summary["schema"]:
        lines.append(f"| {col['name']} | {col['role']} | {col['dtype']} | {col['n_unique']} | {col['missing_pct']} | {', '.join(col['examples'])} |")
    lines.append("")

    # Missingness
    miss = summary["missingness"]
    lines.append("## Missingness")
    lines.append(f"- Rows with any missing: **{miss['rows_with_any_missing']}** "
                 f"({miss['rows_with_any_missing_pct']}%)")
    if miss["columns_with_missing"]:
        lines.append(f"- Columns with missing: {', '.join(miss['columns_with_missing'])}")
    lines.append("")

    # Numeric descriptives (list top few)
    num = summary["descriptives"]["numeric"]
    if num.get("numeric_columns"):
        lines.append("## Numeric Descriptives (selected)")
        for col in num["numeric_columns"][:10]:
            stats = num["stats"].get(col, {})
            lines.append(f"- **{col}**: mean={stats.get('mean')}, std={stats.get('std')}, "
                         f"min={stats.get('min')}, max={stats.get('max')}, "
                         f"skew={stats.get('skew')}, kurtosis={stats.get('kurtosis')}")
        lines.append("")

    # Categorical top-k
    cat = summary["descriptives"]["categorical"]
    if cat.get("categorical_columns"):
        lines.append("## Categorical Value Counts (top-k)")
        for col in cat["categorical_columns"][:10]:
            vc = cat["value_counts_topk"].get(col, {})
            if vc:
                lines.append(f"- **{col}**: {', '.join([f'{k}={v}' for k, v in vc.items()])}")
        lines.append("")

    # Correlations (top 15 by |corr|)
    corr_pairs = summary["correlations"]["pairs"][:15]
    if corr_pairs:
        lines.append(f"## Correlations ({summary['correlations']['method']})")
        for p in corr_pairs:
            lines.append(f"- {p['col_x']} ↔ {p['col_y']}: corr={round(p['corr'], 4)}")
        lines.append("")

    # Leakage checks (if any)
    leak = summary.get("leakage_checks") or {}
    if leak:
        lines.append("## Potential Leakage Signals (heuristics)")
        for k, v in leak.items():
            if v:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Preview (first N rows)
    lines.append(f"## Data Preview (first {preview_rows} rows)")
    lines.append("")
    head_df = df.head(preview_rows).copy()
    # limit very wide tables
    if head_df.shape[1] > 20:
        head_df = head_df.iloc[:, :20]
        lines.append("_Note: table truncated to 20 columns for readability._")
    lines.append(head_df.to_markdown(index=False))
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
