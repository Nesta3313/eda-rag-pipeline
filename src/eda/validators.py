from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

def find_constant_columns(df: pd.DataFrame, tol_unique: int = 1) -> List[str]:
    """Columns with <= tol_unique distinct non-null values (often useless as features)."""
    out = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun <= tol_unique:
            out.append(c)
    return out

def find_id_like_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristic: columns that look like pure identifiers (unique ~ n_rows or match id-like names).
    Why: ID-like features cause leakage or overfit.
    """
    id_cols = []
    patterns = re.compile(r"(id|uuid|guid|ssn|account|user|member|txn|transaction|index)$", re.I)
    n = len(df)
    for c in df.columns:
        if patterns.search(c):
            id_cols.append(c)
            continue
        nun = df[c].nunique(dropna=True)
        if nun > 0.9 * n:
            id_cols.append(c)
    return sorted(list(set(id_cols)))

def find_high_cardinality_cols(df: pd.DataFrame, threshold: int = 100) -> List[str]:
    """
    Categorical columns with many distinct values — often need special encoding/hashing/target encoding.
    """
    out = []
    cat_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in cat_cols:
        nun = df[c].nunique(dropna=True)
        if nun >= threshold:
            out.append(c)
    return out

def find_mixed_type_columns(df: pd.DataFrame) -> List[str]:
    """
    Columns that mix numbers and strings (common in messy CSVs).
    Why: breaks models; must be normalized.
    """
    mixed = []
    for c in df.columns:
        series = df[c]
        # sample a few non-null values
        sample = series.dropna().head(200)
        # skip small columns
        if sample.empty:
            continue
        types = set(type(x).__name__ for x in sample)
        # if looks like numeric but dtype object => mixed formatting (e.g., "1,234" and 1234)
        if len(types) > 1:
            mixed.append(c)
    return mixed

def leakage_heuristics(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Best-effort signals of potential leakage:
    - exact duplicates of target
    - perfect/near-perfect correlation with numeric target
    - columns with names suggesting post-outcome info (e.g., 'outcome_date', 'paid_on', 'diagnosis')
    """
    signals: Dict[str, Any] = {}

    if target and target in df.columns:
        # 1) Exact duplicates of target
        dup_like = []
        tgt = df[target]
        for c in df.columns:
            if c == target:
                continue
            try:
                if df[c].equals(tgt):
                    dup_like.append(c)
            except Exception:
                pass
        if dup_like:
            signals["exact_duplicates_of_target"] = dup_like

        # 2) Near-perfect numeric correlation
        try:
            if pd.api.types.is_numeric_dtype(df[target].dtype):
                num_df = df.select_dtypes(include=[np.number])
                if target in num_df.columns and num_df.shape[1] > 1:
                    corr = num_df.corr(method="pearson")[target].drop(labels=[target])
                    high = corr[abs(corr) >= 0.98].sort_values(ascending=False).index.tolist()
                    if high:
                        signals["near_perfect_corr_with_target"] = high
        except Exception:
            pass

    # 3) Post-outcome name hints (always verify manually)
    post_patterns = re.compile(r"(outcome|paid_on|settled_on|diagnosis|recovered_on|discharged_on|closed_date)", re.I)
    hinted = [c for c in df.columns if post_patterns.search(c)]
    if hinted:
        signals["post_outcome_named_columns"] = hinted

    return signals