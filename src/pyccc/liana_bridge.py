from __future__ import annotations

import pandas as pd

from .analysis import CCCResult


def run_liana(adata, *, groupby: str, resource_name: str = "consensus", **kwargs):
    """Run LIANA's rank aggregate method when LIANA is installed.

    This is a thin optional bridge. It returns LIANA's modified AnnData and
    stores results in `adata.uns["liana_res"]` following LIANA conventions.
    """

    try:
        import liana as li
    except ImportError as exc:
        raise ImportError("Install LIANA support with `uv sync --extra liana`.") from exc

    li.mt.rank_aggregate(adata, groupby=groupby, resource_name=resource_name, **kwargs)
    return adata


def from_liana_results(
    liana_res: pd.DataFrame,
    *,
    groupby: str,
    groups: list[str] | None = None,
    condition: str | None = None,
    score_col: str | None = None,
    pvalue_col: str | None = None,
) -> CCCResult:
    """Convert a LIANA result table into a pyccc CCCResult.

    Expected LIANA columns are mapped flexibly: source/target, ligand/receptor,
    magnitude_rank or score, and specificity_rank or pvalue when available.
    Lower LIANA ranks are converted into larger pyccc probabilities.
    """

    df = liana_res.copy()
    rename = {
        "source": "source",
        "target": "target",
        "ligand_complex": "ligand",
        "receptor_complex": "receptor",
        "ligand": "ligand",
        "receptor": "receptor",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"source", "target", "ligand", "receptor"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LIANA result is missing required mapped columns: {missing}")

    if "pathway" not in df.columns:
        df["pathway"] = "liana"
    if score_col is None:
        for candidate in ("magnitude_rank", "lr_means", "score", "prob"):
            if candidate in df.columns:
                score_col = candidate
                break
    if score_col is None:
        raise ValueError("Could not infer a LIANA score column; pass `score_col=`.")

    score = pd.to_numeric(df[score_col], errors="coerce")
    if score_col.endswith("_rank") or "rank" in score_col:
        df["prob"] = 1.0 / (score + 1e-9)
    else:
        df["prob"] = score
    df["prob"] = df["prob"].fillna(0.0)

    if pvalue_col is None:
        for candidate in ("specificity_rank", "pvalue", "p_value", "padj"):
            if candidate in df.columns:
                pvalue_col = candidate
                break
    df["pvalue"] = pd.to_numeric(df[pvalue_col], errors="coerce") if pvalue_col else pd.NA
    df["annotation"] = df.get("annotation", "")
    df["ligand_expr"] = df.get("ligand_expr", pd.NA)
    df["receptor_expr"] = df.get("receptor_expr", pd.NA)
    df["ligand_pct"] = df.get("ligand_pct", pd.NA)
    df["receptor_pct"] = df.get("receptor_pct", pd.NA)
    df["condition"] = condition or ""

    if groups is None:
        groups = sorted(set(df["source"].astype(str)).union(df["target"].astype(str)))
    out_cols = [
        "source",
        "target",
        "ligand",
        "receptor",
        "pathway",
        "annotation",
        "ligand_expr",
        "receptor_expr",
        "ligand_pct",
        "receptor_pct",
        "prob",
        "pvalue",
        "condition",
    ]
    return CCCResult(df[out_cols].copy(), groupby=groupby, groups=list(groups), condition=condition, lr_name="liana")
