from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .analysis import CCCResult, compute_communication
from .database import CellChatDB
from .patterns import _pathway_embedding_coords, _pathway_network_vectors, _remove_isolated_pathways


@dataclass
class DifferentialCCC:
    """Differential cell-cell communication between two CCCResult objects."""

    a: CCCResult
    b: CCCResult
    label_a: str
    label_b: str
    interactions: pd.DataFrame
    network_delta: pd.DataFrame
    network_a: pd.DataFrame
    network_b: pd.DataFrame
    count_delta: pd.DataFrame
    count_a: pd.DataFrame
    count_b: pd.DataFrame
    pathway_changes: pd.DataFrame
    source_target_changes: pd.DataFrame

    @property
    def groups(self) -> list[str]:
        return self.a.groups

    def differential_network(self, *, measure: str = "weight") -> pd.DataFrame:
        """Return a CellChat-style differential network matrix."""

        if measure in {"weight", "prob"}:
            return self.network_delta.copy()
        if measure == "count":
            return self.count_delta.copy()
        raise ValueError("`measure` must be one of: 'weight', 'prob', or 'count'.")


def signaling_changes(
    diff: DifferentialCCC,
    group: str,
    *,
    pathways: list[str] | tuple[str, ...] | None = None,
    exclude_pathways: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return CellChat-style pathway changes for one cell group's roles."""

    if group not in diff.groups:
        raise ValueError(f"`group={group!r}` is not present in the compared cell groups.")
    df = diff.interactions.copy()
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    if exclude_pathways is not None:
        df = df[~df["pathway"].isin(exclude_pathways)]
    pathway_index = pd.Index(sorted(df["pathway"].astype(str).unique()), dtype=str)
    if len(pathway_index) == 0:
        return pd.DataFrame(
            columns=[
                "pathway",
                "outgoing_a",
                "outgoing_b",
                "incoming_a",
                "incoming_b",
                "delta_outgoing",
                "delta_incoming",
                "delta_total",
                "abs_delta_total",
                "direction",
            ]
        )

    out = pd.DataFrame({"pathway": pathway_index})
    for role, mask_col in (("outgoing", "source"), ("incoming", "target")):
        sub = df[df[mask_col].astype(str) == str(group)]
        a = sub.groupby("pathway", observed=True)["prob_a"].sum().reindex(pathway_index, fill_value=0.0)
        b = sub.groupby("pathway", observed=True)["prob_b"].sum().reindex(pathway_index, fill_value=0.0)
        out[f"{role}_a"] = a.to_numpy(dtype=float)
        out[f"{role}_b"] = b.to_numpy(dtype=float)
        out[f"delta_{role}"] = out[f"{role}_a"] - out[f"{role}_b"]
    out["delta_total"] = out["delta_outgoing"] + out["delta_incoming"]
    out["abs_delta_total"] = out["delta_outgoing"].abs() + out["delta_incoming"].abs()
    out["direction"] = np.where(out["delta_total"] > 0, diff.label_a, np.where(out["delta_total"] < 0, diff.label_b, "unchanged"))
    return out.sort_values("abs_delta_total", ascending=False).reset_index(drop=True)


def rank_pathway_similarity(
    diff: DifferentialCCC,
    *,
    similarity: str = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
    method: str = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    remove_isolates: bool = True,
) -> pd.DataFrame:
    """Rank shared pathways by distance in a joint communication manifold."""

    embedding = pairwise_pathway_embedding(
        diff,
        similarity=similarity,
        significant_only=significant_only,
        min_prob=min_prob,
        thresh=thresh,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        remove_isolates=remove_isolates,
    )
    if embedding.empty:
        return pd.DataFrame(
            {
                "pathway": pd.Series(dtype=str),
                "distance": pd.Series(dtype=float),
                "dim1_a": pd.Series(dtype=float),
                "dim2_a": pd.Series(dtype=float),
                "dim1_b": pd.Series(dtype=float),
                "dim2_b": pd.Series(dtype=float),
                "prob_a": pd.Series(dtype=float),
                "prob_b": pd.Series(dtype=float),
                "delta_prob": pd.Series(dtype=float),
                "embedding_method": pd.Series(dtype=str),
            }
        )
    wide = embedding.pivot(index="pathway", columns="condition", values=["dim1", "dim2"])
    shared = sorted(set(wide["dim1"].dropna().index))
    shared = [pathway for pathway in shared if diff.label_a in wide["dim1"].columns and diff.label_b in wide["dim1"].columns and pd.notna(wide.loc[pathway, ("dim1", diff.label_a)]) and pd.notna(wide.loc[pathway, ("dim1", diff.label_b)])]
    summary = diff.pathway_changes.set_index("pathway")
    rows = []
    for pathway in shared:
        a_xy = np.array([wide.loc[pathway, ("dim1", diff.label_a)], wide.loc[pathway, ("dim2", diff.label_a)]], dtype=float)
        b_xy = np.array([wide.loc[pathway, ("dim1", diff.label_b)], wide.loc[pathway, ("dim2", diff.label_b)]], dtype=float)
        rows.append(
            {
                "pathway": pathway,
                "distance": float(np.linalg.norm(a_xy - b_xy)),
                "dim1_a": a_xy[0],
                "dim2_a": a_xy[1],
                "dim1_b": b_xy[0],
                "dim2_b": b_xy[1],
                "prob_a": float(summary.loc[pathway, "prob_a"]) if pathway in summary.index else 0.0,
                "prob_b": float(summary.loc[pathway, "prob_b"]) if pathway in summary.index else 0.0,
                "delta_prob": float(summary.loc[pathway, "delta_prob"]) if pathway in summary.index else 0.0,
                "embedding_method": str(embedding["embedding_method"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)


def pairwise_pathway_embedding(
    diff: DifferentialCCC,
    *,
    similarity: str = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
    method: str = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    remove_isolates: bool = True,
) -> pd.DataFrame:
    """Embed pathway networks from two conditions into one shared 2D space."""

    rows = []
    vectors = []
    for result, label in ((diff.a, diff.label_a), (diff.b, diff.label_b)):
        result_vectors, pathways = _pathway_network_vectors(result, similarity=similarity, significant_only=significant_only, min_prob=min_prob, thresh=thresh)
        for vector, pathway in zip(result_vectors, pathways):
            vectors.append(vector)
            rows.append({"pathway": str(pathway), "condition": label})
    if not vectors:
        return pd.DataFrame({"pathway": pd.Series(dtype=str), "condition": pd.Series(dtype=str), "dim1": pd.Series(dtype=float), "dim2": pd.Series(dtype=float), "prob": pd.Series(dtype=float), "count": pd.Series(dtype=int), "embedding_method": pd.Series(dtype=str)})

    values = np.vstack(vectors)
    names = pd.Index([f"{row['pathway']}::{row['condition']}" for row in rows])
    norms = np.linalg.norm(values, axis=1)
    denom = np.outer(norms, norms)
    sim = np.divide(values @ values.T, denom, out=np.zeros((len(values), len(values)), dtype=float), where=denom > 0)
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    sim_df = pd.DataFrame(sim, index=names, columns=names)
    sim_df = _remove_isolated_pathways(sim_df) if remove_isolates else sim_df
    keep = set(sim_df.index.astype(str))
    rows = [row for row, name in zip(rows, names.astype(str)) if name in keep]
    if sim_df.empty or not rows:
        return pd.DataFrame({"pathway": pd.Series(dtype=str), "condition": pd.Series(dtype=str), "dim1": pd.Series(dtype=float), "dim2": pd.Series(dtype=float), "prob": pd.Series(dtype=float), "count": pd.Series(dtype=int), "embedding_method": pd.Series(dtype=str)})

    dist = np.sqrt(np.clip(2.0 - 2.0 * sim_df.to_numpy(dtype=float), 0.0, None))
    coords, method_used = _pathway_embedding_coords(sim_df, dist, method=method, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    summaries = {
        diff.label_a: diff.a.pathway_summary(significant_only=significant_only).set_index("pathway"),
        diff.label_b: diff.b.pathway_summary(significant_only=significant_only).set_index("pathway"),
    }
    frame = pd.DataFrame(rows)
    frame["dim1"] = coords[:, 0]
    frame["dim2"] = coords[:, 1]
    frame["prob"] = [float(summaries[row["condition"]].loc[row["pathway"], "prob"]) if row["pathway"] in summaries[row["condition"]].index else 0.0 for row in rows]
    frame["count"] = [int(summaries[row["condition"]].loc[row["pathway"], "count"]) if row["pathway"] in summaries[row["condition"]].index else 0 for row in rows]
    frame["embedding_method"] = method_used
    return frame.sort_values(["pathway", "condition"]).reset_index(drop=True)


def compare_communication(
    a: CCCResult,
    b: CCCResult,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    pseudocount: float = 1e-9,
    significant_only: bool = False,
) -> DifferentialCCC:
    """Compare two CCC results.

    `a` is treated as the numerator/new condition and `b` as the
    denominator/reference condition.
    """

    if a.groups != b.groups:
        raise ValueError("Results must have identical ordered cell groups for comparison.")

    label_a = label_a or a.condition or "condition_a"
    label_b = label_b or b.condition or "condition_b"
    left = a.significant() if significant_only else a.interactions
    right = b.significant() if significant_only else b.interactions
    keys = ["source", "target", "ligand", "receptor", "pathway"]

    value_cols = [
        "prob",
        "pvalue",
        "ligand_expr",
        "receptor_expr",
        "prob_std",
        "stability",
        "sketch_present",
        "sketch_repeats",
    ]
    cols = keys + value_cols
    left = left[[c for c in cols if c in left.columns]].rename(columns={col: f"{col}_a" for col in value_cols})
    right = right[[c for c in cols if c in right.columns]].rename(columns={col: f"{col}_b" for col in value_cols})
    merged = left.merge(right, on=keys, how="outer")
    for col in ("pvalue_a", "pvalue_b"):
        if col not in merged.columns:
            merged[col] = np.nan
    for col in ("prob_a", "prob_b", "ligand_expr_a", "ligand_expr_b", "receptor_expr_a", "receptor_expr_b"):
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    merged["delta_prob"] = merged["prob_a"] - merged["prob_b"]
    merged["abs_delta_prob"] = merged["delta_prob"].abs()
    merged["log2fc"] = np.log2((merged["prob_a"] + pseudocount) / (merged["prob_b"] + pseudocount))
    merged["percent_change"] = (merged["prob_a"] - merged["prob_b"]) / (merged["prob_b"].abs() + pseudocount)
    merged["is_new"] = (merged["prob_a"] > 0) & (merged["prob_b"] <= 0)
    merged["is_lost"] = (merged["prob_a"] <= 0) & (merged["prob_b"] > 0)
    merged["direction"] = np.where(merged["delta_prob"] > 0, label_a, np.where(merged["delta_prob"] < 0, label_b, "unchanged"))
    merged = merged.sort_values("abs_delta_prob", ascending=False).reset_index(drop=True)

    net_a = a.network(value="prob", significant_only=significant_only)
    net_b = b.network(value="prob", significant_only=significant_only)
    count_a = a.network(value="count", significant_only=significant_only)
    count_b = b.network(value="count", significant_only=significant_only)
    return DifferentialCCC(
        a=a,
        b=b,
        label_a=label_a,
        label_b=label_b,
        interactions=merged,
        network_delta=net_a - net_b,
        network_a=net_a,
        network_b=net_b,
        count_delta=count_a - count_b,
        count_a=count_a,
        count_b=count_b,
        pathway_changes=_summarize_changes(merged, ["pathway"], label_a=label_a, label_b=label_b, pseudocount=pseudocount),
        source_target_changes=_summarize_changes(merged, ["source", "target"], label_a=label_a, label_b=label_b, pseudocount=pseudocount),
    )


def compare_samples(
    a_adata,
    b_adata,
    groupby: str,
    lr_table: CellChatDB | pd.DataFrame,
    *,
    label_a: str = "sample_a",
    label_b: str = "sample_b",
    align_groups: str = "union",
    pseudocount: float = 1e-9,
    significant_only: bool = False,
    **compute_kwargs,
) -> DifferentialCCC:
    """Compute and compare CCC directly from two AnnData samples.

    `a_adata` is treated as the numerator/new sample and `b_adata` as the
    denominator/reference sample. Any extra keyword arguments are passed to
    `compute_communication` for both samples.
    """

    if align_groups not in {"union", "intersection", "strict"}:
        raise ValueError("`align_groups` must be one of: 'union', 'intersection', or 'strict'.")
    kwargs_a = dict(compute_kwargs)
    kwargs_b = dict(compute_kwargs)
    if "condition" not in kwargs_a and "condition_key" not in kwargs_a:
        kwargs_a["condition"] = label_a
    if "condition" not in kwargs_b and "condition_key" not in kwargs_b:
        kwargs_b["condition"] = label_b
    a = compute_communication(a_adata, groupby, lr_table, **kwargs_a)
    b = compute_communication(b_adata, groupby, lr_table, **kwargs_b)

    if align_groups == "strict":
        return compare_communication(a, b, label_a=label_a, label_b=label_b, pseudocount=pseudocount, significant_only=significant_only)
    a, b = _align_results(a, b, mode=align_groups)
    return compare_communication(a, b, label_a=label_a, label_b=label_b, pseudocount=pseudocount, significant_only=significant_only)


def _summarize_changes(df: pd.DataFrame, keys: list[str], *, label_a: str, label_b: str, pseudocount: float) -> pd.DataFrame:
    summary = (
        df.groupby(keys, observed=True)
        .agg(
            prob_a=("prob_a", "sum"),
            prob_b=("prob_b", "sum"),
            count_a=("prob_a", lambda values: int((values > 0).sum())),
            count_b=("prob_b", lambda values: int((values > 0).sum())),
            new_count=("is_new", "sum"),
            lost_count=("is_lost", "sum"),
        )
        .reset_index()
    )
    summary["delta_prob"] = summary["prob_a"] - summary["prob_b"]
    summary["abs_delta_prob"] = summary["delta_prob"].abs()
    summary["log2fc"] = np.log2((summary["prob_a"] + pseudocount) / (summary["prob_b"] + pseudocount))
    summary["delta_count"] = summary["count_a"] - summary["count_b"]
    summary["direction"] = np.where(summary["delta_prob"] > 0, label_a, np.where(summary["delta_prob"] < 0, label_b, "unchanged"))
    return summary.sort_values(["abs_delta_prob", "delta_count"], ascending=False).reset_index(drop=True)


def _align_results(a: CCCResult, b: CCCResult, *, mode: str) -> tuple[CCCResult, CCCResult]:
    if mode == "union":
        groups = sorted(set(a.groups).union(b.groups))
        return replace(a, groups=groups), replace(b, groups=groups)
    if mode == "intersection":
        groups = [group for group in a.groups if group in set(b.groups)]
        if len(groups) < 2:
            raise ValueError("At least two shared cell groups are required for `align_groups='intersection'`.")
        return replace(a, groups=groups, interactions=_filter_result_groups(a.interactions, groups)), replace(
            b,
            groups=groups,
            interactions=_filter_result_groups(b.interactions, groups),
        )
    raise ValueError("`align_groups` must be one of: 'union', 'intersection', or 'strict'.")


def _filter_result_groups(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    keep = set(groups)
    return df[df["source"].isin(keep) & df["target"].isin(keep)].copy()
