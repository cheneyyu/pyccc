from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import pyccc as pc


DEFAULT_CELL_TYPES = [
    "classical monocyte",
    "non-classical monocyte",
    "regulatory T cell",
    "effector memory CD4-positive, alpha-beta T cell",
    "CD16-positive, CD56-dim natural killer cell, human",
    "CD16-negative, CD56-bright natural killer cell, human",
]


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    subset = load_tissue_subset(
        Path(args.h5ad),
        tissues=(args.tissue_a, args.tissue_b),
        groupby=args.groupby,
        tissue_key=args.tissue_key,
        cell_types=args.cell_type or DEFAULT_CELL_TYPES,
        max_per_group=args.max_per_group,
        random_state=args.random_state,
    )
    print(f"Loaded subset: {subset.n_obs} cells x {subset.n_vars} genes")
    print(pd.crosstab(subset.obs[args.tissue_key], subset.obs[args.groupby]).to_string())
    print_input_profile(subset, gene_symbols_key=args.gene_symbols_key)

    proxy = args.proxy or os.environ.get("PYCCC_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
    db = pc.load_cellchatdb(args.species, cache_dir=args.cache_dir, proxy=proxy)
    sample_a = subset[subset.obs[args.tissue_key].astype(str).to_numpy() == args.tissue_a].copy()
    sample_b = subset[subset.obs[args.tissue_key].astype(str).to_numpy() == args.tissue_b].copy()

    compare_started = time.perf_counter()
    diff = pc.compare_samples(
        sample_a,
        sample_b,
        args.groupby,
        db,
        label_a=args.tissue_a,
        label_b=args.tissue_b,
        align_groups=args.align_groups,
        gene_symbols_key=args.gene_symbols_key,
        min_pct=args.min_pct,
        aggregate=args.aggregate,
        score_method=args.score_method,
        downsample_per_group=args.downsample_per_group,
        downsample_repeats=args.downsample_repeats,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    print(f"Computed differential CCC in {time.perf_counter() - compare_started:.2f}s")
    print(f"{args.tissue_a}: {len(diff.a.interactions)} interactions")
    print(f"{args.tissue_b}: {len(diff.b.interactions)} interactions")
    print(f"merged differential rows: {len(diff.interactions)}")

    save_tables(diff, out_dir)
    save_plotnine_outputs(diff, out_dir)

    print("Top pathway changes:")
    print(diff.pathway_changes.head(12).to_string(index=False))
    print(f"Saved outputs to {out_dir.resolve()}")
    print(f"Total wall time: {time.perf_counter() - started:.2f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backed CELLxGENE blood-vs-lung pyccc benchmark.")
    parser.add_argument("--h5ad", default="data/cellxgene/global_celltypist_immune_329k.h5ad")
    parser.add_argument("--out-dir", default="data/cellxgene/benchmark_blood_lung_gene_symbols")
    parser.add_argument("--cache-dir", default="data/cellxgene/cache")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--species", default="human", choices=["human", "mouse"])
    parser.add_argument("--tissue-key", default="tissue")
    parser.add_argument("--groupby", default="cell_type")
    parser.add_argument("--gene-symbols-key", default="gene_symbols")
    parser.add_argument("--tissue-a", default="lung")
    parser.add_argument("--tissue-b", default="blood")
    parser.add_argument("--cell-type", action="append", help="Cell type to include. Repeat to override the default panel.")
    parser.add_argument("--max-per-group", type=int, default=250)
    parser.add_argument("--downsample-per-group", type=int, default=200)
    parser.add_argument("--downsample-repeats", type=int, default=1)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-pct", type=float, default=0.05)
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--score-method", default="cellchat")
    parser.add_argument("--align-groups", default="strict", choices=["strict", "intersection", "union"])
    parser.add_argument("--random-state", type=int, default=123)
    return parser.parse_args()


def load_tissue_subset(
    path: Path,
    *,
    tissues: tuple[str, str],
    groupby: str,
    tissue_key: str,
    cell_types: list[str],
    max_per_group: int,
    random_state: int,
):
    rng = np.random.default_rng(random_state)
    backed = ad.read_h5ad(path, backed="r")
    try:
        if tissue_key not in backed.obs:
            raise KeyError(f"`{tissue_key}` is not present in adata.obs.")
        if groupby not in backed.obs:
            raise KeyError(f"`{groupby}` is not present in adata.obs.")
        obs_tissues = backed.obs[tissue_key].astype(str).to_numpy()
        obs_groups = backed.obs[groupby].astype(str).to_numpy()
        indices: list[int] = []
        missing: list[tuple[str, str]] = []
        for tissue in tissues:
            for cell_type in cell_types:
                idx = np.flatnonzero((obs_tissues == tissue) & (obs_groups == cell_type))
                if idx.size == 0:
                    missing.append((tissue, cell_type))
                    continue
                indices.extend(rng.choice(idx, size=min(max_per_group, idx.size), replace=False).tolist())
        if missing:
            preview = ", ".join(f"{tissue}/{cell_type}" for tissue, cell_type in missing[:6])
            print(f"Missing tissue/group combinations skipped: {preview}")
        if not indices:
            raise ValueError("No cells matched the requested tissues and cell types.")
        return backed[np.array(sorted(indices)), :].to_memory()
    finally:
        backed.file.close()


def print_input_profile(adata, *, gene_symbols_key: str | None) -> None:
    print(f"var_names head: {list(adata.var_names[:3])}")
    if gene_symbols_key:
        if gene_symbols_key in adata.var:
            print(f"{gene_symbols_key} head: {list(adata.var[gene_symbols_key].astype(str).head(3))}")
        else:
            print(f"{gene_symbols_key}: missing")
    x = adata.X
    data = x.data if hasattr(x, "data") else np.asarray(x).ravel()
    data = np.asarray(data)
    if data.size == 0:
        return
    sample = data if data.size <= 100000 else data[np.linspace(0, data.size - 1, 100000, dtype=int)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return
    integer_like = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-6)))
    qs = np.quantile(finite, [0.01, 0.1, 0.5, 0.9, 0.99])
    print(f"X dtype={data.dtype}; nonzero quantiles={np.round(qs, 4).tolist()}; integer_like_fraction={integer_like:.4g}")
    if integer_like > 0.95 and np.quantile(finite, 0.99) > 20:
        print("Input looks count-like; normalize/log1p into .X or a layer before CCC scoring.")
    else:
        print("Input looks normalized/log-like enough for CCC scoring.")


def save_tables(diff: pc.DifferentialCCC, out_dir: Path) -> None:
    diff.interactions.to_csv(out_dir / "top_lr_diff.tsv", sep="\t", index=False)
    diff.pathway_changes.to_csv(out_dir / "pathway_changes.tsv", sep="\t", index=False)
    diff.source_target_changes.to_csv(out_dir / "source_target_changes.tsv", sep="\t", index=False)
    diff.differential_network(measure="weight").to_csv(out_dir / "network_delta_weight.tsv", sep="\t")
    diff.differential_network(measure="count").to_csv(out_dir / "network_delta_count.tsv", sep="\t")


def save_plotnine_outputs(diff: pc.DifferentialCCC, out_dir: Path) -> None:
    try:
        import pyccc.ggplot as cg
    except ImportError:
        print("plotnine is not installed; skipping ggplot outputs. Install with `uv sync --extra ggplot`.")
        return
    cg.diff_bubble(diff, top_n=35, top_pairs=18, facet_by="pathway", title="CELLxGENE tissue differential LR programs").save(
        out_dir / "cellxgene_diff_bubble.png", width=11.0, height=7.0, dpi=220, verbose=False
    )
    cg.diff_pathway_rank(diff, top_n=24, title="CELLxGENE differential pathway rank").save(
        out_dir / "cellxgene_diff_pathway_rank.png", width=8.5, height=6.0, dpi=220, verbose=False
    )
    cg.diff_source_target_rank(diff, top_n=24, title="CELLxGENE differential source-target rank").save(
        out_dir / "cellxgene_diff_source_target_rank.png", width=8.5, height=6.0, dpi=220, verbose=False
    )


if __name__ == "__main__":
    main()
