from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyccc-matplotlib")

import numpy as np
import pandas as pd
from scipy import stats

import cuda_copy_scaling_benchmark as bench
import pyccc as pc


KEY_COLS = ["source", "target", "ligand", "receptor", "pathway"]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = bench._environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(env, indent=2, sort_keys=True))

    db = pc.load_cellchatdb(args.species, cache_dir=args.cache_dir)
    lr = db.interactions if args.all_annotations else db.interactions[db.interactions["annotation"].astype(str).isin(args.annotation)].reset_index(drop=True)
    base, lr_filtered, groups = bench.load_base_dataset(
        Path(args.h5ad),
        lr,
        groupby=args.groupby,
        gene_symbols_key=args.gene_symbols_key,
        n_groups=args.n_groups,
        max_per_group=args.max_per_group,
        random_state=args.random_state,
        include_cofactors=args.cofactor_adjust,
    )
    print(f"Base subset: {base.n_obs} cells x {base.n_vars} genes; groups={len(groups)}; LR={len(lr_filtered)}")

    warmup(base, lr_filtered, args)

    runs = [
        ("tri_mean_cpu", "tri_mean", None, "cpu"),
        ("tri_mean_cupy", "tri_mean", None, "cupy"),
        ("clipped_p99_cpu", "clipped_mean", 0.99, "cpu"),
        ("clipped_p99_cupy", "clipped_mean", 0.99, "cupy"),
        ("clipped_p95_cpu", "clipped_mean", 0.95, "cpu"),
        ("clipped_p95_cupy", "clipped_mean", 0.95, "cupy"),
        ("gated_p99_cpu", "gated_mean", 0.99, "cpu"),
        ("gated_p99_cupy", "gated_mean", 0.99, "cupy"),
        ("gated_p95_cpu", "gated_mean", 0.95, "cpu"),
        ("gated_p95_cupy", "gated_mean", 0.95, "cupy"),
    ]

    results = {}
    runtime_rows = []
    for name, aggregate, clip_quantile, backend in runs:
        result, runtime = run_method(base, lr_filtered, args, aggregate=aggregate, clip_quantile=clip_quantile, backend=backend)
        results[name] = result
        runtime_rows.append({"method": name, "aggregate": aggregate, "clip_quantile": clip_quantile or "", "backend": backend, **runtime})
        print(f"{name}: {runtime['seconds']:.4f}s, interactions={runtime['interactions']}, fallback={runtime['fallback']}", flush=True)
        bench._clear_gpu_memory()
        gc.collect()

    runtime_frame = pd.DataFrame(runtime_rows)
    runtime_frame.to_csv(out_dir / "clipped_mean_runtime.tsv", sep="\t", index=False)

    metric_pairs = [
        ("tri_mean_cpu", "clipped_p99_cupy"),
        ("tri_mean_cpu", "clipped_p95_cupy"),
        ("tri_mean_cpu", "gated_p99_cupy"),
        ("tri_mean_cpu", "gated_p95_cupy"),
        ("clipped_p99_cupy", "clipped_p95_cupy"),
        ("gated_p99_cupy", "gated_p95_cupy"),
        ("clipped_p99_cpu", "clipped_p99_cupy"),
        ("clipped_p95_cpu", "clipped_p95_cupy"),
        ("gated_p99_cpu", "gated_p99_cupy"),
        ("gated_p95_cpu", "gated_p95_cupy"),
    ]
    metrics = [pair_metrics(left, right, results[left], results[right]) for left, right in metric_pairs]
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(out_dir / "clipped_mean_pairwise_metrics.tsv", sep="\t", index=False)

    write_summary(out_dir, args, env, base, lr_filtered, runtime_frame, metric_frame)
    print(f"Saved outputs to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tri_mean vs GPU-friendly clipped/gated mean p99/p95 on a 1x CELLxGENE subset.")
    parser.add_argument("--h5ad", default="data/cellxgene/global_celltypist_immune_329k.h5ad")
    parser.add_argument("--cache-dir", default="data/cellxgene/cache")
    parser.add_argument("--out-dir", default="data/cellxgene/clipped_mean_comparison")
    parser.add_argument("--species", default="human", choices=["human", "mouse"])
    parser.add_argument("--groupby", default="Predicted_labels_CellTypist")
    parser.add_argument("--gene-symbols-key", default="gene_symbols")
    parser.add_argument("--annotation", action="append", default=["Secreted Signaling"])
    parser.add_argument("--all-annotations", action="store_true")
    parser.add_argument("--n-groups", type=int, default=91)
    parser.add_argument("--max-per-group", type=int, default=20)
    parser.add_argument("--min-pct", type=float, default=0.05)
    parser.add_argument("--min-expr", type=float, default=0.0)
    parser.add_argument("--score-method", default="cellchat", choices=["cellchat", "sqrt"])
    parser.add_argument("--cofactor-adjust", action="store_true")
    parser.add_argument("--random-state", type=int, default=123)
    return parser.parse_args()


def warmup(adata, lr, args: argparse.Namespace) -> None:
    print("Warming up CuPy clipped/gated mean kernels...", flush=True)
    try:
        run_method(adata, lr, args, aggregate="clipped_mean", clip_quantile=0.99, backend="cupy")
    finally:
        bench._clear_gpu_memory()
        gc.collect()


def run_method(adata, lr, args: argparse.Namespace, *, aggregate: str, clip_quantile: float | None, backend: str):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pc.compute_communication(
            adata,
            args.groupby,
            lr,
            gene_symbols_key=args.gene_symbols_key,
            min_pct=args.min_pct,
            min_expr=args.min_expr,
            aggregate=aggregate,
            clip_quantile=clip_quantile or 0.99,
            score_method=args.score_method,
            cofactor_adjust=args.cofactor_adjust,
            n_permutations=0,
            array_backend=backend,
        )
    seconds = time.perf_counter() - started
    warning_text = " | ".join(str(w.message) for w in caught)
    fallback = "falling back to CPU" in warning_text or "retrying this score on CPU" in warning_text
    runtime = {
        "seconds": seconds,
        "interactions": len(result.interactions),
        "prob_sum": float(result.interactions["prob"].sum()) if not result.interactions.empty else 0.0,
        "warnings": warning_text,
        "fallback": bool(fallback),
    }
    return result, runtime


def pair_metrics(left_name: str, right_name: str, left, right) -> dict[str, object]:
    left_s = indexed_probs(left)
    right_s = indexed_probs(right)
    idx = left_s.index.union(right_s.index)
    a = left_s.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    b = right_s.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    return {
        "left": left_name,
        "right": right_name,
        "n_union": len(idx),
        "pearson": corr(a, b, method="pearson"),
        "spearman": corr(a, b, method="spearman"),
        "max_abs_diff": float(np.max(np.abs(a - b))) if len(idx) else 0.0,
        "top100_overlap": top_overlap(left_s, right_s, 100),
        "top500_overlap": top_overlap(left_s, right_s, 500),
        "network_pearson": matrix_corr(left.network(), right.network()),
        "pathway_pearson": pathway_corr(left.pathway_summary(), right.pathway_summary()),
    }


def indexed_probs(result) -> pd.Series:
    if result.interactions.empty:
        return pd.Series(dtype=float)
    return result.interactions.set_index(KEY_COLS)["prob"]


def corr(a: np.ndarray, b: np.ndarray, *, method: str) -> float:
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    return float(stats.spearmanr(a, b).correlation)


def top_overlap(left: pd.Series, right: pd.Series, n: int) -> float:
    if left.empty and right.empty:
        return 1.0
    left_top = set(left.sort_values(ascending=False).head(n).index)
    right_top = set(right.sort_values(ascending=False).head(n).index)
    denom = max(min(n, len(left_top), len(right_top)), 1)
    return len(left_top & right_top) / denom


def matrix_corr(left: pd.DataFrame, right: pd.DataFrame) -> float:
    idx = left.index.union(right.index)
    cols = left.columns.union(right.columns)
    a = left.reindex(index=idx, columns=cols, fill_value=0.0).to_numpy(dtype=float).ravel()
    b = right.reindex(index=idx, columns=cols, fill_value=0.0).to_numpy(dtype=float).ravel()
    return corr(a, b, method="pearson")


def pathway_corr(left: pd.DataFrame, right: pd.DataFrame) -> float:
    a = left.set_index("pathway")["prob"] if not left.empty else pd.Series(dtype=float)
    b = right.set_index("pathway")["prob"] if not right.empty else pd.Series(dtype=float)
    idx = a.index.union(b.index)
    return corr(a.reindex(idx, fill_value=0.0).to_numpy(dtype=float), b.reindex(idx, fill_value=0.0).to_numpy(dtype=float), method="pearson")


def write_summary(out_dir: Path, args: argparse.Namespace, env: dict[str, object], adata, lr, runtime: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# Clipped Mean Comparison",
        "",
        f"- h5ad: `{args.h5ad}`",
        f"- groupby: `{args.groupby}`; groups requested: `{args.n_groups}`; max_per_group: `{args.max_per_group}`",
        f"- cells: `{adata.n_obs}`; genes: `{adata.n_vars}`; LR pairs: `{len(lr)}`",
        f"- all_annotations: `{args.all_annotations}`; annotation_filter: `{args.annotation}`",
        f"- cupy: `{env.get('cupy')}`; cudf: `{env.get('cudf')}`; device_count: `{env.get('cupy_device_count')}`",
        "",
        "## Runtime",
        "",
        bench.markdown_table(runtime[["method", "seconds", "interactions", "prob_sum", "fallback"]]),
        "",
        "## Pairwise Metrics",
        "",
        bench.markdown_table(metrics),
    ]
    (out_dir / "clipped_mean_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
