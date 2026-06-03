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

import cuda_copy_scaling_benchmark as bench
import pyccc as pc


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
        include_cofactors=False,
    )
    print(f"Base subset: {base.n_obs} cells x {base.n_vars} genes; groups={len(groups)}; LR={len(lr_filtered)}")

    warmup(base, lr_filtered, args)

    rows: list[dict[str, object]] = []
    results = {}
    for task in ("permutations", "sketches"):
        for backend in ("cpu", "cupy"):
            result, runtime = run_task(base, lr_filtered, args, task=task, backend=backend)
            name = f"{task}_{backend}"
            results[name] = result
            rows.append({"task": task, "backend": backend, **runtime})
            print(f"{name}: {runtime['seconds']:.4f}s, interactions={runtime['interactions']}, fallback={runtime['fallback']}", flush=True)
            bench._clear_gpu_memory()
            gc.collect()

    frame = pd.DataFrame(rows)
    cpu_gpu = frame.pivot(index="task", columns="backend", values="seconds")
    speedups = {task: float(cpu_gpu.loc[task, "cpu"] / cpu_gpu.loc[task, "cupy"]) for task in cpu_gpu.index if cpu_gpu.loc[task, "cupy"] > 0}
    frame["speedup_cpu_over_cupy"] = frame["task"].map(speedups)
    frame.to_csv(out_dir / "gpu_batched_resampling_benchmark.tsv", sep="\t", index=False)

    metric_frame = pd.DataFrame(
        [
            compare_results("permutations", results["permutations_cpu"], results["permutations_cupy"]),
            compare_results("sketches", results["sketches_cpu"], results["sketches_cupy"]),
        ]
    )
    metric_frame.to_csv(out_dir / "gpu_batched_resampling_metrics.tsv", sep="\t", index=False)
    write_summary(out_dir, args, env, base, lr_filtered, frame, metric_frame)
    print(f"Saved outputs to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs CuPy batched permutation and sketch resampling on a 1x CELLxGENE subset.")
    parser.add_argument("--h5ad", default="data/cellxgene/global_celltypist_immune_329k.h5ad")
    parser.add_argument("--cache-dir", default="data/cellxgene/cache")
    parser.add_argument("--out-dir", default="data/cellxgene/gpu_batched_resampling_benchmark")
    parser.add_argument("--species", default="human", choices=["human", "mouse"])
    parser.add_argument("--groupby", default="Predicted_labels_CellTypist")
    parser.add_argument("--gene-symbols-key", default="gene_symbols")
    parser.add_argument("--annotation", action="append", default=["Secreted Signaling"])
    parser.add_argument("--all-annotations", action="store_true")
    parser.add_argument("--n-groups", type=int, default=91)
    parser.add_argument("--max-per-group", type=int, default=20)
    parser.add_argument("--min-pct", type=float, default=0.05)
    parser.add_argument("--min-expr", type=float, default=0.0)
    parser.add_argument("--n-permutations", type=int, default=8)
    parser.add_argument("--downsample-per-group", type=int, default=10)
    parser.add_argument("--downsample-repeats", type=int, default=8)
    parser.add_argument("--score-method", default="cellchat", choices=["cellchat", "sqrt"])
    parser.add_argument("--random-state", type=int, default=123)
    return parser.parse_args()


def warmup(adata, lr, args: argparse.Namespace) -> None:
    print("Warming up CuPy batched resampling kernels...", flush=True)
    try:
        run_task(adata, lr, args, task="permutations", backend="cupy")
    finally:
        bench._clear_gpu_memory()
        gc.collect()


def run_task(adata, lr, args: argparse.Namespace, *, task: str, backend: str):
    kwargs = {
        "gene_symbols_key": args.gene_symbols_key,
        "min_pct": args.min_pct,
        "min_expr": args.min_expr,
        "aggregate": "mean",
        "score_method": args.score_method,
        "random_state": args.random_state,
        "array_backend": backend,
    }
    if task == "permutations":
        kwargs["n_permutations"] = args.n_permutations
    elif task == "sketches":
        kwargs["downsample_per_group"] = args.downsample_per_group
        kwargs["downsample_repeats"] = args.downsample_repeats
    else:
        raise ValueError(f"Unknown task: {task}")

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pc.compute_communication(adata, args.groupby, lr, **kwargs)
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


def compare_results(task: str, cpu, cupy) -> dict[str, object]:
    key_cols = ["source", "target", "ligand", "receptor", "pathway"]
    compare_cols = ["prob"]
    if task == "permutations":
        compare_cols.append("pvalue")
    else:
        compare_cols.extend(["prob_std", "stability", "sketch_present"])
    cpu_frame = cpu.interactions.set_index(key_cols)[compare_cols].sort_index()
    cupy_frame = cupy.interactions.set_index(key_cols)[compare_cols].sort_index()
    idx = cpu_frame.index.union(cupy_frame.index)
    cpu_aligned = cpu_frame.reindex(idx, fill_value=0.0)
    cupy_aligned = cupy_frame.reindex(idx, fill_value=0.0)
    diff = (cpu_aligned - cupy_aligned).abs()
    return {
        "task": task,
        "n_union": len(idx),
        "prob_pearson": corr(cpu_aligned["prob"].to_numpy(dtype=float), cupy_aligned["prob"].to_numpy(dtype=float)),
        "max_abs_diff": float(diff.to_numpy(dtype=float).max()) if len(idx) else 0.0,
        "same_interactions": bool(cpu_frame.index.equals(cupy_frame.index)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_summary(out_dir: Path, args: argparse.Namespace, env: dict[str, object], adata, lr, runtime: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# GPU batched resampling benchmark",
        "",
        f"- h5ad: `{args.h5ad}`",
        f"- groupby: `{args.groupby}`; groups requested: `{args.n_groups}`; max_per_group: `{args.max_per_group}`",
        f"- cells: `{adata.n_obs}`; genes: `{adata.n_vars}`; LR pairs: `{len(lr)}`",
        f"- all_annotations: `{args.all_annotations}`; annotation_filter: `{args.annotation}`",
        f"- aggregate: `mean`; score_method: `{args.score_method}`",
        f"- n_permutations: `{args.n_permutations}`",
        f"- downsample_per_group: `{args.downsample_per_group}`; downsample_repeats: `{args.downsample_repeats}`",
        f"- cupy: `{env.get('cupy')}`; cudf: `{env.get('cudf')}`; device_count: `{env.get('cupy_device_count')}`",
        "",
        "## Runtime",
        "",
        bench.markdown_table(runtime[["task", "backend", "seconds", "speedup_cpu_over_cupy", "interactions", "prob_sum", "fallback"]]),
        "",
        "## CPU/GPU Agreement",
        "",
        bench.markdown_table(metrics),
    ]
    (out_dir / "gpu_batched_resampling_benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
