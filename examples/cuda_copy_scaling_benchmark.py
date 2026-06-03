from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyccc-matplotlib")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import pyccc as pc
import pyccc.analysis as analysis


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "matplotlib-cache"))

    started = time.perf_counter()
    env = _environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(env, indent=2, sort_keys=True))

    db = pc.load_cellchatdb(args.species, cache_dir=args.cache_dir)
    lr = db.interactions
    if args.annotation and not args.all_annotations:
        lr = lr[lr["annotation"].astype(str).isin(args.annotation)].reset_index(drop=True)

    base, lr_filtered, groups = load_base_dataset(
        Path(args.h5ad),
        lr,
        groupby=args.groupby,
        gene_symbols_key=args.gene_symbols_key,
        n_groups=args.n_groups,
        max_per_group=args.max_per_group,
        random_state=args.random_state,
        include_cofactors=args.cofactor_adjust,
    )
    print(f"Base subset: {base.n_obs} cells x {base.n_vars} LR/cofactor genes; groups={len(groups)}; LR={len(lr_filtered)}")

    warmup_cuda(base, args, lr_filtered)

    rows: list[dict[str, object]] = []
    for scale in selected_scales(args):
        print(f"\n=== scale {scale}x ===", flush=True)
        record: dict[str, object] = {
            "scale": scale,
            "status": "ok",
            "groups": len(groups),
            "lr_pairs": len(lr_filtered),
            "base_cells": base.n_obs,
            "n_vars": base.n_vars,
        }
        copied = None
        try:
            copy_started = time.perf_counter()
            copied = copy_adata(base, scale, groupby=args.groupby, copy_obs=args.copy_obs)
            record["copy_seconds"] = time.perf_counter() - copy_started
            record["n_obs"] = copied.n_obs
            record["nnz"] = int(copied.X.nnz) if sparse.issparse(copied.X) else int(np.count_nonzero(copied.X))
            record["host_maxrss_mb_before_compute"] = _maxrss_mb()
            if args.gpu_warmup_per_scale:
                warmup_started = time.perf_counter()
                run_once(copied, lr_filtered, args, backend="cupy")
                record["cupy_warmup_seconds"] = time.perf_counter() - warmup_started
                _clear_gpu_memory()
                gc.collect()
            for backend in ("cpu", "cupy"):
                result = run_once(copied, lr_filtered, args, backend=backend)
                record.update({f"{backend}_{key}": value for key, value in result.items()})
                _clear_gpu_memory()
                gc.collect()
            cpu_seconds = float(record["cpu_seconds"])
            cupy_seconds = float(record["cupy_seconds"])
            record["speedup_cpu_over_cupy"] = cpu_seconds / cupy_seconds if cupy_seconds > 0 else np.nan
            print(
                f"{copied.n_obs:,} cells: CPU {cpu_seconds:.4f}s, CuPy {cupy_seconds:.4f}s, "
                f"speedup {record['speedup_cpu_over_cupy']:.2f}x",
                flush=True,
            )
        except Exception as exc:
            record["status"] = "oom" if is_oom(exc) else "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc(limit=8)
            print(f"{record['status']}: {type(exc).__name__}: {exc}", flush=True)
            rows.append(record)
            write_outputs(rows, out_dir, args, env, elapsed=time.perf_counter() - started)
            if record["status"] in {"oom", "failed"}:
                break
        else:
            rows.append(record)
            write_outputs(rows, out_dir, args, env, elapsed=time.perf_counter() - started)
        finally:
            del copied
            _clear_gpu_memory()
            gc.collect()

    write_outputs(rows, out_dir, args, env, elapsed=time.perf_counter() - started)
    print(f"\nSaved benchmark outputs to {out_dir.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pyccc CPU vs CuPy by physically copying a real AnnData subset.")
    parser.add_argument("--h5ad", default="data/cellxgene/global_celltypist_immune_329k.h5ad")
    parser.add_argument("--cache-dir", default="data/cellxgene/cache")
    parser.add_argument("--out-dir", default="data/cellxgene/cuda_copy_scaling_benchmark")
    parser.add_argument("--species", default="human", choices=["human", "mouse"])
    parser.add_argument("--groupby", default="cell_type")
    parser.add_argument("--gene-symbols-key", default="gene_symbols")
    parser.add_argument("--annotation", action="append", default=["Secreted Signaling"])
    parser.add_argument("--all-annotations", action="store_true", help="Use all CellChatDB annotations instead of filtering to --annotation.")
    parser.add_argument("--n-groups", type=int, default=32)
    parser.add_argument("--max-per-group", type=int, default=100)
    parser.add_argument("--max-scale", type=int, default=128)
    parser.add_argument("--scale", action="append", type=int, help="Explicit scale to run. Repeat to run non-power-of-two scales such as 640.")
    parser.add_argument("--min-pct", type=float, default=0.05)
    parser.add_argument("--min-expr", type=float, default=0.0)
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--clip-quantile", type=float, default=0.99)
    parser.add_argument("--score-method", default="cellchat", choices=["cellchat", "sqrt"])
    parser.add_argument("--cofactor-adjust", action="store_true")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--copy-obs", action="store_true", help="Also duplicate obs rows. Disabled by default to isolate expression scaling.")
    parser.add_argument("--gpu-warmup-per-scale", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_base_dataset(
    h5ad: Path,
    lr: pd.DataFrame,
    *,
    groupby: str,
    gene_symbols_key: str | None,
    n_groups: int,
    max_per_group: int,
    random_state: int,
    include_cofactors: bool,
):
    rng = np.random.default_rng(random_state)
    backed = ad.read_h5ad(h5ad, backed="r")
    try:
        if groupby not in backed.obs:
            raise KeyError(f"`{groupby}` is not present in adata.obs.")
        matrix_names = analysis._matrix_var_names(backed.var, backed.var_names, gene_symbols_key=gene_symbols_key)
        gene_lookup = {gene.upper(): gene for gene in matrix_names.astype(str)}
        lr_filtered = analysis._filter_lr_to_genes(lr, gene_lookup)
        if lr_filtered.empty:
            raise ValueError("No LR pairs remain after filtering to dataset genes.")
        selected_genes = analysis._lr_expression_gene_candidates(lr_filtered, gene_lookup, include_cofactors=include_cofactors)
        gene_mask = np.asarray(matrix_names.isin(selected_genes))
        groups = backed.obs[groupby].astype(str).value_counts().head(n_groups).index.astype(str).tolist()
        labels = backed.obs[groupby].astype(str).to_numpy()
        indices: list[int] = []
        for group in groups:
            idx = np.flatnonzero(labels == group)
            if idx.size:
                indices.extend(rng.choice(idx, size=min(max_per_group, idx.size), replace=False).tolist())
        if not indices:
            raise ValueError("No cells selected for benchmark.")
        subset = backed[np.array(sorted(indices)), gene_mask].to_memory()
    finally:
        backed.file.close()

    present_symbols = matrix_names[gene_mask]
    subset.var[gene_symbols_key or "_pyccc_gene"] = present_symbols.to_numpy()
    if gene_symbols_key is None:
        gene_symbols_key = "_pyccc_gene"
    subset.obs[groupby] = subset.obs[groupby].astype(str)
    return subset, lr_filtered, groups


def warmup_cuda(base, args: argparse.Namespace, lr: pd.DataFrame) -> None:
    print("Warming up CuPy kernels...", flush=True)
    try:
        small = copy_adata(base, 1, groupby=args.groupby, copy_obs=True)
        run_once(small, lr, args, backend="cupy")
    finally:
        _clear_gpu_memory()
        gc.collect()


def copy_adata(base, scale: int, *, groupby: str, copy_obs: bool):
    if scale < 1:
        raise ValueError("scale must be positive")
    x = base.X
    if sparse.issparse(x):
        copied_x = sparse.vstack([x] * scale, format="csr")
    else:
        copied_x = np.tile(np.asarray(x), (scale, 1))
    if copy_obs:
        obs_parts = []
        for i in range(scale):
            part = base.obs.copy()
            part.index = [f"{idx}__copy{i}" for idx in base.obs_names]
            obs_parts.append(part)
        obs = pd.concat(obs_parts, axis=0)
    else:
        labels = np.tile(base.obs[groupby].astype(str).to_numpy(dtype=object), scale)
        obs = pd.DataFrame({groupby: labels}, index=[f"cell{i}" for i in range(copied_x.shape[0])])
    return ad.AnnData(copied_x, obs=obs, var=base.var.copy())


def run_once(adata, lr: pd.DataFrame, args: argparse.Namespace, *, backend: str) -> dict[str, object]:
    _clear_gpu_memory()
    gpu_free_before, gpu_total = _gpu_mem_info()
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
            aggregate=args.aggregate,
            clip_quantile=args.clip_quantile,
            score_method=args.score_method,
            cofactor_adjust=args.cofactor_adjust,
            n_permutations=0,
            array_backend=backend,
        )
    seconds = time.perf_counter() - started
    gpu_free_after, _ = _gpu_mem_info()
    warning_text = " | ".join(str(w.message) for w in caught)
    fallback = "falling back to CPU" in warning_text or "retrying this score on CPU" in warning_text
    return {
        "seconds": seconds,
        "interactions": len(result.interactions),
        "prob_sum": float(result.interactions["prob"].sum()) if not result.interactions.empty else 0.0,
        "warnings": warning_text,
        "fallback": bool(fallback),
        "gpu_free_mb_before": gpu_free_before,
        "gpu_free_mb_after": gpu_free_after,
        "gpu_total_mb": gpu_total,
        "host_maxrss_mb_after": _maxrss_mb(),
    }


def write_outputs(rows: list[dict[str, object]], out_dir: Path, args: argparse.Namespace, env: dict[str, object], *, elapsed: float) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "cuda_copy_scaling_benchmark.tsv", sep="\t", index=False)
    lines = [
        "# CUDA copy-scaling benchmark",
        "",
        f"- h5ad: `{args.h5ad}`",
        f"- groupby: `{args.groupby}`; top groups: `{args.n_groups}`; max_per_group: `{args.max_per_group}`",
        f"- all_annotations: `{args.all_annotations}`; annotation_filter: `{args.annotation}`",
        f"- scales: `{selected_scales(args)}`; aggregate: `{args.aggregate}`; score_method: `{args.score_method}`",
        f"- clip_quantile: `{args.clip_quantile}`",
        f"- gpu_warmup_per_scale: `{args.gpu_warmup_per_scale}`",
        f"- elapsed_seconds: `{elapsed:.2f}`",
        f"- cupy: `{env.get('cupy')}`; cudf: `{env.get('cudf')}`; device_count: `{env.get('cupy_device_count')}`",
        "",
    ]
    if not frame.empty:
        display_cols = [col for col in ["scale", "n_obs", "cpu_seconds", "cupy_seconds", "speedup_cpu_over_cupy", "status", "cpu_interactions", "cupy_interactions"] if col in frame.columns]
        lines.append(markdown_table(frame[display_cols]))
    (out_dir / "cuda_copy_scaling_benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _environment() -> dict[str, object]:
    env: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import cupy as cp
        from cupy_backends.cuda.libs import nvrtc

        env.update(
            {
                "cupy": cp.__version__,
                "cupy_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cupy_driver": cp.cuda.runtime.driverGetVersion(),
                "cupy_device_count": cp.cuda.runtime.getDeviceCount(),
                "cupy_device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                if cp.cuda.runtime.getDeviceCount()
                else "",
                "nvrtc": nvrtc.getVersion(),
            }
        )
    except Exception as exc:
        env["cupy_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import cudf

        env["cudf"] = cudf.__version__
    except Exception as exc:
        env["cudf_error"] = f"{type(exc).__name__}: {exc}"
    return env


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = []
    for _, row in frame.iterrows():
        rows.append([format_cell(row[col]) for col in columns])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _powers_of_two(max_scale: int) -> list[int]:
    scales = []
    scale = 1
    while scale <= max_scale:
        scales.append(scale)
        scale *= 2
    return scales


def selected_scales(args: argparse.Namespace) -> list[int]:
    if args.scale:
        scales = sorted(dict.fromkeys(args.scale))
        if any(scale < 1 for scale in scales):
            raise ValueError("All --scale values must be positive.")
        return scales
    return _powers_of_two(args.max_scale)


def _gpu_mem_info() -> tuple[float | None, float | None]:
    try:
        import cupy as cp

        free, total = cp.cuda.Device(0).mem_info
        return free / 1024**2, total / 1024**2
    except Exception:
        return None, None


def _clear_gpu_memory() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "memoryallocation" in text or "oom" in text


if __name__ == "__main__":
    main()
