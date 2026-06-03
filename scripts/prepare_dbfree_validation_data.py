from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from dbfree_validation_utils import (
    load_manifest,
    manifest_data_dir,
    manifest_results_dir,
    prepared_section_path,
    section_local_path,
    selected_sections,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize spatial AnnData sections for DB-free CCC validation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-normalization", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    data_dir = manifest_data_dir(manifest, args.data_dir)
    results_dir = manifest_results_dir(manifest, args.results_dir)
    qc_rows = []
    path_rows = []
    for section in selected_sections(manifest, section_names=args.section, smoke_only=args.smoke_only):
        adata_path = section_local_path(section, data_dir)
        prepared_path = prepared_section_path(section, results_dir)
        qc = _prepare_one_section(
            manifest,
            section,
            adata_path=adata_path,
            output_path=prepared_path,
            skip_normalization=args.skip_normalization,
        )
        qc_rows.append(qc)
        path_rows.append({"dataset": manifest["name"], "section_id": section["name"], "prepared_path": str(prepared_path)})
    write_tsv(pd.DataFrame(qc_rows), results_dir / "section_qc.tsv")
    write_tsv(pd.DataFrame(path_rows), results_dir / "prepared_paths.tsv")


def _prepare_one_section(
    manifest: dict[str, object],
    section: dict[str, object],
    *,
    adata_path: Path,
    output_path: Path,
    skip_normalization: bool,
) -> dict[str, object]:
    if not adata_path.exists():
        raise FileNotFoundError(f"Missing input h5ad for section {section['name']}: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    spatial_key = str(manifest.get("spatial_key", "spatial"))
    spatial_key_found = _ensure_spatial(adata, spatial_key)
    groupby_column = _choose_groupby(adata, manifest.get("annotation_priority", []))
    adata.obs["pyccc_group"] = adata.obs[groupby_column].astype(str).to_numpy() if groupby_column else "unknown"
    adata.obs["section_id"] = str(section["name"])
    gene_id_key = str(manifest.get("gene_id_key", "gene_id"))
    if gene_id_key in adata.var:
        adata.var["gene_id"] = adata.var[gene_id_key].astype(str).to_numpy()
    else:
        adata.var["gene_id"] = adata.var_names.astype(str)
    raw_count_layer = _raw_count_layer(adata)
    normalization_applied = False
    if not skip_normalization and _looks_like_counts(adata.X):
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        normalization_applied = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return {
        "dataset": str(manifest["name"]),
        "section_id": str(section["name"]),
        "input_path": str(adata_path),
        "prepared_path": str(output_path),
        "n_cells_or_bins": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_groups": int(pd.Series(adata.obs["pyccc_group"].astype(str)).nunique()),
        "spatial_key_found": bool(spatial_key_found),
        "groupby_column_used": groupby_column or "",
        "raw_count_layer_found": raw_count_layer,
        "normalization_applied": bool(normalization_applied),
        "n_expression_genes": int(pd.Index(adata.var["gene_id"].astype(str)).nunique()),
    }


def _ensure_spatial(adata, spatial_key: str) -> bool:
    if spatial_key in adata.obsm:
        adata.obsm["spatial"] = np.asarray(adata.obsm[spatial_key])[:, :2]
        return True
    coord_pairs = [("x", "y"), ("X", "Y"), ("spatial_x", "spatial_y"), ("array_row", "array_col")]
    for x_col, y_col in coord_pairs:
        if x_col in adata.obs and y_col in adata.obs:
            adata.obsm["spatial"] = adata.obs[[x_col, y_col]].astype(float).to_numpy()
            return True
    raise KeyError(f"No spatial coordinates found in obsm[{spatial_key!r}] or common obs coordinate columns.")


def _choose_groupby(adata, priority: object) -> str | None:
    columns = [str(col) for col in adata.obs.columns]
    for col in list(priority or []):
        if str(col) in adata.obs:
            return str(col)
        lowered = {name.lower(): name for name in columns}
        if str(col).lower() in lowered:
            return lowered[str(col).lower()]
    n_obs = max(int(adata.n_obs), 1)
    for col in columns:
        if col.lower().endswith("id") or col.lower() in {"cellid", "cell_id", "barcode"}:
            continue
        values = adata.obs[col]
        n_unique = int(values.astype(str).nunique())
        if n_unique < 2 or n_unique > max(100, int(n_obs * 0.5)):
            continue
        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
            return str(col)
    return None


def _raw_count_layer(adata) -> str:
    for layer in ("counts", "raw_counts", "UMI", "umis"):
        if layer in adata.layers:
            return layer
    return ""


def _looks_like_counts(x) -> bool:
    if sparse.issparse(x):
        data = x.data
        if data.size == 0:
            return False
        values = data[: min(data.size, 10000)]
    else:
        arr = np.asarray(x)
        if arr.size == 0:
            return False
        values = arr.ravel()[: min(arr.size, 10000)]
    values = np.asarray(values, dtype=float)
    if np.nanmin(values) < 0:
        return False
    return bool(np.nanmax(values) > 20 or np.allclose(values, np.round(values)))


if __name__ == "__main__":
    main()
