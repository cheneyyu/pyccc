from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DB-free spatial validation manuscript figure from source TSV tables.")
    parser.add_argument("--results-dir", default="results/dbfree_validation")
    parser.add_argument("--output-prefix", default="figures/dbfree_spatial_validation_main")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    topk = _read_optional(results_dir / "baseline_topk_enrichment.tsv")
    comparison = _read_optional(results_dir / "baseline_comparison.tsv")
    summary = _read_all_dataset_table(results_dir, "spatial_validation_summary.tsv")
    prediction = _read_all_dataset_table(results_dir, "prediction_summary.tsv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    _panel_workflow(axes[0, 0])
    _panel_topk(axes[0, 1], topk, dataset="artista_axolotl", title="B  ARTISTA top-K enrichment")
    _panel_prediction_scale(axes[0, 2], prediction)
    _panel_spatial_example(axes[1, 0], summary, results_dir)
    _panel_topk(axes[1, 1], topk, dataset="sota_soybean", title="E  SOTA plant validation")
    _panel_comparison(axes[1, 2], comparison)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    _write_legend(output_prefix.with_name(output_prefix.name + "_legend.md"), topk, summary)
    source_tarball = output_prefix.with_name(output_prefix.name + "_source_tables.tar.gz")
    _write_source_tarball(source_tarball, results_dir)
    if output_prefix.name.endswith("_main"):
        _write_source_tarball(output_prefix.with_name(output_prefix.name.removesuffix("_main") + "_source_tables.tar.gz"), results_dir)


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


def _read_all_dataset_table(results_dir: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob(f"*/{name}")):
        frames.append(pd.read_csv(path, sep="\t"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _panel_workflow(ax) -> None:
    ax.axis("off")
    boxes = [
        "protein FASTA/CDS\nspatial AnnData",
        "ESMC-300M\nembedding",
        "LightGBM protein\nrole classifiers",
        "LightGBM\npair ranker",
        "clade-aware density\nprior",
        "pyccc CCC\nspatial nulls",
    ]
    for i, label in enumerate(boxes):
        y = 0.88 - i * 0.145
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="#f7f7f7", ec="#444", lw=0.8),
            transform=ax.transAxes,
        )
        if i < len(boxes) - 1:
            ax.annotate("", xy=(0.5, y - 0.07), xytext=(0.5, y - 0.105), arrowprops=dict(arrowstyle="->", lw=1), xycoords=ax.transAxes)
    ax.set_title("A  DB-free CCC validation workflow", loc="left")


def _panel_topk(ax, topk: pd.DataFrame, *, dataset: str, title: str) -> None:
    ax.set_title(title, loc="left")
    if topk.empty:
        _empty(ax, "No top-K table")
        return
    frame = topk[
        (topk["dataset"].astype(str) == dataset)
        & (topk["kernel"].astype(str) == "exp")
        & (topk["score_type"].astype(str) == "model_weighted_spatial_ccc_score")
    ].copy()
    frame = _prefer_matched_random_topk(frame)
    if frame.empty:
        _empty(ax, f"No {dataset} exp top-K rows")
        return
    for strategy, sub in frame.groupby("validation_strategy", sort=False):
        grouped = sub.groupby("k", as_index=False)["top_k_enrichment_z"].mean()
        ax.plot(grouped["k"], grouped["top_k_enrichment_z"], marker="o", label=strategy)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Top-K LR pairs")
    ax.set_ylabel("Enrichment z-score")
    ax.legend(frameon=False, fontsize=8)


def _panel_prediction_scale(ax, prediction: pd.DataFrame) -> None:
    ax.set_title("C  prediction scale and density", loc="left")
    if prediction.empty:
        _empty(ax, "No prediction summary")
        return
    frame = prediction.copy()
    if "dataset" not in frame.columns:
        dataset_names = []
        for _, row in frame.iterrows():
            dataset_names.append("sota_soybean" if "PlantCellChatDB" in str(row.get("density_source_resources", "")) else "artista_axolotl")
        frame["dataset"] = dataset_names
    frame["label"] = frame["dataset"].astype(str).map({"artista_axolotl": "ARTISTA", "sota_soybean": "SOTA"}).fillna(frame["dataset"].astype(str))
    if "selected_pair_count" not in frame.columns:
        frame["selected_pair_count"] = 0
    if "density_prior" not in frame.columns:
        frame["density_prior"] = pd.NA
    frame["selected_pair_count"] = pd.to_numeric(frame["selected_pair_count"], errors="coerce").fillna(0)
    frame["density_prior"] = pd.to_numeric(frame["density_prior"], errors="coerce")
    frame = frame.sort_values("selected_pair_count")
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"]
    ax.barh(frame["label"], frame["selected_pair_count"], color=colors[: len(frame)])
    ax.set_xlabel("Selected predicted LR pairs")
    xmax = max(float(frame["selected_pair_count"].max()) * 1.35, 1.0)
    ax.set_xlim(0, xmax)
    for i, row in enumerate(frame.itertuples(index=False)):
        density = getattr(row, "density_prior", float("nan"))
        text = f"{int(row.selected_pair_count):,} pairs; density {density:.3g}" if pd.notna(density) else f"{int(row.selected_pair_count):,} pairs"
        ax.text(float(row.selected_pair_count) + xmax * 0.02, i, text, va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_distance_decay(ax, decay: pd.DataFrame, *, dataset: str) -> None:
    ax.set_title("C  distance-decay", loc="left")
    if decay.empty:
        _empty(ax, "No distance-decay table")
        return
    frame = decay[decay["dataset"].astype(str) == dataset].copy()
    if frame.empty:
        _empty(ax, f"No {dataset} rows")
        return
    score_col = "model_weighted_spatial_ccc_score" if "model_weighted_spatial_ccc_score" in frame.columns else "spatial_ccc_score"
    frame["distance_mid"] = (pd.to_numeric(frame["distance_min"], errors="coerce") + pd.to_numeric(frame["distance_max"], errors="coerce")) / 2.0
    for strategy, sub in frame.groupby("validation_strategy", sort=False):
        grouped = sub.groupby("distance_mid", as_index=False)[score_col].mean()
        ax.plot(grouped["distance_mid"], grouped[score_col], marker="o", label=strategy)
    ax.set_xlabel("Distance bin midpoint")
    ax.set_ylabel("Mean spatial CCC")
    ax.legend(frameon=False, fontsize=8)


def _panel_top_lr(ax, summary: pd.DataFrame) -> None:
    ax.set_title("D  representative predicted LR", loc="left")
    ax.axis("off")
    if summary.empty:
        _empty(ax, "No LR summary")
        return
    score_col = "model_weighted_spatial_ccc_score" if "model_weighted_spatial_ccc_score" in summary.columns else "spatial_ccc_score"
    frame = summary[summary.get("validation_strategy", pd.Series(dtype=str)).astype(str) == "dbfree"].copy()
    if frame.empty:
        frame = summary.copy()
    top = frame.sort_values(score_col, ascending=False).head(6)
    lines = [f"{row.ligand} -> {row.receptor}   {float(getattr(row, score_col)):.3g}" for row in top.itertuples(index=False)]
    ax.text(0.02, 0.95, "Top spatially enriched LR candidates\n\n" + "\n".join(lines), va="top", ha="left", fontsize=9, transform=ax.transAxes)


def _panel_spatial_example(ax, summary: pd.DataFrame, results_dir: Path) -> None:
    ax.set_title("D  representative spatial LR example", loc="left")
    ax.axis("off")
    example = _select_spatial_example(summary)
    if example is None:
        _empty(ax, "No spatial LR summary")
        return
    prepared = _prepared_lookup(results_dir)
    path = prepared.get((str(example["dataset"]), str(example["section_id"])))
    if path is None or not Path(path).exists():
        _panel_top_lr(ax, summary)
        return
    adata = sc.read_h5ad(path)
    coords = np.asarray(adata.obsm["spatial"])[:, :2]
    ligand = str(example["ligand"])
    receptor = str(example["receptor"])
    panels = [
        ("groups", _group_codes(adata)),
        (_short_gene(ligand), _gene_vector(adata, ligand)),
        (_short_gene(receptor), _gene_vector(adata, receptor)),
    ]
    for i, (title, values) in enumerate(panels):
        subax = ax.inset_axes([0.02 + i * 0.32, 0.23, 0.30, 0.62])
        if values is None:
            _empty(subax, "missing")
            continue
        scatter = subax.scatter(coords[:, 0], coords[:, 1], c=values, s=1.2, cmap="viridis", linewidths=0)
        subax.set_title(title, fontsize=8)
        subax.set_xticks([])
        subax.set_yticks([])
        subax.set_aspect("equal")
        for spine in subax.spines.values():
            spine.set_visible(False)
        if title != "groups":
            colorbar = plt.colorbar(scatter, ax=subax, fraction=0.046, pad=0.01)
            colorbar.ax.tick_params(labelsize=6, length=2)
    score = float(example.get("model_weighted_spatial_ccc_score", np.nan))
    dataset = str(example["dataset"]).replace("_", " ")
    section = str(example["section_id"])
    ax.text(0.02, 0.08, f"{dataset}, {section}: {ligand} -> {receptor}; score={score:.3g}", fontsize=8, ha="left", va="bottom", transform=ax.transAxes)


def _select_spatial_example(summary: pd.DataFrame) -> dict[str, object] | None:
    if summary.empty:
        return None
    required = {"dataset", "section_id", "ligand", "receptor"}
    if not required.issubset(summary.columns):
        return None
    frame = summary.copy()
    if "validation_strategy" in frame:
        dbfree = frame[frame["validation_strategy"].astype(str) == "dbfree"].copy()
        if not dbfree.empty:
            frame = dbfree
    if "kernel" in frame:
        exp = frame[frame["kernel"].astype(str) == "exp"].copy()
        if not exp.empty:
            frame = exp
    score_col = "model_weighted_spatial_ccc_score" if "model_weighted_spatial_ccc_score" in frame.columns else "spatial_ccc_score"
    if score_col not in frame:
        return None
    item = frame.sort_values(score_col, ascending=False).iloc[0].to_dict()
    item["model_weighted_spatial_ccc_score"] = item.get(score_col, np.nan)
    return item


def _prepared_lookup(results_dir: Path) -> dict[tuple[str, str], str]:
    lookup = {}
    for path in sorted(results_dir.glob("*/prepared_paths.tsv")):
        dataset = path.parent.name
        table = pd.read_csv(path, sep="\t")
        for row in table.itertuples(index=False):
            lookup[(dataset, str(row.section_id))] = str(row.prepared_path)
    return lookup


def _group_codes(adata) -> np.ndarray:
    groups = adata.obs["pyccc_group"].astype(str) if "pyccc_group" in adata.obs else pd.Series(["unknown"] * adata.n_obs)
    return pd.Categorical(groups).codes.astype(float)


def _gene_vector(adata, gene: str) -> np.ndarray | None:
    genes = adata.var["gene_id"].astype(str).to_numpy() if "gene_id" in adata.var else adata.var_names.astype(str)
    idx = np.flatnonzero(genes == str(gene))
    if len(idx) == 0:
        return None
    col = adata.X[:, int(idx[0])]
    values = np.asarray(col.toarray()).ravel() if sparse.issparse(col) else np.asarray(col).ravel()
    return values.astype(float)


def _short_gene(gene: str, *, max_len: int = 13) -> str:
    gene = str(gene)
    return gene if len(gene) <= max_len else gene[: max_len - 1] + "."


def _panel_comparison(ax, comparison: pd.DataFrame) -> None:
    ax.set_title("F  baseline comparison", loc="left")
    if comparison.empty:
        _empty(ax, "No comparison table")
        return
    frame = comparison[
        (comparison["kernel"].astype(str) == "exp")
        & (comparison["score_type"].astype(str) == "model_weighted_spatial_ccc_score")
    ].copy()
    frame = _prefer_matched_random_topk(frame)
    if frame.empty:
        _empty(ax, "No exp comparison rows")
        return
    grouped = frame.groupby("validation_strategy", as_index=False)["enrichment_z"].mean().sort_values("enrichment_z")
    ax.barh(grouped["validation_strategy"], grouped["enrichment_z"], color="#4c78a8")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Mean enrichment z-score")


def _empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#777", transform=ax.transAxes)


def _prefer_matched_random_topk(frame: pd.DataFrame) -> pd.DataFrame:
    if "null_model" not in frame.columns:
        return frame
    matched = frame[frame["null_model"].astype(str) == "matched_random_lr"].copy()
    return matched if not matched.empty else frame


def _write_legend(path: Path, topk: pd.DataFrame, summary: pd.DataFrame) -> None:
    n_sections = int(topk[["dataset", "section_id"]].drop_duplicates().shape[0]) if not topk.empty else 0
    n_pairs = int(summary[["ligand", "receptor"]].drop_duplicates().shape[0]) if not summary.empty and {"ligand", "receptor"}.issubset(summary.columns) else 0
    n_permutations = int(pd.to_numeric(topk.get("n_permutations", pd.Series(dtype=float)), errors="coerce").max()) if not topk.empty and "n_permutations" in topk else 0
    section_rows = topk.drop_duplicates(["dataset", "section_id"]) if {"dataset", "section_id"}.issubset(topk.columns) else topk
    n_cells = int(pd.to_numeric(section_rows.get("n_cells_or_bins", pd.Series(dtype=float)), errors="coerce").sum()) if not section_rows.empty and "n_cells_or_bins" in section_rows else 0
    n_groups = int(pd.to_numeric(section_rows.get("n_groups", pd.Series(dtype=float)), errors="coerce").max()) if not section_rows.empty and "n_groups" in section_rows else 0
    null_models = ", ".join(sorted(topk["null_model"].dropna().astype(str).unique())) if "null_model" in topk else "pooled"
    path.write_text(
        "# DB-free spatial validation main figure legend\n\n"
        "Predicted LR edges are computational candidates, not experimentally validated biochemical interactions. "
        "Spatial validation is plausibility evidence based on contact and diffusion-style kernels with coordinate, "
        "cell-type, matched-random-LR, and score-permutation null models. "
        "Top-K enrichment panels use matched-random-LR null rows when available.\n\n"
        f"Source tables currently include {n_sections} section-level validation units, {n_cells} cell/bin records across plotted summaries, "
        f"up to {n_groups} groups per section, {n_pairs} unique LR pairs, {n_permutations} permutations, and null models: {null_models}.\n",
        encoding="utf-8",
    )


def _write_source_tarball(path: Path, results_dir: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for tsv in sorted(results_dir.glob("**/*.tsv")):
            tar.add(tsv, arcname=tsv.relative_to(results_dir))


if __name__ == "__main__":
    main()
