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
    parser = argparse.ArgumentParser(description="Build the DB-free spatial validation documentation figure from source TSV tables.")
    parser.add_argument("--results-dir", default="results/dbfree_validation")
    parser.add_argument("--output-prefix", default="figures/dbfree_spatial_validation_main")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    topk = _read_optional(results_dir / "baseline_topk_enrichment.tsv")
    comparison = _read_optional(results_dir / "baseline_comparison.tsv")
    summary = _read_all_dataset_table(results_dir, "spatial_validation_summary.tsv")
    decay = _read_all_dataset_table(results_dir, "spatial_validation_distance_decay.tsv")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    _panel_workflow(axes[0, 0])
    _panel_topk(axes[0, 1], topk, dataset="artista_axolotl", title="B  ARTISTA top-K enrichment")
    _panel_distance_decay(axes[0, 2], decay, dataset="artista_axolotl")
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
    return pd.read_csv(path, sep="\t", low_memory=False) if path.exists() else pd.DataFrame()


def _read_all_dataset_table(results_dir: Path, name: str) -> pd.DataFrame:
    frames = []
    for dataset_dir in _validated_dataset_dirs(results_dir):
        path = dataset_dir / name
        if path.exists():
            frame = pd.read_csv(path, sep="\t", low_memory=False)
            if "dataset" not in frame.columns:
                frame.insert(0, "dataset", dataset_dir.name)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validated_dataset_dirs(results_dir: Path) -> list[Path]:
    return [path.parent for path in sorted(results_dir.glob("*/spatial_validation_summary.tsv"))]


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
    frame = (
        frame.sort_values(["dataset", "selected_pair_count"], ascending=[True, False])
        .drop_duplicates("dataset", keep="first")
        .sort_values("selected_pair_count")
    )
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"]
    y = np.arange(len(frame))
    ax.barh(y, frame["selected_pair_count"], color=colors[: len(frame)])
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("Selected predicted LR pairs")
    xmax = max(float(frame["selected_pair_count"].max()) * 1.35, 1.0)
    ax.set_xlim(0, xmax)
    for i, row in enumerate(frame.itertuples(index=False)):
        density = getattr(row, "density_prior", float("nan"))
        text = f"{int(row.selected_pair_count):,} pairs; density {density:.3g}" if pd.notna(density) else f"{int(row.selected_pair_count):,} pairs"
        x = float(row.selected_pair_count)
        if x > xmax * 0.55:
            ax.text(x - xmax * 0.03, i, text, va="center", ha="right", fontsize=8)
        else:
            ax.text(min(x + xmax * 0.02, xmax * 0.98), i, text, va="center", ha="left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_distance_decay(ax, decay: pd.DataFrame, *, dataset: str) -> None:
    ax.set_title("C  ARTISTA 30DPI distance-decay", loc="left")
    if decay.empty:
        _empty(ax, "No distance-decay table")
        return
    frame = decay[decay["dataset"].astype(str) == dataset].copy()
    if frame.empty:
        _empty(ax, f"No {dataset} rows")
        return
    if "section_id" in frame.columns and "30DPI" in set(frame["section_id"].astype(str)):
        frame = frame[frame["section_id"].astype(str) == "30DPI"].copy()
    if "validation_strategy" in frame.columns:
        keep = ["dbfree", "role_only", "embedding_cosine"]
        focused = frame[frame["validation_strategy"].astype(str).isin(keep)].copy()
        if not focused.empty:
            frame = focused
    score_col = "model_weighted_mean_spatial_ccc_score" if "model_weighted_mean_spatial_ccc_score" in frame.columns else "mean_spatial_ccc_score"
    if score_col not in frame.columns:
        _empty(ax, "No distance-decay score")
        return
    frame["distance_mid"] = (pd.to_numeric(frame["distance_min"], errors="coerce") + pd.to_numeric(frame["distance_max"], errors="coerce")) / 2.0
    strategy_order = ["dbfree", "role_only", "embedding_cosine", "expression_only"]
    strategies = [item for item in strategy_order if item in set(frame["validation_strategy"].astype(str))]
    strategies.extend(sorted(set(frame["validation_strategy"].astype(str)) - set(strategies)))
    for strategy in strategies:
        sub = frame[frame["validation_strategy"].astype(str) == strategy]
        grouped = sub.groupby("distance_mid", as_index=False)[score_col].mean()
        far = float(grouped.sort_values("distance_mid")[score_col].iloc[-1])
        values = grouped[score_col].to_numpy(dtype=float) / far if far > 0 else grouped[score_col].to_numpy(dtype=float)
        ax.plot(grouped["distance_mid"], values, marker="o", label=strategy)
    ax.set_xlabel("Distance bin midpoint")
    ax.set_ylabel("Weighted CCC / far bin")
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
        subax = ax.inset_axes([0.02 + i * 0.28, 0.25, 0.22, 0.56])
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
    if "dataset" in frame:
        artista = frame[frame["dataset"].astype(str) == "artista_axolotl"].copy()
        if not artista.empty:
            frame = artista
    if "section_id" in frame:
        preferred_sections = ["30DPI", "5DPI_1", "Control_Juv"]
        preferred = frame[frame["section_id"].astype(str).isin(preferred_sections)].copy()
        if not preferred.empty:
            preferred["_section_rank"] = preferred["section_id"].astype(str).map({section: i for i, section in enumerate(preferred_sections)}).fillna(len(preferred_sections))
            frame = preferred
    if "kernel" in frame:
        exp = frame[frame["kernel"].astype(str) == "exp"].copy()
        if not exp.empty:
            frame = exp
    score_col = "model_weighted_spatial_ccc_score" if "model_weighted_spatial_ccc_score" in frame.columns else "spatial_ccc_score"
    if score_col not in frame:
        return None
    sort_cols = ["_section_rank", score_col] if "_section_rank" in frame.columns else [score_col]
    ascending = [True, False] if "_section_rank" in frame.columns else [False]
    item = frame.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
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
    results_dir = results_dir.resolve()
    with tarfile.open(path, "w:gz") as tar:
        allowed_dirs = {dataset_dir.resolve() for dataset_dir in _validated_dataset_dirs(results_dir)}
        for tsv in sorted(results_dir.glob("*.tsv")):
            tar.add(tsv, arcname=tsv.relative_to(results_dir))
        for dataset_dir in sorted(allowed_dirs):
            for tsv in sorted(dataset_dir.glob("*.tsv")):
                tar.add(tsv, arcname=tsv.relative_to(results_dir))


if __name__ == "__main__":
    main()
