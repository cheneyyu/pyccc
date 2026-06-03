from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    decay = _read_all_dataset_table(results_dir, "spatial_validation_distance_decay.tsv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    _panel_workflow(axes[0, 0])
    _panel_topk(axes[0, 1], topk, dataset="artista_axolotl", title="ARTISTA top-K enrichment")
    _panel_distance_decay(axes[0, 2], decay, dataset="artista_axolotl")
    _panel_top_lr(axes[1, 0], summary)
    _panel_topk(axes[1, 1], topk, dataset="sota_soybean", title="SOTA plant validation")
    _panel_comparison(axes[1, 2], comparison)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    _write_legend(output_prefix.with_name(output_prefix.name + "_legend.md"), topk, summary)
    _write_source_tarball(output_prefix.with_name(output_prefix.name + "_source_tables.tar.gz"), results_dir)


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
        "protein FASTA/CDS\n+ spatial AnnData",
        "ESMC-300M\nembeddings",
        "LightGBM\nrole filters",
        "LightGBM\npair ranker",
        "density-prior\nLR table",
        "pyccc CCC\n+ spatial nulls",
    ]
    y = 0.5
    for i, label in enumerate(boxes):
        x = 0.05 + i * 0.155
        ax.text(x, y, label, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#444", lw=0.8), transform=ax.transAxes)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.075, y), xytext=(x + 0.105, y), arrowprops=dict(arrowstyle="->", lw=1), xycoords=ax.transAxes)
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


def _panel_comparison(ax, comparison: pd.DataFrame) -> None:
    ax.set_title("F  baseline comparison", loc="left")
    if comparison.empty:
        _empty(ax, "No comparison table")
        return
    frame = comparison[
        (comparison["kernel"].astype(str) == "exp")
        & (comparison["score_type"].astype(str) == "model_weighted_spatial_ccc_score")
    ].copy()
    if frame.empty:
        _empty(ax, "No exp comparison rows")
        return
    grouped = frame.groupby("validation_strategy", as_index=False)["enrichment_z"].mean().sort_values("enrichment_z")
    ax.barh(grouped["validation_strategy"], grouped["enrichment_z"], color="#4c78a8")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Mean enrichment z-score")


def _empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#777", transform=ax.transAxes)


def _write_legend(path: Path, topk: pd.DataFrame, summary: pd.DataFrame) -> None:
    n_sections = int(topk[["dataset", "section_id"]].drop_duplicates().shape[0]) if not topk.empty else 0
    n_pairs = int(summary[["ligand", "receptor"]].drop_duplicates().shape[0]) if not summary.empty and {"ligand", "receptor"}.issubset(summary.columns) else 0
    path.write_text(
        "# DB-free spatial validation main figure legend\n\n"
        "Predicted LR edges are computational candidates, not experimentally validated biochemical interactions. "
        "Spatial validation is plausibility evidence based on contact and diffusion-style kernels with coordinate, "
        "cell-type, matched-random-LR, and score-permutation null models.\n\n"
        f"Source tables currently include {n_sections} section-level validation units and {n_pairs} unique LR pairs.\n",
        encoding="utf-8",
    )


def _write_source_tarball(path: Path, results_dir: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for tsv in sorted(results_dir.glob("**/*.tsv")):
            tar.add(tsv, arcname=tsv.relative_to(results_dir))


if __name__ == "__main__":
    main()
