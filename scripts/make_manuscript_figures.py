from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAPER = DOCS / "paper"
MAIN = PAPER / "figures" / "main"
SUPP = PAPER / "figures" / "supplementary"
SOURCE = PAPER / "source_data"
LEGENDS = PAPER / "legends"
TABLES = PAPER / "tables"
MANIFESTS = PAPER / "manifests"

BLUE = "#2C7FB8"
GREEN = "#41AB5D"
ORANGE = "#D95F0E"
PURPLE = "#756BB1"
RED = "#CB181D"
DARK = "#111827"
GRAY = "#4B5563"
LIGHT = "#E5E7EB"
PALE = "#F8FAFC"


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    ensure_dirs()
    build_source_data()
    if args.check_only:
        return
    selected = list_figures(args.figure)
    for figure in selected:
        FIGURE_BUILDERS[figure]()
    write_legends()
    write_submission_targets()
    write_fallback_document()
    write_artifact_manifest()
    write_qa_contact_sheet()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript figure package from local source tables/assets.")
    parser.add_argument(
        "--figure",
        default="all",
        choices=["all", "1", "2", "3", "4", "5", "6", "supplementary"],
        help="Figure subset to render.",
    )
    parser.add_argument("--check-only", action="store_true", help="Only create/update source-data tables.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    for path in [MAIN, SUPP, SOURCE, LEGENDS, TABLES, MANIFESTS]:
        path.mkdir(parents=True, exist_ok=True)


def list_figures(name: str) -> list[str]:
    if name == "all":
        return ["1", "2", "3", "4", "5", "6", "supplementary"]
    return [name]


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = sanitize_text_cells(frame)
    frame.to_csv(path, sep="\t", index=False)
    strip_trailing_whitespace(path)


def sanitize_text_cells(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(
            lambda value: value.replace("\r", "\\r").replace("\n", "\\n").rstrip() if isinstance(value, str) else value
        )
    return frame


def strip_trailing_whitespace(path: Path) -> None:
    if path.suffix not in {".md", ".svg", ".tsv"}:
        return
    text = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def existing(path: str | Path) -> Path:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def safe_read_tsv(path: str | Path) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, sep="\t", low_memory=False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 320},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        strip_trailing_whitespace(path)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, ha="left", va="top", fontsize=13, weight="bold", color=DARK)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=0.7)
        ax.set_axisbelow(True)


def flow_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str, *, dashed: bool = False) -> None:
    x, y = xy
    w, h = wh
    rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.2, linestyle="--" if dashed else "-")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=DARK, wrap=True)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = GRAY) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, shrinkA=3, shrinkB=3))


def image_panel(ax: plt.Axes, image_path: Path, title: str) -> None:
    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    clean_axis(ax)


def wrapped(ax: plt.Axes, x: float, y: float, text: str, *, width: int = 38, fontsize: float = 8, weight: str | None = None) -> None:
    ax.text(x, y, textwrap.fill(text, width=width), ha="left", va="top", fontsize=fontsize, weight=weight, color=DARK)


def build_source_data() -> None:
    build_figure1_source()
    build_figure2_source()
    build_figure3_source()
    build_figure4_source()
    build_figure5_source()
    build_figure6_source()
    build_supplementary_source()


def build_figure1_source() -> None:
    rows = [
        ("load_cellchatdb", "LR resource", "CellChatDB loader", "curated regular mode"),
        ("load_omnipath_interactions", "LR resource", "OmniPath loader", "external resource"),
        ("load_lr_table", "LR resource", "CSV/TSV/Parquet LR table", "user resource"),
        ("compute_communication", "Regular CCC", "CellChat-like LR/source-target scoring", "core workflow"),
        ("compute_pathway_communication", "Regular CCC", "Pathway aggregation", "core workflow"),
        ("compare_samples", "Differential CCC", "Two-condition differential CCC", "core workflow"),
        ("compare_communication", "Differential CCC", "Compare two CCCResult objects", "core workflow"),
        ("pyccc.plotting", "Visualization", "Matplotlib plotting family", "Python-native figures"),
        ("pyccc.ggplot", "Visualization", "plotnine plotting family", "Python-native figures"),
        ("export_cellchat", "Bridge", "Write CellChat-compatible R export", "optional R plotting"),
        ("predict_lr_dbfree", "DB-free", "Target-species LR table prediction", "optional extension"),
        ("validate_spatial_lr_table", "Spatial validation", "Distance-aware spatial CCC validation", "optional extension"),
    ]
    write_tsv(pd.DataFrame(rows, columns=["api_or_component", "category", "description", "role"]), SOURCE / "figure1_api_inventory.tsv")


def build_figure2_source() -> None:
    rows = [
        ("A", "Differential LR bubble", "docs/figures/differential_lr_bubble.png", "LR-level differential CCC"),
        ("B", "Communication network", "docs/figures/communication_network_gallery.png", "source-target network weights"),
        ("C", "Pathway embedding", "docs/figures/pairwise_pathway_embedding.png", "pathway organization"),
        ("D", "Role heatmap", "docs/figures/role_heatmap_compare_outgoing.png", "outgoing signaling roles"),
        ("E", "Pathway river", "docs/figures/pathway_river_gallery.png", "pattern/pathway flow"),
        ("F", "Spatial network", "docs/figures/spatial_network_gallery.png", "spatial CCC visualization"),
    ]
    frame = pd.DataFrame(rows, columns=["panel", "title", "asset_path", "message"])
    frame["asset_exists"] = frame["asset_path"].map(lambda p: (ROOT / p).exists())
    frame["manuscript_ready"] = frame["asset_exists"]
    write_tsv(frame, SOURCE / "figure2_visual_assets.tsv")


def build_figure3_source() -> None:
    runtime_path = ROOT / "data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core_prepared/runtime.tsv"
    if runtime_path.exists():
        runtime = read_tsv(runtime_path)
    else:
        runtime = read_tsv(ROOT / "docs/paper/results/real1m_runtime.tsv")
        runtime["input_read_seconds"] = 0.0
        runtime["total_with_input_read_seconds"] = runtime["total_seconds"]
    keep = [
        "strategy",
        "scale_label",
        "status",
        "n_cells",
        "n_genes",
        "total_seconds",
        "input_read_seconds",
        "total_with_input_read_seconds",
        "compute_seconds",
        "plot_seconds",
        "export_seconds",
        "r_seconds",
        "peak_cpu_percent",
        "peak_cpu_cores",
        "cpu_sample_count",
        "cpu_affinity_count",
        "cpu_affinity",
        "openblas_num_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "numexpr_num_threads",
        "n_jobs",
    ]
    keep = [c for c in keep if c in runtime.columns]
    runtime = runtime[keep].copy()
    display = {
        "pyccc_python": "pyccc native",
        "pyccc_cellchat_bridge": "pyccc + CellChat R plots",
        "direct_cellchat": "direct CellChat R",
    }
    runtime["display_name"] = runtime["strategy"].map(display).fillna(runtime["strategy"])
    direct = float(runtime.loc[runtime["strategy"].eq("direct_cellchat"), "total_with_input_read_seconds"].iloc[0])
    runtime["speedup_vs_direct"] = direct / runtime["total_with_input_read_seconds"].astype(float)
    write_tsv(runtime, SOURCE / "figure3_runtime_benchmark.tsv")

    parts = []
    for row in runtime.to_dict("records"):
        total = float(row["total_with_input_read_seconds"])
        compute = float(row.get("compute_seconds", 0.0) or 0.0)
        plot = float(row.get("plot_seconds", 0.0) or 0.0)
        export = float(row.get("export_seconds", 0.0) or 0.0)
        input_read = float(row.get("input_read_seconds", 0.0) or 0.0)
        r_seconds = float(row.get("r_seconds", 0.0) or 0.0)
        if row["strategy"] == "direct_cellchat":
            other = max(total - compute - plot, 0.0)
            values = {"MatrixMarket read/object setup": other, "CellChat compute": compute, "R plotting": plot}
        elif row["strategy"] == "pyccc_cellchat_bridge":
            r_plot = max(r_seconds, 0.0)
            values = {"prepared h5ad read": input_read, "pyccc compute": compute, "export": export, "CellChat R plots": r_plot}
        else:
            native_plot = max(plot, 0.0)
            values = {"prepared h5ad read": input_read, "pyccc compute": compute, "Python plotting": native_plot}
        for part, seconds in values.items():
            parts.append({"strategy": row["strategy"], "display_name": row["display_name"], "component": part, "seconds": seconds})
    write_tsv(pd.DataFrame(parts), SOURCE / "figure3_runtime_decomposition.tsv")

    cpu_cols = [c for c in ["strategy", "display_name", "peak_cpu_percent", "peak_cpu_cores", "cpu_sample_count", "cpu_affinity_count", "cpu_affinity", "openblas_num_threads", "omp_num_threads", "mkl_num_threads", "numexpr_num_threads", "n_jobs"] if c in runtime.columns]
    write_tsv(runtime[cpu_cols], SOURCE / "figure3_cpu_affinity.tsv")


def build_figure4_source() -> None:
    base = ROOT / "data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core_prepared"
    meta = base.parent / "human_immune_health_atlas_real1m_raw_4core/cells_1000000/r_input/meta.tsv"
    if meta.exists():
        m = read_tsv(meta)
        comp = m.groupby(["disease", "cell_type"], as_index=False).size().rename(columns={"size": "n_cells"})
    else:
        comp = pd.DataFrame(columns=["disease", "cell_type", "n_cells"])
    write_tsv(comp, SOURCE / "figure4_celltype_composition.tsv")

    export = base / "pyccc_cellchat_bridge/merged_export"
    network = read_tsv(export / "pyccc_cellchat_diff_source_targets.tsv")
    lr = read_tsv(export / "pyccc_cellchat_diff_interactions.tsv")
    pathways = read_tsv(export / "pyccc_cellchat_diff_pathways.tsv")
    write_tsv(network.sort_values("abs_delta_prob", ascending=False), SOURCE / "figure4_differential_network.tsv")
    write_tsv(lr.sort_values("abs_delta_prob", ascending=False), SOURCE / "figure4_differential_lr.tsv")
    write_tsv(pathways.sort_values("abs_delta_prob", ascending=False), SOURCE / "figure4_pathway_rank.tsv")


def model_card_rows(path: Path, model_label: str) -> list[dict[str, object]]:
    card = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    metrics = card.get("metrics", {})
    if "pr_auc" in metrics:
        rows.append({"model": model_label, "metric": "PR-AUC", "method": "DB-free pair ranker", "value": metrics["pr_auc"]})
        for method, value in metrics.get("baseline_pr_auc", {}).items():
            rows.append({"model": model_label, "metric": "PR-AUC", "method": method, "value": value})
        for k, value in metrics.get("top_k_precision", {}).items():
            rows.append({"model": model_label, "metric": f"{k} precision", "method": "DB-free pair ranker", "value": value})
            for method, values in metrics.get("baseline_top_k_precision", {}).items():
                if k in values:
                    rows.append({"model": model_label, "metric": f"{k} precision", "method": method, "value": values[k]})
        for k, value in metrics.get("top_k_enrichment", {}).items():
            rows.append({"model": model_label, "metric": f"{k} enrichment", "method": "DB-free pair ranker", "value": value})
            for method, values in metrics.get("baseline_top_k_enrichment", {}).items():
                if k in values:
                    rows.append({"model": model_label, "metric": f"{k} enrichment", "method": method, "value": values[k]})
    return rows


def build_figure5_source() -> None:
    rows = []
    rows.extend(model_card_rows(ROOT / "models/animal_esmc300m_lgbm_pair_ranker_v0/model_card.json", "animal"))
    rows.extend(model_card_rows(ROOT / "models/universal_esmc300m_lgbm_pair_ranker_v0/model_card.json", "universal"))
    write_tsv(pd.DataFrame(rows), SOURCE / "figure5_model_validation.tsv")

    density_rows = []
    for dataset in ["artista_axolotl", "sota_soybean"]:
        p = ROOT / f"results/dbfree_validation/{dataset}/prediction_summary.tsv"
        if p.exists():
            d = read_tsv(p).iloc[0].to_dict()
            density_rows.append({"dataset": dataset, **d})
    write_tsv(pd.DataFrame(density_rows), SOURCE / "figure5_density_prior.tsv")

    example_rows = []
    columns = [
        "ligand",
        "receptor",
        "model_score",
        "ligand_role_score",
        "receptor_role_score",
        "candidate_strategy",
        "nearest_reference_lr",
        "nearest_reference_species",
        "nearest_reference_resource",
        "warning",
    ]
    for dataset in ["artista_axolotl", "sota_soybean"]:
        p = ROOT / f"results/dbfree_validation/{dataset}/predicted_lr.tsv"
        if p.exists():
            e = read_tsv(p).head(4).copy()
            e.insert(0, "dataset", dataset)
            example_rows.append(e[["dataset", *columns]])
    write_tsv(pd.concat(example_rows, ignore_index=True), SOURCE / "figure5_predicted_lr_examples.tsv")

    role_rows = []
    for label, p in [
        ("animal", ROOT / "models/animal_esmc300m_lgbm_role_classifiers_v0/model_card.json"),
        ("universal", ROOT / "models/universal_esmc300m_lgbm_role_classifiers_v0/model_card.json"),
    ]:
        card = json.loads(p.read_text(encoding="utf-8"))
        for role, metrics in card.get("metrics", {}).items():
            role_rows.append(
                {
                    "model": label,
                    "role": role,
                    "pr_auc": metrics.get("validation_pr_auc"),
                    "baseline_pr_auc": metrics.get("validation_baseline_pr_auc"),
                    "roc_auc": metrics.get("validation_roc_auc"),
                    "top100_precision": metrics.get("validation_top_k_precision", {}).get("top_100"),
                }
            )
    write_tsv(pd.DataFrame(role_rows), SOURCE / "figure5_role_model_validation.tsv")


def build_figure6_source() -> None:
    topk_frames = []
    null_frames = []
    distance_frames = []
    qc_frames = []
    for dataset in ["artista_axolotl", "sota_soybean"]:
        base = ROOT / f"results/dbfree_validation/{dataset}"
        topk = read_tsv(base / "spatial_validation_top_k_enrichment.tsv")
        topk.insert(0, "source_dataset", dataset)
        topk_frames.append(topk)
        null_frames.append(topk[topk["null_model"].isin(["matched_random_lr", "coordinate_permutation", "celltype_permutation", "score_permutation"])].copy())
        decay = read_tsv(base / "spatial_validation_distance_decay.tsv")
        decay.insert(0, "source_dataset", dataset)
        distance_frames.append(decay)
        qc = read_tsv(base / "section_qc.tsv")
        qc.insert(0, "source_dataset", dataset)
        qc_frames.append(qc)
    topk_all = pd.concat(topk_frames, ignore_index=True)
    null_all = pd.concat(null_frames, ignore_index=True)
    distance_all = pd.concat(distance_frames, ignore_index=True)
    distance_all["distance_bin"] = pd.qcut(
        distance_all["mean_distance"].rank(method="first"),
        q=min(8, len(distance_all)),
        labels=False,
        duplicates="drop",
    )
    distance_summary = (
        distance_all.groupby(["source_dataset", "distance_bin"], as_index=False)
        .agg(
            mean_distance=("mean_distance", "mean"),
            model_weighted_mean_spatial_ccc_score=("model_weighted_mean_spatial_ccc_score", "mean"),
            n_observations=("model_weighted_mean_spatial_ccc_score", "size"),
        )
        .sort_values(["source_dataset", "distance_bin"])
    )
    qc_all = pd.concat(qc_frames, ignore_index=True)
    write_tsv(topk_all, SOURCE / "figure6_spatial_topk_enrichment.tsv")
    write_tsv(null_all, SOURCE / "figure6_null_comparison.tsv")
    write_tsv(distance_summary, SOURCE / "figure6_distance_decay.tsv")
    write_tsv(qc_all, SOURCE / "figure6_section_qc.tsv")


def build_supplementary_source() -> None:
    coverage_rows = [
        ("network circle", True, True, True, True, True, True),
        ("network heatmap", True, True, True, True, True, True),
        ("chord", True, True, True, False, True, False),
        ("bubble", True, True, True, True, True, True),
        ("pathway heatmap", True, True, True, True, True, True),
        ("LR contribution", True, True, True, False, True, False),
        ("role scatter", True, True, True, False, True, False),
        ("role heatmap", True, True, True, True, True, True),
        ("rank signaling", True, True, True, True, True, True),
        ("differential network", True, True, True, True, True, True),
        ("pathway river", True, True, True, True, True, True),
        ("pattern extraction", True, True, True, True, True, True),
        ("spatial communication", True, True, True, True, True, True),
        ("cofactor-aware scoring", True, False, True, False, True, True),
        ("DE gate", True, False, True, False, True, True),
        ("CellChat R export", True, False, True, True, True, True),
        ("LIANA import/run support", True, False, True, False, True, False),
        ("DB-free LR prediction", True, False, True, True, True, True),
        ("spatial validation", True, True, True, True, True, True),
    ]
    write_tsv(
        pd.DataFrame(
            coverage_rows,
            columns=["feature", "pyccc_api_exists", "python_plot_exists", "source_table_exists", "example_asset_exists", "tested", "manuscript_ready"],
        ),
        SOURCE / "supplementary_figure1_visual_coverage.tsv",
    )
    repro = pd.DataFrame(
        [
            ("pytest -q", "78 passed, 4 skipped, 2 warnings", "local"),
            ("sphinx-build -W -b html docs docs/_build/html", "passed", "local"),
            ("uv build", "sdist and wheel built", "local"),
            ("GitHub Actions", "success on Python 3.10 and 3.11", "https://github.com/cheneyyu/pyccc/actions/runs/26989277542"),
        ],
        columns=["gate", "result", "evidence"],
    )
    write_tsv(repro, SOURCE / "supplementary_figure2_reproducibility.tsv")

    sens_frames = []
    for p in sorted((ROOT / "data/runtime_benchmark/human_immune_health_atlas_real1m").glob("cells_*/runtime.tsv")):
        f = read_tsv(p)
        f.insert(0, "source_path", str(p.relative_to(ROOT)))
        if "error" in f.columns:
            f["error_summary"] = f["error"].fillna("").map(lambda value: "failed during local sensitivity run" if value else "")
            f = f.drop(columns=["error"])
        sens_frames.append(f)
    if sens_frames:
        write_tsv(pd.concat(sens_frames, ignore_index=True), SOURCE / "supplementary_figure3_benchmark_sensitivity.tsv")
    else:
        write_tsv(pd.DataFrame(columns=["source_path"]), SOURCE / "supplementary_figure3_benchmark_sensitivity.tsv")


def figure1() -> None:
    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_label(ax_a, "A")
    ax_a.set_title("Problem setting", loc="left", weight="bold")
    flow_box(ax_a, (0.05, 0.70), (0.36, 0.14), "single-cell or spatial AnnData", BLUE)
    flow_box(ax_a, (0.05, 0.46), (0.36, 0.14), "cell labels, conditions, coordinates", BLUE)
    flow_box(ax_a, (0.05, 0.22), (0.36, 0.14), "CCC inference and differential analysis", GREEN)
    arrow(ax_a, (0.23, 0.70), (0.23, 0.60))
    arrow(ax_a, (0.23, 0.46), (0.23, 0.36))
    wrapped(ax_a, 0.52, 0.82, "Pain points: R-centric conversion, dense aggregation bottlenecks, and incomplete LR databases for some target species.", width=32)
    wrapped(ax_a, 0.52, 0.43, "pyccc: AnnData-native CellChat-like scoring, tidy outputs, Python plots, CellChat export, and optional DB-free LR prediction.", width=32)

    panel_label(ax_b, "B")
    ax_b.set_title("Regular-mode pipeline", loc="left", weight="bold")
    labels = [
        ("AnnData + curated LR table", BLUE),
        ("group ligand/receptor/cofactor summaries", GREEN),
        ("CellChat-like probabilities and pathways", GREEN),
        ("LR, source-target, pathway, and differential tables", ORANGE),
        ("Python-native plots and optional R export", PURPLE),
    ]
    y = 0.82
    for text, color in labels:
        flow_box(ax_b, (0.10, y), (0.78, 0.105), text, color)
        if y > 0.20:
            arrow(ax_b, (0.49, y), (0.49, y - 0.07))
        y -= 0.16

    panel_label(ax_c, "C")
    ax_c.set_title("LR resource layer", loc="left", weight="bold")
    resources = ["CellChatDB", "OmniPath", "CSV/TSV/Parquet", "user LR table", "DB-free predicted table"]
    for i, res in enumerate(resources):
        flow_box(ax_c, (0.08, 0.78 - i * 0.13), (0.42, 0.08), res, [BLUE, GREEN, ORANGE, PURPLE, RED][i], dashed=i == 4)
        arrow(ax_c, (0.50, 0.82 - i * 0.13), (0.64, 0.52))
    flow_box(ax_c, (0.62, 0.44), (0.30, 0.16), "CellChatDB-compatible LR table", DARK)
    flow_box(ax_c, (0.62, 0.20), (0.30, 0.12), "compute_communication(...)", GREEN)
    arrow(ax_c, (0.77, 0.44), (0.77, 0.32))

    panel_label(ax_d, "D")
    ax_d.set_title("Optional target-species DB-free branch", loc="left", weight="bold")
    dbfree = [
        "CDS/protein FASTA",
        "longest protein product",
        "ESMC-300M embeddings",
        "LightGBM role classifiers",
        "LightGBM pair ranker",
        "clade-aware density prior",
        "predicted LR table",
        "regular pyccc CCC",
    ]
    xs = [0.04, 0.29, 0.54, 0.79]
    ys = [0.66, 0.38]
    w, h = 0.17, 0.12
    positions = []
    for i, text in enumerate(dbfree):
        row, col = divmod(i, 4)
        x, y = xs[col], ys[row]
        positions.append((x, y))
        color = RED if i < 7 else GREEN
        flow_box(ax_d, (x, y), (w, h), text, color, dashed=i < 7)
        if col < 3:
            arrow(ax_d, (x + w, y + h / 2), (xs[col + 1], y + h / 2))
    arrow(ax_d, (positions[3][0] + w / 2, positions[3][1]), (positions[4][0] + w / 2, positions[4][1] + h))
    wrapped(ax_d, 0.04, 0.20, "Dashed boxes indicate the optional experimental LR-table generator. Regular-mode benchmarks use curated LR tables, not DB-free prediction.", width=100, fontsize=8)

    panel_label(ax_e, "E")
    ax_e.set_title("Minimal API", loc="left", weight="bold")
    code = """import pyccc as pc

db = pc.load_cellchatdb("human")
res = pc.compute_communication(
    adata, groupby="cell_type", lr_table=db)
diff = pc.compare_samples(
    adata_a, adata_b, groupby="cell_type", lr_table=db)

predicted_db = pc.predict_lr_dbfree(
    adata, protein_fasta="target.fa")
report = pc.validate_spatial_lr_table(
    adata, lr_table=predicted_db)"""
    ax_e.text(0.04, 0.90, code, ha="left", va="top", family="monospace", fontsize=7.1, color=DARK)
    save_figure(fig, MAIN / "figure1_overview")


def figure2() -> None:
    assets = read_tsv(SOURCE / "figure2_visual_assets.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    for ax, row, label in zip(axes.ravel(), assets.to_dict("records"), list("ABCDEF")):
        panel_label(ax, label)
        image_panel(ax, ROOT / row["asset_path"], row["title"])
        ax.text(0.02, -0.05, row["message"], transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=GRAY)
    save_figure(fig, MAIN / "figure2_visual_coverage")


def figure3() -> None:
    runtime = read_tsv(SOURCE / "figure3_runtime_benchmark.tsv")
    decomp = read_tsv(SOURCE / "figure3_runtime_decomposition.tsv")
    cpu = read_tsv(SOURCE / "figure3_cpu_affinity.tsv")
    order = ["pyccc_python", "pyccc_cellchat_bridge", "direct_cellchat"]
    colors = {"pyccc_python": BLUE, "pyccc_cellchat_bridge": GREEN, "direct_cellchat": ORANGE}

    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    for ax in [ax_a, ax_d, ax_f]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_label(ax_a, "A")
    ax_a.set_title("Benchmark design", loc="left", weight="bold")
    steps = ["1.82M x 32,357 h5ad", "1M real cells", "1M x 787 LR/cofactor genes", "1,254 LR interactions", "3 matched workflows"]
    y = 0.82
    for step in steps:
        flow_box(ax_a, (0.12, y), (0.76, 0.10), step, BLUE if y > 0.55 else GREEN)
        if y > 0.23:
            arrow(ax_a, (0.50, y), (0.50, y - 0.06))
        y -= 0.145
    wrapped(ax_a, 0.12, 0.13, "Strict 4-core affinity; prepared 1M input; regular mode only.", width=38, fontsize=8)

    panel_label(ax_b, "B")
    ax_b.set_title("Runtime", loc="left", weight="bold")
    r = runtime.set_index("strategy").loc[order].reset_index()
    y = np.arange(len(r))
    ax_b.barh(y, r["total_with_input_read_seconds"], color=[colors[s] for s in r["strategy"]])
    ax_b.set_yticks(y, r["display_name"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("seconds from prepared input")
    style_axis(ax_b, grid="x")
    direct = float(r.loc[r["strategy"].eq("direct_cellchat"), "total_with_input_read_seconds"].iloc[0])
    for i, row in r.iterrows():
        seconds = float(row["total_with_input_read_seconds"])
        ax_b.text(seconds + 5, i, f"{seconds:.1f}s\n{direct / seconds:.2f}x", va="center", fontsize=7)

    panel_label(ax_c, "C")
    ax_c.set_title("Time decomposition", loc="left", weight="bold")
    pivot = decomp.pivot_table(index="display_name", columns="component", values="seconds", aggfunc="sum").fillna(0)
    pivot = pivot.loc[r["display_name"]]
    left = np.zeros(len(pivot))
    comp_colors = [BLUE, GREEN, ORANGE, PURPLE, RED, "#8C6D31"]
    for color, comp in zip(comp_colors, pivot.columns):
        ax_c.barh(np.arange(len(pivot)), pivot[comp], left=left, label=comp, color=color)
        left += pivot[comp].to_numpy()
    ax_c.set_yticks(np.arange(len(pivot)), pivot.index)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("seconds")
    style_axis(ax_c, grid="x")
    ax_c.legend(frameon=False, loc="lower right", fontsize=6)

    panel_label(ax_d, "D")
    ax_d.set_title("Sparse tri-mean optimization", loc="left", weight="bold")
    flow_box(ax_d, (0.08, 0.68), (0.32, 0.12), "Before:\nsparse matrix", ORANGE)
    flow_box(ax_d, (0.57, 0.68), (0.32, 0.12), "dense percentile\npartition path", ORANGE)
    arrow(ax_d, (0.40, 0.74), (0.57, 0.74))
    flow_box(ax_d, (0.08, 0.36), (0.32, 0.12), "After:\nsparse matrix", GREEN)
    flow_box(ax_d, (0.57, 0.36), (0.32, 0.12), "zero-aware\nsparse quantiles", GREEN)
    arrow(ax_d, (0.40, 0.42), (0.57, 0.42), color=GREEN)
    wrapped(ax_d, 0.08, 0.18, "The speedup comes from exact sparse aggregation, not high worker count.", width=42, fontsize=8)

    panel_label(ax_e, "E")
    ax_e.set_title("Peak CPU", loc="left", weight="bold")
    c = cpu.set_index("strategy").loc[order].reset_index()
    ax_e.bar(np.arange(len(c)), c["peak_cpu_cores"], color=[colors[s] for s in c["strategy"]])
    ax_e.axhline(4, color=GRAY, ls="--", lw=1)
    ax_e.set_xticks(np.arange(len(c)), c["display_name"], rotation=25, ha="right")
    ax_e.set_ylabel("peak CPU cores")
    ax_e.set_ylim(0, 4.2)
    style_axis(ax_e, grid="y")
    for i, row in c.iterrows():
        ax_e.text(i, row["peak_cpu_cores"] + 0.12, f"{row['peak_cpu_cores']:.2f}", ha="center", fontsize=7)

    panel_label(ax_f, "F")
    ax_f.set_title("Timing policy", loc="left", weight="bold")
    wrapped(ax_f, 0.07, 0.88, "Counted: reading the prepared 1M pyccc h5ad; direct CellChat reading MatrixMarket inside R; analysis and matched visualizations.", width=48)
    wrapped(ax_f, 0.07, 0.55, "Not counted: sampling from the full 1.82M h5ad and LR filtering/preparation.", width=48)
    wrapped(ax_f, 0.07, 0.34, "Caption must state: prepared 1M input, strict 4-core affinity, regular mode only.", width=48, weight="bold")
    save_figure(fig, MAIN / "figure3_runtime_scaling")


def short_label(value: str) -> str:
    mapping = {
        "CD14-positive, CD16-negative classical monocyte": "CD14 mono",
        "central memory CD4-positive, alpha-beta T cell": "CM CD4 T",
        "effector memory CD4-positive, alpha-beta T cell": "EM CD4 T",
        "effector memory CD8-positive, alpha-beta T cell": "EM CD8 T",
        "naive thymus-derived CD4-positive, alpha-beta T cell": "Naive CD4 T",
        "cytomegalovirus infection": "CMV",
    }
    return mapping.get(str(value), str(value))


def figure4() -> None:
    comp = read_tsv(SOURCE / "figure4_celltype_composition.tsv")
    network = read_tsv(SOURCE / "figure4_differential_network.tsv")
    lr = read_tsv(SOURCE / "figure4_differential_lr.tsv")
    pathways = read_tsv(SOURCE / "figure4_pathway_rank.tsv")
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    panel_label(ax, "A")
    ax.set_title("1M-cell comparison design", loc="left", weight="bold")
    if not comp.empty:
        pivot = comp.pivot_table(index="disease", columns="cell_type", values="n_cells", fill_value=0)
        pivot = pivot.rename(index=short_label, columns=short_label)
        bottom = np.zeros(len(pivot))
        for i, col in enumerate(pivot.columns):
            ax.bar(np.arange(len(pivot)), pivot[col], bottom=bottom, label=col)
            bottom += pivot[col].to_numpy()
        ax.set_xticks(np.arange(len(pivot)), pivot.index, rotation=0)
        ax.set_ylabel("cells")
        style_axis(ax, grid="y")
        ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    else:
        clean_axis(ax)
        wrapped(ax, 0.05, 0.80, "1M cells; CMV infection vs normal; top five shared cell types.", width=42)

    ax = axes[1]
    panel_label(ax, "B")
    ax.set_title("Differential source-target network", loc="left", weight="bold")
    mat = network.pivot_table(index="source", columns="target", values="delta_prob", fill_value=0)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]), [short_label(x) for x in mat.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]), [short_label(x) for x in mat.index])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="delta prob")

    ax = axes[2]
    panel_label(ax, "C")
    ax.set_title("Top differential LR pairs", loc="left", weight="bold")
    top = lr.head(12).copy()
    labels = (top["ligand"].astype(str) + "->" + top["receptor"].astype(str)).to_list()
    y = np.arange(len(top))
    ax.scatter(top["delta_prob"], y, s=60 * (top["abs_delta_prob"] / top["abs_delta_prob"].max() + 0.2), color=np.where(top["delta_prob"] > 0, RED, BLUE))
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("delta probability")
    style_axis(ax, grid="x")

    ax = axes[3]
    panel_label(ax, "D")
    ax.set_title("Pathway-level differential rank", loc="left", weight="bold")
    p = pathways.sort_values("delta_prob").copy()
    ax.barh(np.arange(len(p)), p["delta_prob"], color=np.where(p["delta_prob"] > 0, RED, BLUE))
    ax.set_yticks(np.arange(len(p)), p["pathway"])
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("delta probability")
    style_axis(ax, grid="x")

    ax = axes[4]
    panel_label(ax, "E")
    ax.set_title("Outgoing role shifts", loc="left", weight="bold")
    out = network.groupby("source", as_index=False)["delta_prob"].sum()
    out = out.sort_values("delta_prob")
    ax.barh(np.arange(len(out)), out["delta_prob"], color=np.where(out["delta_prob"] > 0, RED, BLUE))
    ax.set_yticks(np.arange(len(out)), [short_label(x) for x in out["source"]])
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("sum delta probability")
    style_axis(ax, grid="x")

    ax = axes[5]
    panel_label(ax, "F")
    image_panel(ax, ROOT / "docs/figures/pairwise_pathway_embedding.png", "Pathway embedding")
    save_figure(fig, MAIN / "figure4_differential_case_study")


def figure5() -> None:
    validation = read_tsv(SOURCE / "figure5_model_validation.tsv")
    density = read_tsv(SOURCE / "figure5_density_prior.tsv")
    examples = read_tsv(SOURCE / "figure5_predicted_lr_examples.tsv")
    roles = read_tsv(SOURCE / "figure5_role_model_validation.tsv")
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    panel_label(ax, "A")
    ax.set_title("Training resources", loc="left", weight="bold")
    resources = read_tsv(ROOT / "data/lr_training_resources/resource_summary.tsv")
    summary = resources.groupby(["clade", "resource"], as_index=False)["normalized_rows"].sum()
    labels = summary["clade"] + "\n" + summary["resource"]
    ax.bar(np.arange(len(summary)), summary["normalized_rows"], color=[BLUE if c == "animal" else GREEN for c in summary["clade"]])
    ax.set_xticks(np.arange(len(summary)), labels, rotation=25, ha="right")
    ax.set_ylabel("normalized LR rows")
    style_axis(ax, grid="y")

    ax = axes[1]
    panel_label(ax, "B")
    ax.set_title("DB-free architecture", loc="left", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = ["protein sequence", "ESMC-300M embedding", "role classifiers", "candidate LR pairs", "LightGBM pair ranker", "density prior", "predicted LR table"]
    y = 0.86
    for i, step in enumerate(steps):
        flow_box(ax, (0.16, y), (0.68, 0.08), step, RED if i < 6 else GREEN, dashed=i < 6)
        if i < len(steps) - 1:
            arrow(ax, (0.50, y), (0.50, y - 0.05))
        y -= 0.12

    ax = axes[2]
    panel_label(ax, "C")
    ax.set_title("Pair-ranker validation", loc="left", weight="bold")
    v = validation[(validation["metric"].eq("PR-AUC")) & (validation["model"].eq("universal"))].copy()
    order = ["DB-free pair ranker", "embedding_cosine", "role_only", "expression_only", "degree_prior", "density_matched_random", "random"]
    v = v.set_index("method").loc[[m for m in order if m in set(v["method"])]].reset_index()
    ax.barh(np.arange(len(v)), v["value"], color=[RED if m == "DB-free pair ranker" else GRAY for m in v["method"]])
    ax.set_yticks(np.arange(len(v)), v["method"])
    ax.invert_yaxis()
    ax.set_xlabel("PR-AUC")
    ax.set_xlim(0, 1.0)
    style_axis(ax, grid="x")

    ax = axes[3]
    panel_label(ax, "D")
    ax.set_title("Density prior calibration", loc="left", weight="bold")
    x = np.arange(len(density))
    ax.bar(x - 0.18, density["density_prior"], width=0.34, label="prior", color=BLUE)
    ax.bar(x + 0.18, density["achieved_density"], width=0.34, label="achieved", color=GREEN)
    ax.set_xticks(x, density["dataset"], rotation=20, ha="right")
    ax.set_ylabel("LR density")
    ax.legend(frameon=False)
    style_axis(ax, grid="y")

    ax = axes[4]
    panel_label(ax, "E")
    ax.set_title("Predicted LR examples", loc="left", weight="bold")
    ax.axis("off")
    ex = examples.head(6).copy()
    ex["pair"] = ex["ligand"].astype(str).str[:11] + "->" + ex["receptor"].astype(str).str[:11]
    table = ax.table(
        cellText=ex[["dataset", "pair", "model_score", "nearest_reference_lr"]].round(3).astype(str).values,
        colLabels=["dataset", "pair", "score", "nearest ref"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.25)

    ax = axes[5]
    panel_label(ax, "F")
    ax.set_title("Protein-role validation", loc="left", weight="bold")
    r = roles[roles["model"].eq("universal")].copy()
    x = np.arange(len(r))
    ax.bar(x - 0.18, r["pr_auc"], width=0.34, color=RED, label="PR-AUC")
    ax.bar(x + 0.18, r["baseline_pr_auc"], width=0.34, color=GRAY, label="baseline")
    ax.set_xticks(x, r["role"], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    style_axis(ax, grid="y")
    save_figure(fig, MAIN / "figure5_dbfree_model")


def dbfree_model_score_mask(frame: pd.DataFrame) -> pd.Series:
    axolotl = (
        frame["source_dataset"].eq("artista_axolotl")
        & frame["validation_score_method"].eq("model_x_expression_potential")
        & frame["expression_potential_stat"].fillna("").eq("mean")
        & frame["expression_potential_power"].astype(float).eq(0.5)
    )
    soybean = frame["source_dataset"].eq("sota_soybean") & frame["validation_score_method"].eq("model")
    return axolotl | soybean


def figure6() -> None:
    topk = read_tsv(SOURCE / "figure6_spatial_topk_enrichment.tsv")
    nulls = read_tsv(SOURCE / "figure6_null_comparison.tsv")
    decay = read_tsv(SOURCE / "figure6_distance_decay.tsv")
    qc = read_tsv(SOURCE / "figure6_section_qc.tsv")
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    panel_label(ax, "A")
    ax.set_title("Spatial validation datasets", loc="left", weight="bold")
    ax.axis("off")
    q = qc[["source_dataset", "section_id", "n_cells_or_bins", "n_groups", "n_expression_genes"]].head(8).copy()
    table = ax.table(cellText=q.astype(str).values, colLabels=q.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 1.18)

    ax = axes[1]
    panel_label(ax, "B")
    ax.set_title("Top-K spatial enrichment", loc="left", weight="bold")
    b = topk[
        topk["kernel"].eq("exp")
        & topk["score_type"].eq("model_weighted_spatial_ccc_score")
        & topk["null_model"].eq("matched_random_lr")
        & topk["k"].isin([500, 1000])
        & dbfree_model_score_mask(topk)
    ].copy()
    if b.empty:
        b = topk[topk["null_model"].eq("matched_random_lr") & topk["k"].isin([500, 1000])].copy()
    b = (
        b.groupby(["source_dataset", "section_id", "k"], as_index=False)
        .agg(top_k_enrichment_z=("top_k_enrichment_z", "mean"))
        .sort_values(["source_dataset", "section_id", "k"])
    )
    b["label"] = b["section_id"].astype(str) + "\nK=" + b["k"].astype(str)
    ax.bar(np.arange(len(b)), b["top_k_enrichment_z"], color=[BLUE if "artista" in d else GREEN for d in b["source_dataset"]])
    ax.axhline(2, color=GRAY, ls="--", lw=1)
    ax.set_xticks(np.arange(len(b)), b["label"], rotation=45, ha="right")
    ax.set_ylabel("enrichment z")
    style_axis(ax, grid="y")

    ax = axes[2]
    panel_label(ax, "C")
    ax.set_title("Null model comparison", loc="left", weight="bold")
    c = nulls[
        nulls["section_id"].eq("5DPI_1")
        & nulls["kernel"].eq("exp")
        & nulls["score_type"].eq("model_weighted_spatial_ccc_score")
        & nulls["k"].eq(500)
        & dbfree_model_score_mask(nulls)
    ].copy()
    if c.empty:
        c = nulls[(nulls["k"].eq(500))].head(5).copy()
    c = c.groupby("null_model", as_index=False).agg(top_k_enrichment_z=("top_k_enrichment_z", "mean"))
    ax.barh(np.arange(len(c)), c["top_k_enrichment_z"], color=PURPLE)
    ax.set_yticks(np.arange(len(c)), c["null_model"])
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("enrichment z")
    style_axis(ax, grid="x")

    ax = axes[3]
    panel_label(ax, "D")
    ax.set_title("Distance-decay curves", loc="left", weight="bold")
    for dataset, part in decay.groupby("source_dataset"):
        part = part.sort_values("mean_distance")
        ax.plot(part["mean_distance"], part["model_weighted_mean_spatial_ccc_score"], marker="o", label=dataset)
    ax.set_xlabel("distance")
    ax.set_ylabel("model-weighted CCC")
    ax.legend(frameon=False)
    style_axis(ax, grid="y")

    ax = axes[4]
    panel_label(ax, "E")
    ax.set_title("Baseline ranking comparison", loc="left", weight="bold")
    base = safe_read_tsv("results/dbfree_validation/baseline_comparison.tsv")
    e = base[(base["section_id"].eq("5DPI_1")) & (base["null_model"].eq("matched_random_lr")) & (base["k"].eq(500))].copy()
    if e.empty:
        e = base[(base["null_model"].eq("matched_random_lr")) & (base["k"].eq(500))].head(8).copy()
    e = e.groupby("validation_strategy", as_index=False).agg(enrichment_z=("enrichment_z", "mean"))
    ax.barh(np.arange(len(e)), e["enrichment_z"], color=np.where(e["validation_strategy"].eq("dbfree"), RED, GRAY))
    ax.set_yticks(np.arange(len(e)), e["validation_strategy"])
    ax.set_xlabel("enrichment z")
    style_axis(ax, grid="x")

    ax = axes[5]
    panel_label(ax, "F")
    image_panel(ax, ROOT / "docs/figures/dbfree_spatial_validation_main.png", "Example spatial CCC validation")
    save_figure(fig, MAIN / "figure6_spatial_validation")


def supplementary() -> None:
    supplementary1()
    supplementary2()
    supplementary3()
    supplementary4()
    supplementary5()


def supplementary1() -> None:
    frame = read_tsv(SOURCE / "supplementary_figure1_visual_coverage.tsv")
    bool_cols = frame.columns[1:]
    data = frame[bool_cols].astype(bool).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.imshow(data, cmap=matplotlib.colors.ListedColormap(["#FEE0D2", "#A1D99B"]), aspect="auto")
    ax.set_yticks(np.arange(len(frame)), frame["feature"])
    ax.set_xticks(np.arange(len(bool_cols)), bool_cols, rotation=35, ha="right")
    ax.set_title("Supplementary Figure 1. Visual/API coverage matrix", loc="left", weight="bold")
    save_figure(fig, SUPP / "supplementary_figure1_visual_coverage")


def supplementary2() -> None:
    frame = read_tsv(SOURCE / "supplementary_figure2_reproducibility.tsv")
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=frame.values, colLabels=frame.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.45)
    ax.set_title("Supplementary Figure 2. Reproducibility gates", loc="left", weight="bold")
    save_figure(fig, SUPP / "supplementary_figure2_reproducibility")


def supplementary3() -> None:
    frame = read_tsv(SOURCE / "supplementary_figure3_benchmark_sensitivity.tsv")
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    if {"n_cells", "strategy", "total_seconds"}.issubset(frame.columns):
        ok = frame[frame["status"].eq("ok")].copy()
        for strategy, part in ok.groupby("strategy"):
            ax.plot(part["n_cells"], part["total_seconds"], marker="o", label=strategy)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cells")
        ax.set_ylabel("seconds")
        ax.legend(frameon=False)
        style_axis(ax, grid="both")
    else:
        ax.axis("off")
        wrapped(ax, 0.05, 0.85, "No benchmark sensitivity source table was available.", width=60)
    ax.set_title("Supplementary Figure 3. Benchmark sensitivity", loc="left", weight="bold")
    save_figure(fig, SUPP / "supplementary_figure3_benchmark_sensitivity")


def supplementary4() -> None:
    val = read_tsv(SOURCE / "figure5_model_validation.tsv")
    density = read_tsv(SOURCE / "figure5_density_prior.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    pr = val[(val["metric"].eq("PR-AUC")) & (val["method"].eq("DB-free pair ranker"))].copy()
    axes[0].bar(pr["model"], pr["value"], color=RED)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("Pair ranker")
    style_axis(axes[0], grid="y")
    axes[1].bar(density["dataset"], density["selected_pair_count"], color=[BLUE, GREEN])
    axes[1].set_ylabel("selected LR pairs")
    axes[1].set_title("Density-controlled edge count")
    style_axis(axes[1], grid="y")
    fig.suptitle("Supplementary Figure 4. DB-free model card summary", weight="bold")
    save_figure(fig, SUPP / "supplementary_figure4_dbfree_model_card")


def supplementary5() -> None:
    topk = read_tsv(SOURCE / "figure6_spatial_topk_enrichment.tsv")
    b = topk[
        topk["null_model"].eq("matched_random_lr")
        & topk["k"].isin([500, 1000])
        & topk["kernel"].eq("exp")
        & topk["score_type"].eq("model_weighted_spatial_ccc_score")
        & dbfree_model_score_mask(topk)
    ].copy()
    if b.empty:
        b = topk[(topk["null_model"].eq("matched_random_lr")) & (topk["k"].isin([500, 1000]))].copy()
    b = (
        b.groupby(["source_dataset", "section_id", "k"], as_index=False)
        .agg(top_k_enrichment_z=("top_k_enrichment_z", "mean"))
        .sort_values(["source_dataset", "section_id", "k"])
    )
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    b["label"] = b["source_dataset"] + "\n" + b["section_id"].astype(str) + "\nK=" + b["k"].astype(str)
    ax.bar(np.arange(len(b)), b["top_k_enrichment_z"], color=[BLUE if "artista" in d else GREEN for d in b["source_dataset"]])
    ax.axhline(2, color=GRAY, ls="--", lw=1)
    ax.set_xticks(np.arange(len(b)), b["label"], rotation=50, ha="right")
    ax.set_ylabel("enrichment z")
    ax.set_title("Supplementary Figure 5. All DB-free spatial validation sections", loc="left", weight="bold")
    style_axis(ax, grid="y")
    save_figure(fig, SUPP / "supplementary_figure5_all_spatial_sections")


def write_legends() -> None:
    legends = {
        "figure1_legend.md": """# Figure 1. pyccc overview

pyccc unifies CellChat-like CCC inference, differential analysis, visualization,
LR resource loading, and optional DB-free LR prediction in an AnnData-native
Python workflow.

Panel A shows the input problem setting and the pyccc solution surface. Panel B
shows the regular curated-LR workflow. Panel C shows CellChatDB, OmniPath,
external files, user tables, and DB-free predicted LR tables converging into a
CellChatDB-compatible LR table. Panel D shows the optional target-species
DB-free prediction branch. Panel E shows the minimal API surface.

Data source: API inventory in `docs/paper/source_data/figure1_api_inventory.tsv`.
The DB-free branch is an optional experimental LR-table generator, not the
default regular-mode benchmark path.
""",
        "figure2_legend.md": """# Figure 2. CellChat-like analysis and visualization coverage

pyccc covers major CellChat-style visualization families with Python-native
plotting while keeping results in AnnData/pandas-compatible objects.

Panels A-F show differential LR bubble, communication network, pathway
embedding, role heatmap, pathway river, and spatial network examples.

Data source: asset manifest in `docs/paper/source_data/figure2_visual_assets.tsv`.
Caveat: this figure demonstrates CellChat-like visualization coverage for the
covered workflow; it does not claim complete visual parity with CellChat.
""",
        "figure3_legend.md": """# Figure 3. Million-cell scaling and sparse tri-mean optimization

pyccc completes a matched 1M-cell CellChat-like workflow in 11.2 seconds from
prepared input under strict 4-core affinity.

Panel A shows the benchmark design. Panel B shows total runtime from prepared
input read through matched visualizations. Panel C decomposes time by input read,
compute, export, and plotting components. Panel D summarizes the sparse
tri-mean optimization. Panel E reports peak CPU cores. Panel F states the timing
policy.

Data source: `docs/paper/source_data/figure3_runtime_benchmark.tsv`,
`figure3_runtime_decomposition.tsv`, and `figure3_cpu_affinity.tsv`.
The benchmark uses prepared 1M input, strict 4-core affinity, and regular mode
only. Sampling from the full 1.82M-cell h5ad and LR filtering/preparation are
not counted.
""",
        "figure4_legend.md": """# Figure 4. Differential CCC case study

pyccc enables interpretable large-scale differential CCC analysis directly in
Python.

Panels A-F summarize the CMV infection versus normal 1M-cell comparison,
differential source-target network shifts, top differential LR pairs,
pathway-level changes, outgoing role shifts, and pathway embedding.

Data source: `docs/paper/source_data/figure4_*.tsv`. This figure is a biological
use case demonstrating interpretability and scale; it does not claim novel CMV
biology without external validation.
""",
        "figure5_legend.md": """# Figure 5. DB-free LR prediction model

DB-free prediction converts target-species protein sequences into
provenance-rich LR candidate tables using ESMC-300M embeddings, LightGBM
ranking, and density-controlled edge selection.

Panels A-F show training resources, model architecture, pair-ranker validation,
density prior calibration, example predicted LR rows, and protein-role
classifier validation.

Data source: `docs/paper/source_data/figure5_*.tsv` plus model-card JSON files
under `models/`. Predicted LR rows are computational candidates and should not
be interpreted as experimentally validated biochemical interactions.
""",
        "figure6_legend.md": """# Figure 6. DB-free CCC spatial validation

In axolotl and soybean spatial sections, DB-free predicted LR tables produce
spatially enriched CCC signals against matched and permutation nulls.

Panels A-F show dataset QC, top-K enrichment, null model comparison,
distance-decay curves, baseline strategy comparison, and an example spatial CCC
validation panel.

Data source: `docs/paper/source_data/figure6_*.tsv`. Spatial enrichment is
plausibility evidence and does not establish direct biochemical binding.
""",
    }
    for name, text in legends.items():
        (LEGENDS / name).write_text(text.strip() + "\n", encoding="utf-8")


def write_submission_targets() -> None:
    text = """# Submission target strategy

## Primary recommendation

### Genome Biology

Recommended as the main target if DB-free spatial validation and the million-cell
benchmark remain strong. The scope fits genomic/post-genomic methods, software,
systems/network biology, and spatial transcriptomics.

Scope page: https://genomebiology.biomedcentral.com/about

## Strong alternatives

### PLOS Computational Biology

Good if the manuscript emphasizes method/software plus real-world data
validation and reusable workflows.

Journal information: https://journals.plos.org/ploscompbiol/s/journal-information

### NAR Genomics and Bioinformatics

Good second-line target for a complete methods/software paper with reproducible
workflows.

### Bioinformatics

Good practical target for a focused software/methods article. Prefer a full
article rather than an Application Note if DB-free prediction and spatial
validation remain central.

### GigaScience

Good if emphasizing open data, source data, reproducible workflows, and reusable
artifacts.

## Stretch target

### Nature Methods

Only attempt as a pre-submission inquiry if DB-free spatial validation is strong
across special species, model validation is substantially stronger than
baselines, and the paper can argue a new capability beyond Python
implementation/speed.

## Recommended order

If DB-free validation is stable: Genome Biology, PLOS Computational Biology,
Nature Methods pre-submission inquiry, NAR Genomics and Bioinformatics,
Bioinformatics, GigaScience.

If DB-free validation remains experimental: Bioinformatics full article, NAR
Genomics and Bioinformatics, GigaScience, PLOS Computational Biology, Genome
Biology.
"""
    (PAPER / "submission_targets.md").write_text(text, encoding="utf-8")


def write_fallback_document() -> None:
    text = """# Five-figure fallback version

If a journal strongly prefers fewer main figures, use this 5-figure structure:

1. Figure 1 - Overview and architecture.
2. Figure 2 - CellChat-like visual and analytical coverage.
3. Figure 3 - Million-cell scaling and sparse tri-mean optimization.
4. Figure 4 - Large real-dataset differential CCC use case.
5. Figure 5 - DB-free LR prediction and spatial validation.

Fallback merge plan:

- Combine current Figure 5A/B/C/D/E/F with current Figure 6B/C/D/F.
- Move all-section spatial validation, model-card details, and density-prior
  expanded views to supplementary figures.
- Keep caveats explicit: predicted LR rows are computational candidates; spatial
  enrichment is plausibility evidence, not biochemical validation.
"""
    (MAIN / "fallback_5figure_version.md").write_text(text, encoding="utf-8")


def write_artifact_manifest() -> None:
    rows = []
    for folder, artifact_type in [(MAIN, "main_figure"), (SUPP, "supplementary_figure"), (SOURCE, "source_data"), (LEGENDS, "legend"), (TABLES, "table")]:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                rows.append({"artifact_type": artifact_type, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
    write_tsv(pd.DataFrame(rows), MANIFESTS / "manuscript_artifacts.tsv")


def write_qa_contact_sheet() -> None:
    pngs = sorted(MAIN.glob("figure*.png")) + sorted(SUPP.glob("supplementary_figure*.png"))
    thumbs = []
    for p in pngs:
        img = Image.open(p).convert("RGB")
        img = ImageOps.contain(img, (420, 300), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (440, 340), "white")
        canvas.paste(img, ((440 - img.width) // 2, 12))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 312), p.stem, fill=(17, 24, 39))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 440, rows * 340), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 440, (i // cols) * 340))
    sheet.save(PAPER / "figures" / "qa_contact_sheet.png")


FIGURE_BUILDERS = {
    "1": figure1,
    "2": figure2,
    "3": figure3,
    "4": figure4,
    "5": figure5,
    "6": figure6,
    "supplementary": supplementary,
}


if __name__ == "__main__":
    main()
