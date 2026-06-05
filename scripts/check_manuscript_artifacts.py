from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
MAIN = PAPER / "figures" / "main"
SUPP = PAPER / "figures" / "supplementary"
SOURCE = PAPER / "source_data"
LEGENDS = PAPER / "legends"
MANIFEST = PAPER / "manifests" / "manuscript_artifacts.tsv"

MAIN_STEMS = [
    "figure1_overview",
    "figure2_visual_coverage",
    "figure3_runtime_scaling",
    "figure4_differential_case_study",
    "figure5_dbfree_model",
    "figure6_spatial_validation",
]

SUPPLEMENTARY_STEMS = [
    "supplementary_figure1_visual_coverage",
    "supplementary_figure2_reproducibility",
    "supplementary_figure3_benchmark_sensitivity",
    "supplementary_figure4_dbfree_model_card",
    "supplementary_figure5_all_spatial_sections",
]

SOURCE_TABLES = [
    "figure1_api_inventory.tsv",
    "figure2_visual_assets.tsv",
    "figure3_runtime_benchmark.tsv",
    "figure3_runtime_decomposition.tsv",
    "figure3_cpu_affinity.tsv",
    "figure4_differential_network.tsv",
    "figure4_differential_lr.tsv",
    "figure4_pathway_rank.tsv",
    "figure5_model_validation.tsv",
    "figure5_density_prior.tsv",
    "figure5_predicted_lr_examples.tsv",
    "figure6_spatial_topk_enrichment.tsv",
    "figure6_null_comparison.tsv",
    "figure6_distance_decay.tsv",
    "figure6_section_qc.tsv",
]


def main() -> None:
    errors: list[str] = []
    check_required_files(errors)
    check_source_tables(errors)
    check_figure3_benchmark(errors)
    check_legends(errors)
    check_manifest(errors)
    check_no_retired_benchmark(errors)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print("OK: manuscript figure package artifacts passed local checks.")


def check_required_files(errors: list[str]) -> None:
    for stem in MAIN_STEMS:
        for suffix in [".svg", ".pdf", ".png"]:
            require_file(MAIN / f"{stem}{suffix}", errors)
    for stem in SUPPLEMENTARY_STEMS:
        for suffix in [".svg", ".pdf", ".png"]:
            require_file(SUPP / f"{stem}{suffix}", errors)
    for i in range(1, 7):
        require_file(LEGENDS / f"figure{i}_legend.md", errors)
    require_file(MAIN / "fallback_5figure_version.md", errors)
    require_file(PAPER / "figures" / "qa_contact_sheet.png", errors)
    require_file(PAPER / "submission_targets.md", errors)
    require_file(MANIFEST, errors)


def check_source_tables(errors: list[str]) -> None:
    for name in SOURCE_TABLES:
        path = SOURCE / name
        require_file(path, errors)
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, sep="\t")
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{relative(path)} could not be read as TSV: {exc}")
            continue
        if frame.empty:
            errors.append(f"{relative(path)} is empty")
    distance = SOURCE / "figure6_distance_decay.tsv"
    if distance.exists() and distance.stat().st_size > 2_000_000:
        errors.append(f"{relative(distance)} is too large for committed plotted source data")


def check_figure3_benchmark(errors: list[str]) -> None:
    path = SOURCE / "figure3_runtime_benchmark.tsv"
    if not path.exists():
        return
    frame = pd.read_csv(path, sep="\t")
    expected = {
        "pyccc_python": 11.24407549900934,
        "pyccc_cellchat_bridge": 27.534923942002933,
        "direct_cellchat": 195.54915448097745,
    }
    for strategy, seconds in expected.items():
        rows = frame[frame["strategy"].eq(strategy)]
        if rows.empty:
            errors.append(f"figure3 benchmark missing strategy {strategy}")
            continue
        observed = float(rows.iloc[0]["total_with_input_read_seconds"])
        if abs(observed - seconds) > 0.01:
            errors.append(f"{strategy} total_with_input_read_seconds={observed:.6f}, expected {seconds:.6f}")
    for column, expected_value in [
        ("cpu_affinity_count", 4),
        ("openblas_num_threads", 4),
        ("omp_num_threads", 4),
        ("mkl_num_threads", 4),
        ("numexpr_num_threads", 4),
        ("n_jobs", 1),
    ]:
        if column not in frame.columns:
            errors.append(f"figure3 benchmark missing {column}")
            continue
        bad = frame[frame[column].astype(int).ne(expected_value)]
        if not bad.empty:
            values = ", ".join(f"{r.strategy}={getattr(r, column)}" for r in bad.itertuples())
            errors.append(f"figure3 benchmark {column} mismatch: {values}")


def check_legends(errors: list[str]) -> None:
    required_terms = {
        "figure3_legend.md": ["prepared 1M input", "strict 4-core affinity", "regular mode"],
        "figure5_legend.md": ["computational candidates", "biochemical interactions"],
        "figure6_legend.md": ["plausibility evidence", "biochemical binding"],
    }
    for name, terms in required_terms.items():
        path = LEGENDS / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for term in terms:
            if term.lower() not in lower:
                errors.append(f"{relative(path)} missing required term: {term}")


def check_manifest(errors: list[str]) -> None:
    if not MANIFEST.exists():
        return
    frame = pd.read_csv(MANIFEST, sep="\t")
    if "path" not in frame.columns:
        errors.append(f"{relative(MANIFEST)} missing path column")
        return
    paths = set(frame["path"].astype(str))
    required = {
        "docs/paper/figures/main/figure3_runtime_scaling.svg",
        "docs/paper/figures/main/figure6_spatial_validation.pdf",
        "docs/paper/source_data/figure3_runtime_benchmark.tsv",
        "docs/paper/source_data/figure6_spatial_topk_enrichment.tsv",
        "docs/paper/legends/figure5_legend.md",
    }
    missing = sorted(required - paths)
    for path in missing:
        errors.append(f"manifest missing {path}")


def check_no_retired_benchmark(errors: list[str]) -> None:
    banned = [
        "64-core speedup",
        "64-thread",
        "47.014",
        "47.01",
        "201.096",
        "201.10",
        "8.627",
        "23.31",
    ]
    paths = [
        MAIN / "figure3_runtime_scaling.svg",
        LEGENDS / "figure3_legend.md",
        SOURCE / "figure3_runtime_benchmark.tsv",
        SOURCE / "figure3_runtime_decomposition.tsv",
        SOURCE / "figure3_cpu_affinity.tsv",
    ]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in banned:
            if needle in text:
                errors.append(f"{relative(path)} contains retired benchmark text: {needle}")


def require_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing {relative(path)}")
    elif path.stat().st_size == 0:
        errors.append(f"{relative(path)} is empty")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
