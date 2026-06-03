from __future__ import annotations

import math
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "paper" / "results"
OUT = ROOT / "docs" / "paper" / "main_figure"
SUBFIGURES = OUT / "subfigures"
QA_PAGES = OUT / "qa_pages"
PUBLIC_PNG = ROOT / "docs" / "figures" / "publication_main_figure.png"

BLUE = "#2C7FB8"
GREEN = "#41AB5D"
ORANGE = "#D95F0E"
GRAY = "#4B5563"
LIGHT = "#E5E7EB"
DARK = "#111827"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def square_fig(size: float = 2.65):
    fig, ax = plt.subplots(figsize=(size, size))
    return fig, ax


def style_axes(ax, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8, pad=2)
    if grid:
        ax.grid(axis="x", color=LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)


def save_panel(fig, name: str) -> None:
    SUBFIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 320},
    }.items():
        fig.savefig(SUBFIGURES / f"{name}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_workflow() -> None:
    fig, ax = square_fig()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "AnnData-first CCC workflow", ha="left", va="top", fontsize=8.5, weight="bold", color=DARK)

    boxes = [
        (0.08, 0.70, 0.32, 0.14, "AnnData\nsparse/backed", BLUE),
        (0.60, 0.70, 0.32, 0.14, "CellChatDB\nLR + cofactors", ORANGE),
        (0.08, 0.43, 0.84, 0.15, "LR/cofactor gene subset -> group summaries", GREEN),
        (0.08, 0.19, 0.36, 0.14, "Python-native\nplots + tables", BLUE),
        (0.56, 0.19, 0.36, 0.14, "CellChat RDS\nbridge plots", ORANGE),
    ]
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.12)
        ax.add_patch(rect)
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.4, color=DARK)

    arrows = [
        ((0.40, 0.77), (0.60, 0.77)),
        ((0.24, 0.70), (0.24, 0.58)),
        ((0.76, 0.70), (0.76, 0.58)),
        ((0.32, 0.43), (0.28, 0.33)),
        ((0.68, 0.43), (0.74, 0.33)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=0.9, color=GRAY, shrinkA=2, shrinkB=2))

    ax.text(
        0.08,
        0.06,
        "Shared LR table and grouped expression keep results comparable\nwhile avoiding CellChat's cell-level dense path.",
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=6.4,
        linespacing=1.18,
    )
    save_panel(fig, "panel_a_workflow")


def panel_parity() -> None:
    metrics = pd.read_csv(RESULTS / "cellchat_parity_metrics.tsv", sep="\t")
    fig, ax = square_fig()
    names = [
        "LR prob.",
        "network",
        "pathway\nflow",
        "LR flow",
        "MIF contrib.",
    ]
    y = np.arange(len(metrics))
    ax.scatter(metrics["pearson"], y + 0.18, s=24, color=BLUE, label="Pearson")
    ax.scatter(metrics["spearman"], y, s=24, color=GREEN, marker="s", label="Spearman")
    ax.scatter(metrics["top20_overlap"], y - 0.18, s=24, color=ORANGE, marker="^", label="Top-20")
    ax.set_yticks(y, names)
    ax.set_xlim(0.965, 1.006)
    ax.set_xlabel("Agreement with CellChat R")
    ax.set_title("Numerical parity")
    style_axes(ax, grid=True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.03), frameon=False, fontsize=6.2, handletextpad=0.2)
    max_diff = metrics["max_abs_diff"].max()
    ax.text(0.967, len(metrics) - 0.55, f"max |delta| <= {max_diff:.1e}", ha="left", va="center", fontsize=6.5, color=GRAY)
    save_panel(fig, "panel_b_parity")


def panel_runtime() -> None:
    frame = pd.read_csv(RESULTS / "real1m_runtime.tsv", sep="\t")
    frame = frame[frame["status"].eq("ok")].copy()
    order = ["pyccc_python", "pyccc_cellchat_bridge", "direct_cellchat"]
    labels = {
        "pyccc_python": "pyccc native",
        "pyccc_cellchat_bridge": "pyccc +\nCellChat plots",
        "direct_cellchat": "direct\nCellChat R",
    }
    colors = {
        "pyccc_python": BLUE,
        "pyccc_cellchat_bridge": GREEN,
        "direct_cellchat": ORANGE,
    }
    frame["strategy"] = pd.Categorical(frame["strategy"], categories=order, ordered=True)
    frame = frame.sort_values("strategy")
    direct = float(frame.loc[frame["strategy"].astype(str).eq("direct_cellchat"), "total_seconds"].iloc[0])
    y = np.arange(len(frame))

    fig, ax = square_fig()
    ax.barh(y, frame["total_seconds"], color=[colors[str(s)] for s in frame["strategy"]], height=0.58)
    ax.set_yticks(y, [labels[str(s)] for s in frame["strategy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Total time (s)")
    ax.set_title("Real 1M-cell workflow")
    style_axes(ax, grid=True)
    ax.set_xlim(0, 260)
    for i, row in enumerate(frame.itertuples(index=False)):
        seconds = float(row.total_seconds)
        ax.text(seconds + 6, i, f"{seconds:.1f}s\n{direct / seconds:.2f}x", va="center", fontsize=6.4, color=DARK)
    save_panel(fig, "panel_c_runtime")


def panel_summary_tradeoff() -> None:
    frame = pd.read_csv(RESULTS / "summary_mean_tradeoff.tsv", sep="\t")
    tri_seconds = float(frame.loc[frame["method"].eq("tri_mean_cpu"), "seconds"].iloc[0])
    frame = frame[~frame["method"].eq("tri_mean_cpu")].copy()
    frame["speedup_vs_trimean_cpu"] = tri_seconds / frame["seconds"]
    label_map = {
        "clipped_p99_cpu": "clipped p99",
        "clipped_p95_cpu": "clipped p95",
        "gated_p99_cpu": "gated p99",
        "gated_p95_cpu": "gated p95",
    }
    label_offset = {
        "clipped_p99_cpu": (0.025, 0.020),
        "clipped_p95_cpu": (0.025, -0.024),
        "gated_p99_cpu": (0.025, 0.018),
        "gated_p95_cpu": (0.025, -0.024),
    }
    color_map = {"clipped_mean": ORANGE, "gated_mean": GREEN}
    marker_map = {0.99: "o", 0.95: "s"}

    fig, ax = square_fig()
    for row in frame.itertuples(index=False):
        ax.scatter(
            row.speedup_vs_trimean_cpu,
            row.spearman_to_trimean,
            s=36 + 90 * float(row.top100_overlap_to_trimean),
            color=color_map[row.aggregate],
            marker=marker_map[float(row.clip_quantile)],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        dx, dy = label_offset[row.method]
        ax.text(
            row.speedup_vs_trimean_cpu + dx,
            row.spearman_to_trimean + dy,
            label_map[row.method],
            va="center",
            fontsize=6.1,
            color=DARK,
        )
    ax.axvline(1.0, color=LIGHT, linewidth=0.9)
    ax.axhline(0.9, color=LIGHT, linewidth=0.9)
    ax.set_xlim(0.48, 1.20)
    ax.set_ylim(0.50, 1.01)
    ax.set_xlabel("Speed vs triMean CPU")
    ax.set_ylabel("Spearman to triMean")
    ax.set_title("Robust mean summaries")
    style_axes(ax, grid=False)
    ax.text(0.50, 0.97, "dot size: top-100 overlap", ha="left", va="top", fontsize=6.0, color=GRAY)
    save_panel(fig, "panel_d_summary_tradeoff")


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_legend() -> None:
    legend = """# pyccc Main Figure Legend

Figure 1. Publication-level summary of pyccc as an AnnData-native, CellChat-compatible CCC workflow.

Panel A shows the analysis path: sparse or backed AnnData input and CellChatDB ligand-receptor metadata are reduced to LR/cofactor gene summaries before scoring. Results remain in Python-native tables and can also be exported back to minimal CellChat R objects for R-native plotting.

Panel B shows numerical agreement with direct CellChat R on the official human skin tutorial benchmark. Pearson correlation, Spearman correlation, and top-20 overlap are all 1.0 across LR-source-target probabilities, global network weights, pathway information flow, LR information flow, and MIF LR contribution; maximum absolute differences are floating-point roundoff.

Panel C compares total time for within-sample CCC, two-condition differential CCC, and matched visual outputs on a real 1,000,000-cell sample from the CELLxGENE Human Immune Health Atlas. The benchmark uses raw.X for all methods because the h5ad main X matrix is scaled and contains negative values.

Panel D summarizes the tradeoff between outlier-robust mean summaries and CellChat triMean similarity on the local CELLxGENE benchmark. The gated mean variants preserve higher rank agreement to triMean than plain clipped means while remaining slightly faster than the triMean CPU reference in this environment.

Statistical notes. All reported communication scores are inferred group-level quantities from expression summaries and ligand-receptor metadata, not direct physical interaction measurements. The CellChat parity panel uses the same filtered LR table and metadata as the R reference. The 1M-cell runtime panel reports one local run on the stated environment and should be interpreted as an artifact-level benchmark rather than a universal performance guarantee.
"""
    (OUT / "pyccc_main_figure_legend.md").write_text(legend, encoding="utf-8")


def write_latex() -> Path:
    tex = r"""
\documentclass[10pt]{article}
\usepackage[a4paper,margin=13mm]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\pagestyle{empty}
\newcommand{\panel}[3]{%
  \begin{minipage}[t]{#1}
  \textbf{\Large #2}\vspace{1mm}\\
  \includegraphics[width=\linewidth]{#3}
  \end{minipage}
}
\begin{document}
\noindent{\Large\textbf{pyccc: CellChat-compatible CCC at AnnData scale}}\\[2mm]
\noindent\panel{0.47\textwidth}{a}{subfigures/panel_a_workflow.pdf}\hfill
\panel{0.47\textwidth}{b}{subfigures/panel_b_parity.pdf}\\[5mm]
\noindent\panel{0.47\textwidth}{c}{subfigures/panel_c_runtime.pdf}\hfill
\panel{0.47\textwidth}{d}{subfigures/panel_d_summary_tradeoff.pdf}\\[4mm]
{\small
\textbf{Figure 1.} pyccc keeps CellChat-like CCC in an AnnData-native sparse workflow,
matches CellChat R numerics on the official tutorial benchmark, accelerates the
real 1M-cell workflow by avoiding CellChat's dense cell-level path, and exposes
outlier-robust summary options with explicit similarity tradeoffs.
}
\end{document}
"""
    path = OUT / "pyccc_main_figure.tex"
    path.write_text(textwrap.dedent(tex).strip() + "\n", encoding="utf-8")
    return path


def compile_latex(tex_path: Path) -> Path:
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to assemble the publication figure PDF.")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=OUT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for suffix in [".aux", ".log"]:
        aux = OUT / f"{tex_path.stem}{suffix}"
        if aux.exists():
            aux.unlink()
    return OUT / f"{tex_path.stem}.pdf"


def render_pdf(pdf_path: Path) -> Path:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required for figure QA rendering.")
    QA_PAGES.mkdir(parents=True, exist_ok=True)
    prefix = QA_PAGES / "page"
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "180", str(pdf_path), str(prefix)], check=True)
    page_png = prefix.with_suffix(".png")
    PUBLIC_PNG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(page_png, PUBLIC_PNG)
    return page_png


def make_contact_sheet(page_png: Path) -> None:
    images = [(page_png, "assembled PDF page")]
    images.extend((path, path.stem.replace("_", " ")) for path in sorted(SUBFIGURES.glob("panel_*.png")))
    thumb_w, thumb_h = 430, 330
    label_h = 24
    cols = 2
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(images):
        col = i % cols
        row = i // cols
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w - 18, thumb_h - 18), Image.Resampling.LANCZOS)
        framed = ImageOps.expand(img, border=1, fill="#D1D5DB")
        x = col * thumb_w + (thumb_w - framed.width) // 2
        y = row * (thumb_h + label_h) + label_h
        draw.text((col * thumb_w + 10, row * (thumb_h + label_h) + 5), label, fill=DARK)
        sheet.paste(framed, (x, y))
    sheet.save(OUT / "qa_multipage_contact_sheet.png")


def edge_scan() -> None:
    rows = []
    for path in sorted(SUBFIGURES.glob("panel_*.png")):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)
        ink = np.any(arr < 245, axis=2)
        band = 18
        total = ink.size
        edge = (
            ink[:band, :].sum()
            + ink[-band:, :].sum()
            + ink[:, :band].sum()
            + ink[:, -band:].sum()
        )
        rows.append({"panel": path.name, "edge_ink_fraction": edge / total})
    pd.DataFrame(rows).to_csv(OUT / "qa_edge_scan.tsv", sep="\t", index=False)


def main() -> None:
    configure_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    SUBFIGURES.mkdir(parents=True, exist_ok=True)
    panel_workflow()
    panel_parity()
    panel_runtime()
    panel_summary_tradeoff()
    write_legend()
    tex_path = write_latex()
    pdf_path = compile_latex(tex_path)
    page_png = render_pdf(pdf_path)
    make_contact_sheet(page_png)
    edge_scan()
    print(f"Wrote {pdf_path.relative_to(ROOT)}")
    print(f"Wrote {PUBLIC_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
