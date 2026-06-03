# Paper Artifacts

This folder contains compact, committed artifacts that support the repository's
publication-style claims without requiring large local benchmark directories.

## Contents

- `results/`: small TSV snapshots extracted from local CellChat parity,
  real-1M runtime, and summary-mean tradeoff benchmarks.
- `main_figure/`: generated publication figure artifacts.
- `main_figure/subfigures/`: square panel PDFs, SVGs, and PNGs.
- `main_figure/pyccc_main_figure.pdf`: A4 LaTeX-assembled main figure.
- `main_figure/pyccc_main_figure_legend.md`: figure legend and statistical
  notes.
- `main_figure/qa_multipage_contact_sheet.png`: visual QA contact sheet.
- `main_figure/qa_edge_scan.tsv`: edge-pixel scan for clipped-panel triage.

## Regeneration

```bash
uv run python scripts/make_publication_figure.py
```

The script reads only `docs/paper/results/*.tsv`, so the figure can be
regenerated without the large ignored `data/` and CellChat comparison output
directories. The upstream commands that created those summary tables are
recorded in `../reproducibility.md`.
