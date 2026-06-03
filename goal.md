# pyccc vs CellChat plotting, acceleration, and scale goal

## Status

Completed on 2026-06-02 against the current worktree.

## Objective

Compare pyccc Python-generated figures with real CellChat R-generated figures
on the same single-cell dataset and ligand-receptor set. For the classic
CellChat analyses, generate one pyccc version and one CellChat R version, place
them side by side, and merge the comparisons into an A4 report. After the
plotting comparison is implemented, try optional CuPy/cuDF-style acceleration
for CCC analysis. Finally, document what prevents CellChat from scaling to
1M-10M cells and whether pyccc can overcome those limits.

## Dataset And LR Set

- Dataset: official CellChat human skin scRNA-seq tutorial data.
- Conditions: primarily `LS`; comparison plots may also use `LS` vs `NL`.
- Grouping: `cell_type`.
- LR set: CellChatDB human `Secreted Signaling`, with the same filtered LR table
  used by pyccc and CellChat R.

## Required Plot Comparisons

- Overall source-target network.
- Ligand-receptor bubble plot.
- Pathway/source-target heatmap.
- Pathway information-flow rank.
- Top-pathway ligand-receptor contribution.
- Signaling role outgoing/incoming scatter.
- Classic pathway-level network views where supported by both stacks:
  hierarchy, circle, chord, and LR-mediated chord/gene view.
- Differential/ranked signaling comparison for `LS` vs `NL` where practical.

## Required Acceleration Work

- Inspect the current CCC computation bottlenecks.
- Try introducing optional CuPy/cuDF-style acceleration without making GPU
  packages required dependencies.
- Preserve CPU fallback behavior.
- Add regression coverage for the acceleration dispatch or fallback semantics.

## Required Scale Analysis

- Explain the main reasons CellChat struggles at 1M-10M cells.
- Separate cell-level bottlenecks from grouped-expression bottlenecks.
- State which limits pyccc can realistically overcome and which still require
  approximation, sketching, batching, or a different statistical workflow.

## Acceptance Criteria

- A reproducible script generates side-by-side A4 Python-vs-R plotting
  comparisons.
- The R figures are actually drawn by CellChat R plotting functions, not by
  replotting CellChat values in Python.
- The pyccc figures are drawn by pyccc Python plotting functions.
- Generated artifacts include individual Python/R figure files and a merged A4
  PDF report.
- Numeric parity from the previous comparison is preserved.
- Tests pass with `uv run pytest -q`.
- Documentation records the GPU acceleration status and the 1M-10M scale
  analysis.

## Completed Artifacts

- Plot comparison script:
  `examples/cellchat_plot_comparison.py`.
- Python-vs-R plot comparison PDF:
  `cellchat_plot_comparison/pyccc_python_vs_cellchat_r_plots_a4.pdf`.
- Plot comparison notes:
  `cellchat_plot_comparison/pyccc_python_vs_cellchat_r_plots_notes.md`.
- Individual pyccc Python PNGs:
  `cellchat_plot_comparison/pyccc_python_figures/`.
- Individual CellChat R PNGs and R manifest:
  `cellchat_plot_comparison/cellchat_r_figures/`.
- Numeric CellChat R parity PDF and metrics are preserved:
  `cellchat_reference_comparison/pyccc_vs_cellchat_r_a4.pdf`,
  `cellchat_reference_comparison/pyccc_vs_cellchat_r_metrics.tsv`.
- GPU and scaling analysis:
  `docs/gpu_and_million_cell_scaling.md`.

## Plot Comparisons Verified

The generated A4 plot report has 13 pages: 1 summary page plus 12 paired
analysis pages. `pdfinfo` confirms A4 page size. The R manifest reports `ok`
for every R-generated figure.

Included paired analyses:

- Overall source-target circle network.
- CCL/CXCL ligand-receptor bubble.
- Source-target signaling heatmap.
- Pathway information-flow rank.
- MIF ligand-receptor contribution.
- Outgoing/incoming signaling role scatter.
- CXCL hierarchy network.
- CXCL circle network.
- CXCL chord network.
- CXCL LR-mediated chord.
- `LS` vs `NL` rankNet comparison.
- `LS` vs `NL` stacked rankNet comparison.

The R-side figures are produced by CellChat R functions including
`netVisual_circle`, `netVisual_bubble`, `netVisual_heatmap`,
`netAnalysis_contribution`, `netAnalysis_signalingRole_scatter`,
`netVisual_aggregate`, `netVisual_chord_gene`, and `rankNet`.

## Acceleration Work Completed

- `compute_communication(..., array_backend=...)` now accepts `cpu`, `numpy`,
  `auto`, and `cupy`.
- `array_backend="cupy"` attempts CuPy for the vectorized LR probability tensor.
- If CuPy or CUDA is unavailable, pyccc warns and falls back to CPU.
- `PYCCC_ARRAY_BACKEND=cupy` can select the same behavior from the environment.
- Regression coverage verifies invalid backend validation and CuPy fallback
  equality with CPU.
- The current environment does not have `cupy` or `cudf`; `nvidia-smi` reports
  an NVML driver/library mismatch, so real GPU speedup was not benchmarked in
  this run.

## Scale Analysis Summary

CellChat's 1M-10M cell bottlenecks are dominated by raw-cell handling and
bootstrapping:

- dense `as.matrix(object@data.signaling)`;
- whole signaling matrix scaling and transposed aggregation;
- `nboot` permutations over all cells with repeated group summaries;
- ordinary in-memory R object storage;
- plotting assumptions that do not scale to many groups or dense edges.

pyccc can realistically improve this by subsetting LR/cofactor genes before
aggregation, preserving sparse/grouped paths, using sketches for stability and
p-values, batching LR scoring/output, and optionally using CuPy for the
group-level probability tensor. Exact all-cell bootstrap p-values at 10M cells
remain impractical without approximation.

## Verification

```bash
uv run python examples/cellchat_plot_comparison.py
uv run python examples/cellchat_reference_comparison.py
uv run pytest -q
```

Latest observed results:

- Plot comparison: 12/12 R/Python pairs rendered, 13-page A4 PDF.
- Numeric CellChat R parity:
  - LR-source-target probability Pearson/Spearman/top20 all `1.0`, max abs diff
    `4.996004e-16`.
  - Global network Pearson/Spearman/top20 all `1.0`, max abs diff
    `9.992007e-16`.
  - Pathway flow, LR flow, and MIF contribution also match at floating-point
    roundoff.
- Tests: `30 passed, 2 warnings in 33.94s`; warnings are UMAP fixed
  `random_state` parallelism warnings.

## CUDA Copy-Scaling Continuation

Completed on 2026-06-03 against the current `.venv` and local NVIDIA L40.

The CUDA stack is now usable from the project venv:

- `cupy-cuda12x==14.1.1`
- `cudf-cu12==25.6.0`
- CUDA 12.8 runtime/NVRTC/NVCC/NVJitLink wheels
- `nvidia-cusparse-cu12==12.5.8.93` for CuPy sparse group aggregation

The venv contains `pyccc_cuda_preload.pth` and `pyccc_cuda_preload.py` so
Python preloads CUDA 12.x libraries from the venv instead of mixing with older
system CUDA libraries.

New reproducible benchmark:

```bash
.venv/bin/python examples/cuda_copy_scaling_benchmark.py \
  --groupby Predicted_labels_CellTypist \
  --max-scale 128 \
  --n-groups 91 \
  --max-per-group 20 \
  --aggregate tri_mean \
  --all-annotations \
  --out-dir data/cellxgene/cuda_copy_scaling_benchmark
```

The benchmark physically copies the selected CELLxGENE sparse matrix by powers
of two and compares full `compute_communication` CPU vs CuPy runs. It completed
1x through 128x without OOM. Output artifacts:

- `examples/cuda_copy_scaling_benchmark.py`
- `data/cellxgene/cuda_copy_scaling_benchmark/cuda_copy_scaling_benchmark.tsv`
- `data/cellxgene/cuda_copy_scaling_benchmark/cuda_copy_scaling_benchmark_summary.md`
- `data/cellxgene/cuda_copy_scaling_benchmark/environment.json`

Latest observed copy-scaling result:

| Scale | Cells | CPU seconds | CuPy seconds | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1x | 1,818 | 1.856 | 1.708 | 1.09x |
| 2x | 3,636 | 1.922 | 1.259 | 1.53x |
| 4x | 7,272 | 2.036 | 1.377 | 1.48x |
| 8x | 14,544 | 2.266 | 1.607 | 1.41x |
| 16x | 29,088 | 2.854 | 2.192 | 1.30x |
| 32x | 58,176 | 3.842 | 3.187 | 1.21x |
| 64x | 116,352 | 6.061 | 5.394 | 1.12x |
| 128x | 232,704 | 12.329 | 11.662 | 1.06x |

CPU and CuPy produced the same interaction count (`191738`) and identical
probability sums at each scale. CuPy fallback was `False` for every row and no
CuPy warnings were emitted.

An additional explicit 640x run also completed without OOM:

```bash
.venv/bin/python examples/cuda_copy_scaling_benchmark.py \
  --groupby Predicted_labels_CellTypist \
  --scale 640 \
  --n-groups 91 \
  --max-per-group 20 \
  --aggregate tri_mean \
  --all-annotations \
  --out-dir data/cellxgene/cuda_copy_scaling_benchmark_640x
```

Result: 1,163,520 copied cells, CPU `51.802 s`, CuPy `51.133 s`, speedup
`1.01x`. CPU/CuPy interaction counts were both `191738`, probability sums
matched exactly, and CuPy fallback remained `False`.

Additional explicit 640x `gated_mean` run:

```bash
.venv/bin/python examples/cuda_copy_scaling_benchmark.py \
  --groupby Predicted_labels_CellTypist \
  --scale 640 \
  --n-groups 91 \
  --max-per-group 20 \
  --aggregate gated_mean \
  --clip-quantile 0.99 \
  --all-annotations \
  --out-dir data/cellxgene/cuda_copy_scaling_benchmark_640x_gated
```

Result: 1,163,520 copied cells, CPU `11.443 s`, CuPy `9.569 s`, speedup
`1.20x`. CPU/CuPy interaction counts were both `144311`, probability sums
matched to floating-point roundoff (`6071.316595781537` vs
`6071.316595782698`), and fallback remained `False`.

## GPU-Friendly Mean And Batched Resampling Continuation

Completed on 2026-06-03 against the same `.venv` and NVIDIA L40.

Implementation updates:

- Added `aggregate="clipped_mean"` with `clip_quantile` for a GPU-friendly
  robust mean alternative.
- Added CuPy sparse group aggregation for `mean` and `clipped_mean`:
  a sparse group-indicator matrix is multiplied by the cached sparse expression
  matrix on GPU instead of repeatedly copying and aggregating expression on CPU.
- Added batched CuPy permutation scoring for `mean`/`clipped_mean` when spatial
  weighting is not requested.
- Added batched CuPy repeated-sketch scoring for `mean`. The final sketch
  summary is now reduced directly from GPU tensors, avoiding the previous
  `pd.concat` plus `groupby` over per-repeat interaction tables.
- Added the same repeated-sketch summary optimization for CPU `mean`, using
  SciPy sparse group aggregation and chunked numpy LR scoring to avoid the
  per-repeat interaction-table concat/groupby bottleneck without requiring all
  LR pairs to be materialized in one CPU tensor.
- Added `aggregate="gated_mean"`, a GPU-friendly positive clipped mean with
  CellChat triMean-style detection gates at 25%, 50%, and 75% group expression.
  It keeps the dropout behavior closer to `tri_mean` while avoiding full
  group-wise quantile sorting.
- Added `pyccc.export_cellchat(...)` plus `pyccc_to_cellchat.R` to export pyccc
  results back into a minimal CellChat R `.rds` object for R-native plotting.
  The bridge fills `net`, `netP`, `LR`, `DB`, and `idents`; expression slots are
  intentionally empty.
- Added `pyccc.export_cellchat_merged(...)` plus
  `pyccc_to_merged_cellchat.R` for two-sample diff workflows. The export writes
  both single-sample CellChat bridge directories, pyccc differential TSVs, and
  an R helper that builds a CellChat `mergeCellChat(...)` object and optional
  CellChat comparison plots.
- Added package-local copies of both R helpers under `src/pyccc/r/` so source
  checkouts use `inst/r` while installed packages can still copy the helpers.
- Kept CPU fallback behavior for unsupported combinations. Exact
  `clipped_mean` repeated sketches still use the existing loop because each
  sketch needs its own clipping/scaling semantics.

New reproducible comparison:

```bash
.venv/bin/python examples/clipped_mean_comparison.py \
  --all-annotations \
  --out-dir data/cellxgene/clipped_mean_comparison
```

Latest observed 1x clipped-mean comparison:

| Method | Seconds | Interactions | Fallback |
| --- | ---: | ---: | --- |
| tri_mean CPU | 1.903 | 191,738 | False |
| tri_mean CuPy | 1.197 | 191,738 | False |
| clipped p99 CPU | 3.512 | 1,296,270 | False |
| clipped p99 CuPy | 2.777 | 1,296,270 | False |
| clipped p95 CPU | 3.505 | 1,296,270 | False |
| clipped p95 CuPy | 2.777 | 1,296,270 | False |
| gated p99 CPU | 1.760 | 144,311 | False |
| gated p99 CuPy | 1.037 | 144,311 | False |
| gated p95 CPU | 1.771 | 144,311 | False |
| gated p95 CuPy | 1.039 | 144,311 | False |

Observed p99 vs p95 agreement was very high: LR/source/target probability
Pearson `0.999818`, Spearman `0.999784`, top100 overlap `0.94`, top500 overlap
`0.958`, network Pearson `0.999769`, and pathway Pearson `0.999997`.

Compared with `tri_mean`, the clipped means were moderately similar at the
full LR/source/target level and more similar after network/pathway aggregation:

| Comparison | Pearson | Spearman | Top100 overlap | Top500 overlap | Network Pearson | Pathway Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tri_mean vs clipped p99 | 0.912624 | 0.558871 | 0.73 | 0.884 | 0.850000 | 0.933950 |
| tri_mean vs clipped p95 | 0.912234 | 0.559087 | 0.70 | 0.870 | 0.850977 | 0.933667 |
| tri_mean vs gated p99 | 0.977044 | 0.928537 | 0.81 | 0.836 | 0.986223 | 0.999263 |
| tri_mean vs gated p95 | 0.976851 | 0.928598 | 0.74 | 0.830 | 0.986237 | 0.999330 |

`gated_mean` gave a much closer LR/source/target ranking than `clipped_mean`
while staying faster than `tri_mean` on this 1x subset. CPU/CuPy `gated_mean`
matched exactly at floating-point precision: p99 max abs diff `2.22e-16`, p95
max abs diff `1.94e-16`.

New reproducible batched-resampling benchmark:

```bash
.venv/bin/python examples/gpu_batched_resampling_benchmark.py \
  --all-annotations \
  --out-dir data/cellxgene/gpu_batched_resampling_benchmark
```

Latest observed batched-resampling result on the same 1x subset:

| Task | CPU seconds | CuPy seconds | Speedup | Interactions | Fallback |
| --- | ---: | ---: | ---: | ---: | --- |
| 8 permutations | 30.918 | 5.018 | 6.16x | 1,296,270 | False |
| 8 repeated sketches | 14.479 | 2.746 | 5.27x | 1,270,294 | False |

CPU/CuPy agreement remained at floating-point roundoff:

- Permutations: same interaction set, probability Pearson `1.0`, max abs diff
  `4.36e-08`.
- Repeated sketches: same interaction set, probability Pearson `1.0`, max abs
  diff `2.26e-15`.

Verification:

```bash
.venv/bin/python -m pytest -q
```

Latest observed CellChat bridge smoke:

```bash
.venv/bin/python - <<'PY'
from tests.test_core import make_adata
import pyccc as pc
adata = make_adata()
lr = pc.toy_lr_table()
res = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", score_method="cellchat")
pc.export_cellchat(res, "data/cellchat_export_smoke", lr_table=lr, group_sizes=adata.obs["cell_type"].value_counts())
PY
R_LIBS_USER=/home/mcp2/pyccc/.r-lib Rscript \
  data/cellchat_export_smoke/pyccc_to_cellchat.R \
  data/cellchat_export_smoke \
  data/cellchat_export_smoke/pyccc_cellchat.rds \
  data/cellchat_export_smoke/plots
```

The smoke run wrote `pyccc_cellchat.rds` plus CellChat-generated
`network_circle`, `bubble`, `pathway_heatmap`, `pathway_rank`, and
`pathway_circle` PNGs under `data/cellchat_export_smoke/plots/`.

Latest observed merged CellChat bridge smoke:

```bash
.venv/bin/python - <<'PY'
from tests.test_core import make_adata
import pyccc as pc
adata = make_adata()
lr = pc.toy_lr_table()
ctrl = adata[adata.obs["condition"] == "ctrl"].copy()
stim = adata[adata.obs["condition"] == "stim"].copy()
diff = pc.compare_samples(stim, ctrl, "cell_type", lr, label_a="stim", label_b="ctrl", min_pct=0.0, aggregate="mean", score_method="cellchat")
pc.export_cellchat_merged(
    diff,
    "data/cellchat_merged_export_smoke",
    lr_table=lr,
    group_sizes_a=stim.obs["cell_type"].value_counts(),
    group_sizes_b=ctrl.obs["cell_type"].value_counts(),
)
PY
R_LIBS_USER=/home/mcp2/pyccc/.r-lib Rscript \
  data/cellchat_merged_export_smoke/pyccc_to_merged_cellchat.R \
  data/cellchat_merged_export_smoke \
  data/cellchat_merged_export_smoke/pyccc_merged_cellchat.rds \
  data/cellchat_merged_export_smoke/plots
```

The smoke run wrote `pyccc_merged_cellchat.rds`, pyccc differential TSVs, and
CellChat-generated comparison plot manifest/PNGs under
`data/cellchat_merged_export_smoke/plots/`. In the toy smoke, five comparison
plots were `ok`; `diff_interaction_count_circle` was correctly marked
`skipped` because the two samples had no nonzero count-network delta.

Latest observed tests: `37 passed, 2 warnings in 43.39s`; warnings are UMAP
fixed `random_state` parallelism warnings.
