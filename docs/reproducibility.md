# Reproducibility

This document records the commands needed to reproduce the package benchmarks
and example figures used in the repository README and documentation.

## Environment

Install the Python package and test dependencies:

```bash
uv sync --extra dev
uv run pytest -q
```

Optional R-side CellChat comparisons require a local R installation with
CellChat and its plotting dependencies. The Python tests skip R smoke tests
when that local R library is unavailable.

## Official CellChat Parity

The numeric parity check uses the official CellChat human skin tutorial data
and the human CellChatDB Secreted Signaling subset:

```bash
uv run python examples/cellchat_reference_comparison.py
```

Expected local outputs:

- `cellchat_reference_comparison/pyccc_vs_cellchat_r_metrics.tsv`
- `cellchat_reference_comparison/pyccc_vs_cellchat_r_a4.pdf`

The current parity snapshot reports Pearson, Spearman, and top-20 overlap of
`1.0` for LR-source-target probabilities, global network weights, pathway
information flow, LR information flow, and MIF LR contribution.

## Real 1M-cell Benchmark

Dataset:

- CELLxGENE collection: Human Immune Health Atlas
- Dataset explorer:
  <https://cellxgene.cziscience.com/e/e522d2cd-7927-4e59-a4ed-064009569279.cxg/>
- Local h5ad path used in this run:
  `data/cellxgene/human_immune_health_atlas_1p82m.h5ad`
- Observed shape: `1,821,725 x 32,357`
- Conditions: `disease == "cytomegalovirus infection"` versus `normal`
- Grouping: `cell_type`

The h5ad's main `X` matrix is scaled and contains negative values, which is not
a valid direct CellChat input. The benchmark therefore prepares a local
1,000,000-cell `raw.X` sample, matching LR table, and CellChat MatrixMarket
directory first. Those prepared artifacts are ignored by Git. The timed public
run starts from reading those prepared inputs:

```bash
taskset -c 0-3 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
uv run python examples/three_way_runtime_benchmark.py \
  --mode prepared \
  --out-dir data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core_prepared \
  --h5ad data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core/prepared/human_immune_health_atlas_1m_top5_lrgenes.h5ad \
  --lr-table data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core/prepared/pyccc_lr.tsv \
  --r-input-dir data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core/cells_1000000/r_input \
  --condition-key disease \
  --condition-a "cytomegalovirus infection" \
  --condition-b normal \
  --groupby cell_type \
  --gene-symbols-key feature_name \
  --n-jobs 1 \
  --timeout-seconds 1200
```

Expected local output:

- `data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core_prepared/runtime.tsv`
- `data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core_prepared/input_runtime.tsv`

The top five cell types shared between conditions cover 1,100,447 real cells,
so the 1,000,000-cell benchmark is sampled without replacement.
