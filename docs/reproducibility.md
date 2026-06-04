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
a valid direct CellChat input. The benchmark therefore uses `raw.X` for all
three strategies:

```bash
uv run python examples/three_way_runtime_benchmark.py \
  --mode cellxgene \
  --out-dir data/runtime_benchmark/human_immune_health_atlas_real1m_raw \
  --h5ad data/cellxgene/human_immune_health_atlas_1p82m.h5ad \
  --use-raw \
  --condition-key disease \
  --condition-a "cytomegalovirus infection" \
  --condition-b normal \
  --groupby cell_type \
  --gene-symbols-key feature_name \
  --target-cells 1000000 \
  --min-cells 125000 \
  --n-groups 5 \
  --timeout-seconds 600
```

Expected local output:

- `data/runtime_benchmark/human_immune_health_atlas_real1m_raw/adaptive_runtime.tsv`

The top five cell types shared between conditions cover 1,100,447 real cells,
so the 1,000,000-cell benchmark is sampled without replacement.
