# Million-cell scaling

pyccc's supported large-data path is CPU/NumPy/SciPy based. The main scaling
choice is to reduce single-cell expression to group-level ligand, receptor, and
cofactor summaries before scoring source-target communication. This keeps the
expensive step proportional to the number of genes used by the ligand-receptor
database and the number of cell groups, rather than the full cell-by-gene
matrix.

## Current 1M-cell benchmark

The current large benchmark uses the CELLxGENE Human Immune Health Atlas:

- Dataset explorer:
  <https://cellxgene.cziscience.com/e/e522d2cd-7927-4e59-a4ed-064009569279.cxg/>
- Local input used for the benchmark:
  `data/cellxgene/human_immune_health_atlas_1p82m.h5ad`
- Observed shape: `1,821,725 x 32,357`
- Comparison: `disease == "cytomegalovirus infection"` versus `normal`
- Grouping: `cell_type`
- Cell sample: `1,000,000` real cells sampled without replacement from the top
  five shared cell types
- Matrix: `raw.X`, because the h5ad main `X` matrix is scaled and includes
  negative values

The benchmark runs the same within-sample CCC, two-condition differential CCC,
and matched visualization workflow through the three strategies shown in the
README runtime figure. The committed detailed result table is
`data/runtime_benchmark/human_immune_health_atlas_real1m_raw/adaptive_runtime.tsv`.

## Reproduce

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

If the target size cannot complete because of memory pressure or timeout, rerun
at half the target cell count and repeat that halving step until all three
strategies produce usable rows.

## Practical limits

For very large datasets, the primary bottlenecks are data loading, expression
subsetting, group aggregation, permutation-style resampling, and R-side object
materialization. pyccc improves the common path by keeping data in AnnData and
reducing early to compact group-level summaries. Exact all-cell bootstrap
procedures can still become impractical at multi-million-cell scale; for those
cases, prefer matched downsampling, repeated sketches, or effect-size driven
screening before expensive resampling.
