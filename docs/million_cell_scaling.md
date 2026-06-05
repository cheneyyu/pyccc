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
README runtime figure. The detailed local result table is
`data/runtime_benchmark/human_immune_health_atlas_real1m_raw_4core_prepared/runtime.tsv`.

## Current Optimization Snapshot

The public runtime figure uses a strict 4-core prepared-input run:
`taskset -c 0-3`, `OPENBLAS_NUM_THREADS=4`, `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`, and `pyccc n_jobs=1`.
The 1M-cell sampled h5ad, LR table, and CellChat MatrixMarket directory are
generated once as local ignored artifacts; sampling from the original 1.82M-cell
h5ad and LR preparation are not counted. The timed pyccc rows include reading
the prepared sampled h5ad. The direct CellChat R row starts before `readMM()`,
so its MatrixMarket input read is included in its total.

| Strategy | Total from prepared input seconds | In-memory workflow seconds | Compute seconds | Peak CPU cores | Speedup vs direct CellChat |
| --- | ---: | ---: | ---: | ---: | ---: |
| pyccc native | 11.244 | 9.691 | 8.129 | 1.33 | 17.39x |
| pyccc + CellChat R plots | 27.535 | 25.982 | 8.094 | 1.87 | 7.10x |
| direct CellChat R | 195.549 | 195.549 | 157.183 | 2.31 | 1.00x |

The main pyccc bottleneck was CellChat-compatible tri-mean expression
aggregation on sparse input. The optimized path computes exact zero-aware sparse
quantiles for `tri_mean`, keeping regular-mode 1M-cell compute near eight
seconds under the 4-core benchmark. `n_jobs` is not a meaningful control for
this regular, non-permutation path; reserve Python worker scaling tests for
permutation or repeated-downsampling workflows.

## Reproduce

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

The prepared h5ad and MatrixMarket files stay under `data/runtime_benchmark/`
and are intentionally ignored by Git.

## Practical limits

For very large datasets, the primary bottlenecks are data loading, expression
subsetting, group aggregation, permutation-style resampling, and R-side object
materialization. pyccc improves the common path by keeping data in AnnData and
reducing early to compact group-level summaries. Exact all-cell bootstrap
procedures can still become impractical at multi-million-cell scale; for those
cases, prefer matched downsampling, repeated sketches, or effect-size driven
screening before expensive resampling.
