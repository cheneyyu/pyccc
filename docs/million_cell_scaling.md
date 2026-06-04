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
`data/runtime_benchmark/human_immune_health_atlas_real1m_raw/adaptive_runtime.tsv`.

## Current Optimization Snapshot

The current pyccc runtime figure uses the same 1,000,000-cell sample and reuses
the original direct CellChat R row because the optimization only changes pyccc's
sparse tri-mean aggregation path. The benchmark was rerun with
`OPENBLAS_NUM_THREADS=64`, `OMP_NUM_THREADS=64`, `MKL_NUM_THREADS=64`,
`NUMEXPR_NUM_THREADS=64`, and `pyccc n_jobs=1`.

| Strategy | Pre-optimization total seconds | Current total seconds | Current compute seconds | Speedup vs pre-optimization | Speedup vs direct CellChat |
| --- | ---: | ---: | ---: | ---: | ---: |
| pyccc native | 47.014 | 8.627 | 7.188 | 5.45x | 23.31x |
| pyccc + CellChat R plots | 63.692 | 25.190 | 7.246 | 2.53x | 7.98x |
| direct CellChat R | 201.096 | 201.096 | 162.847 | 1.00x | 1.00x |

The main pyccc bottleneck before this change was CellChat-compatible tri-mean
expression aggregation on sparse input. A profile of the old path attributed
about 39 seconds of the 52-second profiled pyccc compute run to
`np.percentile`/`numpy.ndarray.partition` after densifying each group matrix.
The optimized path computes exact zero-aware sparse quantiles for `tri_mean`,
reducing the 1M-cell pyccc compute time from about 45.6 seconds to about
7.2 seconds on this run.

A compute-only debug run with `pyccc n_jobs=64` took 7.445 seconds versus
7.438 seconds with `n_jobs=1` on the same loaded AnnData object, confirming that
`n_jobs` is not a meaningful control for this regular, non-permutation path.
Use BLAS/OpenMP thread controls for this benchmark; reserve `n_jobs` scaling
tests for permutation or repeated-downsampling workflows.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64 \
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
  --n-jobs 1 \
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
