# GPU acceleration and million-cell CCC scaling

## Current GPU Status

pyccc now has an optional `array_backend` argument on `compute_communication`:

```python
pc.compute_communication(adata, "cell_type", lr_db, array_backend="cupy")
```

Supported values are:

- `cpu` or `numpy`: always use NumPy.
- `cupy`: try CuPy for the vectorized LR probability tensor and fall back to
  CPU with a warning if CuPy or CUDA is unavailable.
- `auto`: use CuPy only when it imports and can see a CUDA device; otherwise
  use CPU.

The same backend can also be selected with:

```bash
PYCCC_ARRAY_BACKEND=cupy
```

This is intentionally optional. CuPy and cuDF are not required dependencies.
The current local benchmark environment has `cupy-cuda12x==14.1.1`,
`cudf-cu12==25.6.0`, CUDA 12.8 runtime/NVRTC wheels, and an NVIDIA L40 visible
through CuPy. The venv preloads CUDA runtime/NVRTC/NVJitLink/cuSPARSE libraries
from `.venv/lib/python3.10/site-packages/nvidia` so CuPy and cuDF do not pick
up older system CUDA libraries.

For routine two-sample differential CCC, CPU is still the recommended default.
The diff step compares group-level result tables and networks, so it is not a
GPU-bound operation. GPU should be treated as an experimental optional backend
for larger LR tensors, permutation batches, or repeated sketches.

## What CuPy Accelerates

The CuPy path accelerates the dense tensor operations after group-level
expression has already been computed:

- ligand and receptor outer products across source-target cell groups,
- CellChat Hill-transform probabilities,
- cofactor multiplication,
- population-size and spatial pair weighting,
- nonzero probability filtering.

This tensor has shape:

```text
n_ligand_receptor_pairs x n_cell_groups x n_cell_groups
```

For normal CellChat-style workflows this is much smaller than the raw cell by
gene matrix. It is still useful when there are many LR pairs, many groups, many
permutations, or repeated sketches, but it is not the dominant bottleneck for
most million-cell inputs.

For `aggregate="mean"`, pyccc can also attempt CuPy/cuSPARSE group-expression
aggregation for sparse AnnData inputs. This path is correct, but benchmarked
performance is workload-dependent because the expression slice must be copied
to the GPU. On the current CELLxGENE benchmark, CPU SciPy sparse aggregation is
competitive for low group counts, so the clearest CUDA gain comes from
CellChat-style LR tensor scoring when the number of groups and LR pairs is
large.

## Verified Copy-Scaling Benchmark

The reproducible benchmark is:

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

It uses the real CELLxGENE immune h5ad, keeps the top 91 CellTypist labels,
physically copies the selected sparse matrix by powers of two, and runs the
same full `compute_communication` workflow on CPU and CuPy. The run completed
through 128x without OOM:

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

No CuPy fallback warnings were emitted, and CPU/CuPy interaction counts and
probability sums matched for every scale.

An additional explicit 640x run used the same settings and completed without
OOM:

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

At 640x, the copied matrix represented 1,163,520 cells. CPU took 51.802 s and
CuPy took 51.133 s, for a 1.01x speedup. CPU/CuPy interaction counts were both
191,738 and probability sums matched exactly. This confirms CUDA still runs
correctly at million-cell copy scale, but the end-to-end advantage is mostly
absorbed by CPU-side `tri_mean` group aggregation.

The same explicit 640x copy-scale check with `aggregate="gated_mean"` and
`clip_quantile=0.99` gave a modest improvement:

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

At 640x, CPU took 11.443 s and CuPy took 9.569 s, for a 1.20x speedup.
Interaction counts were both 144,311 and probability sums matched to
floating-point precision. This is useful evidence that the GPU path is correct,
but it is not a decisive advantage for the standard no-permutation two-sample
diff workflow.

## Why cuDF Is Not In The Core Path Yet

cuDF is most useful for large tabular joins and groupbys. In pyccc, the largest
object is usually the expression matrix, not the output interaction table. The
result table has at most:

```text
n_ligand_receptor_pairs x n_source_groups x n_target_groups
```

That table is usually small enough for pandas. Moving it to cuDF would add a
large optional dependency and extra host/device transfers without addressing
the main scaling pressure. A future cuDF path could help if users deliberately
work with thousands of groups and retain very large long-form outputs, but it
should stay optional.

## Why CellChat Struggles At 1M-10M Cells

The main CellChat scaling issue is not the final group-by-group CCC tensor. It
is the path from raw cells to grouped signaling expression and permutation
statistics.

Key bottlenecks in CellChat R `computeCommunProb`:

- It materializes `object@data.signaling` as a dense matrix via `as.matrix`.
  With 1M-10M cells, even a restricted signaling-gene matrix can become too
  large for RAM.
- It scales the whole signaling matrix by `max(data)` before aggregation.
- It computes group summaries by aggregating transposed dense data.
- For p-values, it creates bootstrap permutations of all cells and recomputes
  group averages for each bootstrap.
- It stores probability and p-value arrays of shape
  `n_groups x n_groups x n_LR`. This part is manageable for tens of groups, but
  becomes large when users create hundreds or thousands of cell states.
- Many plotting functions assume ordinary in-memory R objects and are not built
  for huge numbers of groups or retained LR edges.

For 1M cells, exact no-permutation CellChat-style scoring may still be possible
if the signaling-gene matrix is sparse and the implementation avoids dense
copies. For 10M cells, exact bootstrap inference over all cells becomes
impractical unless the workflow changes.

## What pyccc Can Realistically Overcome

pyccc has a better route to large data because the statistical object needed for
CellChat-style CCC is group-level expression, not all cell-level values after
aggregation.

Already useful:

- It subsets expression to LR/cofactor genes before aggregation.
- It keeps sparse matrix paths for group summaries.
- It supports sketching with `downsample_per_group` and repeated sketches.
- It can skip permutations and still compute deterministic communication
  probabilities.
- It now has optional CuPy acceleration for the LR probability tensor.

Realistic next steps:

- Stream group expression summaries from backed or chunked AnnData instead of
  copying large cell matrices.
- Add sketch-first workflows for p-values and stability rather than exact
  all-cell bootstraps.
- Add GPU sparse group aggregation only when the selected expression slice fits
  device memory.
- Batch LR scoring and output writing for thousands of groups.

What remains hard:

- Exact CellChat bootstrap p-values over 10M cells are inherently expensive.
- Very large numbers of cell groups make the `groups x groups x LR` tensor grow
  quadratically.
- Plotting thousands of groups is a visualization problem, not just a compute
  problem; it requires aggregation or filtering.

## Practical Recommendation

For 1M-10M cells, pyccc should target this workflow:

1. Restrict to relevant LR/cofactor genes.
2. Compute exact or streaming group-level summaries.
3. Score all LR pairs at group level.
4. Use repeated per-group sketches for stability or approximate p-values.
5. Batch large output tables and plot only selected pathways/groups.

This can preserve the CellChat-style interpretation while avoiding the largest
R CellChat memory and bootstrap bottlenecks.
