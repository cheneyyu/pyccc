# Spatial Validation

Spatial validation tests whether candidate LR-driven CCC scores are enriched
under spatial proximity compared with matched null models. It is plausibility
evidence, not biochemical validation.

Inputs:

- `adata.obsm["spatial"]` with cell coordinates,
- cell type labels in `adata.obs[groupby]`,
- a curated or predicted LR table,
- optional cell area metadata in `adata.obs["cell_area"]`.

```python
report = pc.validate_spatial_lr_table(
    adata,
    lr_table=predicted_db,
    groupby="cell_type",
    spatial_key="spatial",
    mode="cellbin",
    radius="auto",
    sigma="auto",
    n_permutations=1000,
    section_key="section_id",
    top_k=(100, 500, 1000),
    curated_lr_table=curated_db,  # optional same-species curated reference
)
```

The report contains:

- `summary`: LR-pair spatial enrichment scores, z-scores, and empirical p-values,
- `celltype_pair_summary`: source-target scores by LR pair and kernel,
- `null_distribution`: permutation scores,
- `distance_decay`: distance-bin metadata,
- `section_reproducibility`: optional cross-section reproducibility summaries,
- `top_k_enrichment`: top predicted LR-pair enrichment summaries for each
  kernel, score type, and requested K,
- `role_kernel_enrichment`: optional role-aware summaries that compare
  secreted-like pairs under the `exp` kernel and membrane/contact-like pairs
  under the `contact` kernel against role-score background pairs,
- `curated_overlap_enrichment`: optional curated-overlap summaries when a
  same-species curated LR table is supplied,
- `metadata`: radius, sigma, null model, and permutation settings.

Supported kernels:

- `contact`: cells are connected when distance is within `radius`,
- `exp`: secreted/diffusion-style `exp(-distance / sigma)`.

Supported null models:

- `coordinate_permutation`,
- `celltype_permutation`: permutes cell-type labels globally, or within
  `section_key` strata when section/sample labels are provided,
- `matched_random_lr`: samples ligand and receptor genes by matching global
  expression quantile, role score, and LR-table gene degree,
- `score_permutation`: permutes model scores before recomputing the reported
  score-aware summaries.

`null_distribution` is tracked by ligand, receptor, kernel, and `score_type`.
The default score types are `spatial_ccc_score` and
`model_weighted_spatial_ccc_score`. Empirical p-values use pair-specific nulls
when available and fall back to kernel-level nulls otherwise. `distance_decay`
reports distance-bin summaries with `mean_spatial_ccc_score` for each LR pair.
For matched-random nulls, the original LR pair remains the statistical key and
the sampled genes are recorded in `matched_ligand` and `matched_receptor`
alongside expression, role-score, and degree match deltas.
`top_k_enrichment` ranks predicted LR pairs by `model_score` when available,
then reports the observed top-K mean score, null mean/SD, z-score, and
empirical p-value for K values such as 100, 500, and 1000.
If predicted LR tables contain `ligand_secreted_like_score`,
`ligand_membrane_like_score`, or `receptor_membrane_like_score`,
`role_kernel_enrichment` reports whether secreted-like pairs are preferentially
enriched under the diffusion-style `exp` kernel and whether membrane/contact-like
pairs are enriched under the `contact` kernel.
When `curated_lr_table` is provided, pyccc marks predicted LR pairs that overlap
the curated table in `summary["curated_overlap"]` and reports whether those
overlapping pairs have stronger spatial scores than non-overlapping predicted
pairs in `curated_overlap_enrichment`.

For Stereo-seq/cellbin data, provide centroid coordinates in `obsm["spatial"]`.
If cell area is available, `radius="auto"` estimates a contact radius from the
median cell area; otherwise it uses the median nearest-neighbor distance.

When `section_key` is provided, pyccc recomputes observed spatial scores within
each section or sample and reports `n_sections`, cross-section score mean and
standard deviation, positive-section fraction, top-K-section fraction, and
median section rank for each LR pair and kernel. The cell-type permutation null
also stays within sections, preserving section-level composition while testing
whether labels explain the spatial signal. Other null p-values remain computed
from the full object unless you run validation separately per section.
