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
)
```

The report contains:

- `summary`: LR-pair spatial enrichment scores, z-scores, and empirical p-values,
- `celltype_pair_summary`: source-target scores by LR pair and kernel,
- `null_distribution`: permutation scores,
- `distance_decay`: distance-bin metadata,
- `metadata`: radius, sigma, null model, and permutation settings.

Supported kernels:

- `contact`: cells are connected when distance is within `radius`,
- `exp`: secreted/diffusion-style `exp(-distance / sigma)`.

Supported null models:

- `coordinate_permutation`,
- `celltype_permutation`,
- `matched_random_lr`: samples ligand and receptor genes from matching global
  expression quantiles,
- `score_permutation`: permutes model scores before recomputing the reported
  score-aware summaries.

`null_distribution` is tracked by ligand, receptor, kernel, and `score_type`.
The default score types are `spatial_ccc_score` and
`model_weighted_spatial_ccc_score`. Empirical p-values use pair-specific nulls
when available and fall back to kernel-level nulls otherwise. `distance_decay`
reports distance-bin summaries with `mean_spatial_ccc_score` for each LR pair.

For Stereo-seq/cellbin data, provide centroid coordinates in `obsm["spatial"]`.
If cell area is available, `radius="auto"` estimates a contact radius from the
median cell area; otherwise it uses the median nearest-neighbor distance.
