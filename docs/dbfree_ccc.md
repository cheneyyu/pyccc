# Experimental DB-Free CCC

DB-free target-species CCC means the target species does not need a curated
ligand-receptor database. The predictor still learns from curated resources in
other species, then generates a candidate LR table for the target species from
protein sequences and expression constraints.

This is experimental. Predicted LR rows are computational candidates, not
validated biochemical interactions.

The intended production stack is:

```text
ESMC-300M mean-pooled protein embeddings
  -> LightGBM protein role classifiers
  -> LightGBM pair ranker
  -> clade-aware LR-density prior
  -> CellChatDB-compatible predicted LR table
```

Install prediction dependencies only when needed:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath,predict] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

## Inputs

Required inputs:

- AnnData expression matrix,
- gene identifiers matching `adata.var_names` or `adata.var[gene_id_key]`,
- CDS FASTA or longest protein FASTA,
- trained role and LR link predictor models, or explicit candidate lists.

## Sequence Preparation

```python
proteins = pc.load_cds_translations(
    "species.longest_cds.fa",
    gene_id_regex=r"gene=([^\s]+)",
    transcript_id_regex=r"transcript=([^\s]+)",
    select="longest",
)

match = pc.match_expression_genes(adata, proteins, gene_id_key="gene_id")
```

Protein FASTA is also supported:

```python
proteins = pc.load_protein_fasta("species.longest_protein.fa")
```

## Embeddings and Roles

```python
emb = pc.embed_proteins_esmc(
    proteins,
    model_name="biohub/ESMC-300M",
    pooling="mean",
    cache_dir=".pyccc-cache/esmc",
)

roles = pc.predict_protein_roles(
    proteins=proteins,
    embeddings=emb,
    model="models/universal_esmc300m_role_v0",
)
```

The cache key includes sequence hash, model name, model revision, and pooling.

## Candidate Generation and Scoring

```python
candidates = pc.generate_lr_candidates_dbfree(
    adata=adata,
    proteins=proteins,
    roles=roles,
    embeddings=emb,
    gene_id_key="gene_id",
    expression_min_fraction=0.02,
    ligand_role_min=0.30,
    receptor_role_min=0.30,
    max_ligands=3000,
    max_receptors=3000,
    max_candidate_pairs=5_000_000,
    nearest_neighbor_pairs=0,
)

scores = pc.score_lr_candidates(
    candidate_pairs=candidates,
    embeddings=emb,
    model="models/universal_esmc300m_lgbm_v0",
)
```

`model_score` is the raw rank score used for top-K density filtering.
`calibrated_probability` is produced by the held-out calibrator when the saved
model includes one; otherwise it falls back to `model_score`.

By default, candidate generation uses a role/expression cross product. Set
`nearest_neighbor_pairs` above zero to reserve part of the candidate budget for
high-cosine ligand/receptor neighbors in ESMC embedding space. These rows keep
`candidate_strategy="embedding_nearest_neighbor"` or a semicolon-separated
combined strategy after deduplication, so downstream tables preserve where each
candidate came from.

Users can bypass role prediction with explicit candidates:

```python
candidates = pc.generate_lr_candidates_dbfree(
    adata,
    proteins,
    roles,
    gene_id_key="gene_id",
    ligand_candidates="known_secreted_genes.txt",
    receptor_candidates="known_surface_genes.txt",
)
```

## Density Thresholding

```python
density = pc.estimate_lr_density_prior(train.interactions, groupby="clade")

predicted_db = pc.build_predicted_lr_table(
    scored_pairs=scores,
    roles=roles,
    density_prior=density,
    species_hint="plant",
    min_score=0.50,
    max_pairs=50000,
)
```

The output is a normal `CellChatDB` object. It includes `model_score`,
`calibrated_probability`, `density_prior`, `density_rank`, confidence,
provenance columns, ligand/receptor role scores, secreted/membrane role scores,
nearest-reference annotations, warnings, and prediction summary metadata. The
summary also records the target pair count, capped target count, achieved
density, density delta, and density ratio.

When a trained pair ranker is used, `score_lr_candidates(...)` annotates each
candidate with the nearest curated positive LR pair in the ranker feature
space: `nearest_reference_ligand`, `nearest_reference_receptor`,
`nearest_reference_lr`, `nearest_reference_species`,
`nearest_reference_resource`, `nearest_reference_pathway`, and
`nearest_reference_distance`. The primary pathway remains
`DB-free predicted`; nearest-reference pathway is interpretive provenance only.
The `warning` column records fallback choices such as heuristic rankers,
heuristic role models, hash embeddings, or default density priors.

Check whether the selected table remains close to the prior:

```python
density_gates = pc.evaluate_predicted_lr_density_prior(
    predicted_db,
    max_fold_error=2.0,
)
print(density_gates)
print(density_gates.attrs["passed"])
```

## End-to-End Wrapper

```python
predicted_db = pc.predict_lr_dbfree(
    adata,
    protein_fasta="target.longest_protein.fa",
    gene_id_key="gene_id",
    species_name="target_species",
    species_hint="unknown",
    role_model="models/universal_esmc300m_role_v0",
    model="models/universal_esmc300m_lgbm_v0",
    density_prior="auto",
    min_score=0.50,
    max_pairs=50000,
    max_pairs_per_ligand=200,
    max_pairs_per_receptor=200,
    cache_dir=".pyccc-cache",
)

res = pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=predicted_db,
    gene_symbols_key="gene_id",
    score_method="cellchat",
)
```

`compute_communication` remains deterministic. It does not infer LR pairs
silently; prediction is an explicit upstream step.

`predict_lr_dbfree(...)` forwards density-threshold parameters to
`build_predicted_lr_table(...)`, including `min_score`, `max_pairs`,
`max_pairs_per_ligand`, `max_pairs_per_receptor`, and
`allow_low_score_density_fill`.

With `density_prior="auto"`, `predict_lr_dbfree(...)` first looks for
`density_prior.tsv` in the LR ranker model directory and uses the row matching
`species_hint` by `clade` when available. If a density table exists but no row
matches the target hint, pyccc falls back to the median prior across table rows
and records `density_mode="table_fallback_median"`. If no trained-model prior
is present, it falls back to a conservative default density.

## Limitations

- Do not interpret candidate rows as validated LR biology.
- Do not invent curated pathway labels; predicted rows use
  `pathway = "DB-free predicted"`.
- Use leave-species/resource/family validation before making biological claims.
- Use spatial validation as plausibility evidence only.
