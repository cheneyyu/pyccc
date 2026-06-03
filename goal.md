# pyccc goal: DB-free target-species CCC with ESMC-300M link prediction

## One-line goal

Make pyccc able to run CellChat-style CCC for species without a curated ligand-receptor database by predicting a species-specific LR table from the expression matrix plus CDS/longest protein sequences, then passing that LR table into the existing deterministic CCC engine.

This is **DB-free for the target species**, not DB-free during model training. The training labels still come from curated LR resources such as CellChatDB, OmniPath/CellPhoneDB-style resources, FlyPhoneDB2, PlantPhoneDB, and PlantCellChat when available. The target species only needs:

1. an expression matrix, preferably AnnData,
2. gene identifiers matching the matrix,
3. CDS FASTA or longest translated protein FASTA,
4. optional spatial coordinates, especially Stereo-seq cellbin coordinates, for validation.

## Core decision

Use this architecture:

```text
curated LR resources across annotated species
        ↓
resource normalization + provenance + leakage-safe training splits
        ↓
ESMC-300M protein embeddings
        ↓
role prediction: ligand-like protein / receptor-like protein
        ↓
LightGBM pairwise link prediction over candidate ligand-receptor pairs
        ↓
clade-aware LR-density thresholding
        ↓
predicted CellChatDB-compatible LR table
        ↓
existing pyccc compute_communication(..., lr_table=predicted_db)
        ↓
optional spatial-distance validation on Stereo-seq/cellbin or other spatial data
```

Do **not** hide ML prediction inside `compute_communication`. Prediction should produce an explicit LR table with confidence/provenance columns. The CCC engine should remain deterministic once the LR table is supplied.

## Why this is possible but must be scoped carefully

The plan is feasible as a practical MVP because ESMC-300M provides fixed-length sequence representations for proteins, and a gradient-boosted classifier can learn whether two protein representations are compatible with known LR annotations. It is not a proof of biochemical binding. The output should always be called a **candidate LR table** or **computationally predicted LR table**.

The safest claim is:

> pyccc can perform DB-free CCC analysis for a target species by generating a predicted LR table from CDS/protein sequences and expression constraints, calibrated against curated LR densities from well-annotated species.

Do not claim:

> pyccc discovers true ligand-receptor biology de novo.

## Packaging decisions

### Recommended user install

Keep `omnipath` as an optional dependency, but recommend it by default in docs and tutorials:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

### Prediction install

Put all ML dependencies behind a separate experimental extra:

```toml
predict = [
  "torch>=2.2",
  "transformers>=4.45",
  "huggingface-hub>=0.24",
  "scikit-learn>=1.4",
  "lightgbm>=4.0",
  "joblib>=1.3",
]
```

Users who want DB-free prediction should install:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath,predict] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

Default pyccc tests must not import torch, transformers, ESMC, Hugging Face, or LightGBM unless the `predict` extra is installed.

## Definitions

### DB-backed CCC

A user runs CCC with a curated or imported LR table:

```python
res = pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=pc.load_cellchatdb("human"),
)
```

### Target-species DB-free CCC

A user runs CCC for a species that has no curated LR database:

```python
predicted_db = pc.predict_lr_dbfree(
    adata,
    cds_fasta="species.longest_cds.fa",
    gene_id_key="gene_id",
    species_name="custom_species",
    species_hint="plant",  # or "animal", "mammal", "insect", "unknown"
    model="universal_esmc300m_lgbm_v0",
    density_prior="auto",
    max_pairs=50000,
)

res = pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=predicted_db,
    gene_symbols_key="gene_id",
    score_method="cellchat",
)
```

The target species does not need a curated LR DB. The model still learned from curated LR DBs during training.

## Data sources to generalize

### Built-in / online-supported resources

- CellChatDB human and mouse through `pc.load_cellchatdb(...)`.
- OmniPath resources through `pc.load_omnipath_interactions(...)`.
- CellPhoneDB-style resources when exposed through OmniPath or user files.

### Local-file adapters for additional species resources

Add import adapters for these resources, but do not require live network fetching in v0:

- `schema="flyphonedb2"`
- `schema="plantphonedb"`
- `schema="plantcellchat"`
- `schema="cellchat"`
- `schema="omnipath"`
- `schema="generic"`

The first implementation should treat FlyPhoneDB2, PlantPhoneDB, and PlantCellChat as **user-supplied local tables** unless their public download/API formats are confirmed. This avoids brittle code and licensing problems.

Example:

```python
train_lr = pc.load_training_lr_resources([
    {"path": "resources/cellchat_human.tsv", "schema": "cellchat", "species": "human", "taxon_id": 9606},
    {"path": "resources/cellchat_mouse.tsv", "schema": "cellchat", "species": "mouse", "taxon_id": 10090},
    {"path": "resources/flyphonedb2.tsv", "schema": "flyphonedb2", "species": "drosophila", "taxon_id": 7227},
    {"path": "resources/plantphonedb.tsv", "schema": "plantphonedb", "species": "arabidopsis", "taxon_id": 3702},
    {"path": "resources/plantcellchat.tsv", "schema": "plantcellchat", "species": "rice", "taxon_id": 4530},
])
```

## Normalized training table

All resources must normalize to one long-form interaction table with these columns:

### Required columns

- `ligand_gene`
- `receptor_gene`
- `species`
- `taxon_id`
- `resource`
- `evidence_type`
- `annotation`
- `pathway`

### Strongly recommended columns

- `ligand_protein_id`
- `receptor_protein_id`
- `ligand_sequence`
- `receptor_sequence`
- `ligand_role`
- `receptor_role`
- `ligand_complex_id`
- `receptor_complex_id`
- `complex_subunit_gene`
- `complex_required_subunits`
- `pmid`
- `source_url`
- `curation_type`
- `directed`
- `confidence_original`
- `license`

### Evidence classes

Use explicit evidence categories:

- `curated_direct`: direct curated LR interaction in that species.
- `curated_inferred`: curated resource but species transfer or weak provenance is involved.
- `orthology_transfer`: computational transfer from another species.
- `embedding_link_prediction`: ESMC + classifier prediction.
- `user_supplied`: user-provided LR pair with unknown curation level.

Only `curated_direct` and high-confidence `curated_inferred` rows should be used as positive labels for the first model. Orthology-transferred and ML-predicted rows should not be mixed into positive labels unless explicitly marked and ablated.

## MVP model: ESMC-300M + two-vector LightGBM

### Protein representation

For every protein sequence, compute one ESMC-300M embedding. Use mean pooling over residues as the default.

```python
emb = pc.embed_proteins_esmc(
    proteins,
    model_name="biohub/esmc-300m-2024-12",
    batch_size=8,
    device="auto",
    pooling="mean",
    cache_dir=".pyccc-cache/esmc",
)
```

Return a table with:

- `gene_id`
- `protein_id`
- `sequence_hash`
- `embedding`
- `model_name`
- `model_revision`
- `pooling`
- `sequence_length`

No model weights or large embedding artifacts should be committed to git.

### Pair feature design

Keep v0 simple and inspectable. The classifier receives two protein vectors plus simple pairwise transformations.

Recommended v0 feature encoder:

```text
z_lig = PCA128(ESMC_mean_embedding_ligand)
z_rec = PCA128(ESMC_mean_embedding_receptor)

X_pair = [
    z_lig,
    z_rec,
    abs(z_lig - z_rec),
    z_lig * z_rec,
    cosine(z_lig, z_rec),
    euclidean(z_lig, z_rec),
    ligand_length_log1p,
    receptor_length_log1p,
    ligand_role_score,
    receptor_role_score,
]
```

LightGBM v0:

```python
model = lightgbm.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=0,
)
```

Then calibrate probabilities on a held-out validation split:

```python
calibrated_model = CalibratedClassifierCV(model, method="isotonic")
```

If calibration is weak cross-species, expose the raw rank score and use density-based top-K selection instead of a hard probability threshold.

### Why not a neural model first

Do not start with a GNN, cross-attention protein-pair transformer, docking model, or end-to-end neural LR predictor. They are harder to validate, harder to install, and more likely to overfit resource artifacts. A two-vector LightGBM model is fast, debuggable, serializable, and good enough to test whether this idea works.

## Role prediction

A DB-free target species still needs a way to reduce the candidate pair space. Do this with role classifiers:

```python
roles = pc.predict_protein_roles(
    proteins=query_proteins,
    embeddings=query_embeddings,
    model="universal_esmc300m_role_v0",
)
```

Return:

- `gene_id`
- `ligand_like_score`
- `receptor_like_score`
- `secreted_like_score`
- `membrane_like_score`
- `ecm_like_score`
- `out_of_domain_score`

Candidate proteins are selected by role scores and expression:

```text
candidate ligands  = expressed genes with ligand_like_score >= ligand_role_min
candidate receptors = expressed genes with receptor_like_score >= receptor_role_min
```

Default caps:

- `max_ligands=3000`
- `max_receptors=3000`
- `max_candidate_pairs=5_000_000` before chunked scoring
- `max_pairs=50_000` after density filtering

The API must allow users to pass their own ligand/receptor candidate lists if they know the biology:

```python
predicted_db = pc.predict_lr_dbfree(
    adata,
    protein_fasta="species.longest_protein.fa",
    ligand_candidates="known_secreted_genes.txt",
    receptor_candidates="known_surface_genes.txt",
)
```

## Candidate generation

Do not score all gene-gene pairs by default.

Default candidate generator:

1. keep only genes expressed in the AnnData object,
2. keep only proteins with valid sequence and ESMC embedding,
3. rank genes by ligand-like and receptor-like role scores,
4. generate ligand-like × receptor-like candidate pairs,
5. optionally add nearest-neighbor pairs in embedding space,
6. score in chunks with the LightGBM model,
7. filter by clade-aware density prior and quality caps.

API:

```python
candidate_pairs = pc.generate_lr_candidates_dbfree(
    adata=adata,
    proteins=query_proteins,
    roles=query_roles,
    gene_id_key="gene_id",
    expression_min_fraction=0.02,
    ligand_role_min=0.30,
    receptor_role_min=0.30,
    max_ligands=3000,
    max_receptors=3000,
    max_candidate_pairs=5_000_000,
)
```

## LR-density prior for thresholding

The user-proposed average LR density idea is good, but use a robust density prior rather than a raw mean.

For each well-annotated training species `s`:

```text
L_s = number of curated ligand-like proteins in species s
R_s = number of curated receptor-like proteins in species s
E_s = number of curated directed LR edges in species s
rho_s = E_s / (L_s * R_s)
```

For a target species:

```text
K_target = round(rho_prior * L_target * R_target)
```

where:

```text
rho_prior = median(rho_s) within the nearest clade if available
          = trimmed median across all well-annotated species otherwise
```

Use `rho_prior` to select the top `K_target` predicted edges by model score, with safety clamps:

```text
K_target = min(K_target, max_pairs)
K_target = max(K_target, min_pairs_if_any_confident)
exclude pairs with model_score < min_score unless allow_low_score_density_fill=True
cap per ligand and per receptor to avoid hub artifacts
```

Recommended defaults:

```python
predicted_db = pc.predict_lr_dbfree(
    adata,
    protein_fasta="species.longest_protein.fa",
    species_hint="unknown",
    density_prior="auto",
    density_stat="median",
    min_score=0.50,
    max_pairs=50000,
    max_pairs_per_ligand=200,
    max_pairs_per_receptor=200,
)
```

The prediction output must include the actual threshold metadata:

- `density_prior`
- `density_source_species`
- `density_source_resources`
- `candidate_ligand_count`
- `candidate_receptor_count`
- `candidate_pair_count`
- `selected_pair_count`
- `score_threshold`
- `min_score`
- `max_pairs`

## Predicted LR table output

`pc.predict_lr_dbfree(...)` should return a normal `CellChatDB`-compatible object.

Required interaction columns:

- `ligand`
- `receptor`
- `pathway`
- `annotation`
- `evidence`
- `evidence_type`
- `confidence`
- `model_score`
- `calibrated_probability`
- `model_name`
- `model_version`
- `model_revision`
- `feature_encoder`
- `density_prior`
- `density_rank`
- `candidate_strategy`
- `ligand_role_score`
- `receptor_role_score`
- `nearest_reference_ligand`
- `nearest_reference_receptor`
- `nearest_reference_lr`
- `nearest_reference_species`
- `nearest_reference_resource`
- `warning`

Default values:

```text
pathway = "DB-free predicted"
annotation = "Predicted LR"
evidence = "ESMC-300M + LightGBM link prediction"
evidence_type = "embedding_link_prediction"
```

Do not invent pathway biology. If the model can assign a nearest known pathway, store it as `nearest_reference_pathway`, not as the primary curated `pathway` unless explicitly requested.

## Public APIs

### 1. Resource normalization

```python
resources = pc.load_training_lr_resources(
    [
        {"path": "cellchat_human.tsv", "schema": "cellchat", "species": "human", "taxon_id": 9606},
        {"path": "flyphonedb2.tsv", "schema": "flyphonedb2", "species": "drosophila", "taxon_id": 7227},
        {"path": "plantcellchat.tsv", "schema": "plantcellchat", "species": "arabidopsis", "taxon_id": 3702},
    ],
    strict=True,
)
```

### 2. Training table construction

```python
train = pc.build_lr_training_table(
    resources,
    protein_fasta_by_species={
        "human": "human.longest_protein.fa",
        "mouse": "mouse.longest_protein.fa",
        "drosophila": "dmel.longest_protein.fa",
        "arabidopsis": "ath.longest_protein.fa",
    },
    positive_evidence={"curated_direct", "curated_inferred"},
    drop_complexes="partial",
)
```

### 3. Embedding

```python
emb = pc.embed_proteins_esmc(
    train.proteins,
    model_name="biohub/esmc-300m-2024-12",
    pooling="mean",
    cache_dir=".pyccc-cache/esmc",
)
```

### 4. Model training

```python
model_card = pc.train_lr_link_predictor(
    training_table=train,
    embeddings=emb,
    model="lightgbm",
    feature_encoder="pca128_absdiff_hadamard_v1",
    validation_splits=["leave_species_out", "leave_resource_out", "leave_family_out"],
    negative_strategy="pu_degree_matched",
    output_dir="models/universal_esmc300m_lgbm_v0",
)
```

### 5. DB-free target-species prediction

```python
predicted_db = pc.predict_lr_dbfree(
    adata,
    cds_fasta="target_species.cds.fa",
    gene_id_key="gene_id",
    species_name="target_species",
    species_hint="plant",
    model="models/universal_esmc300m_lgbm_v0",
    density_prior="auto",
    min_score=0.50,
    max_pairs=50000,
    cache_dir=".pyccc-cache",
)
```

### 6. CCC with predicted LR table

```python
res = pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=predicted_db,
    gene_symbols_key="gene_id",
    score_method="cellchat",
    aggregate="tri_mean",
)
```

### 7. Spatial validation

```python
spatial_report = pc.validate_spatial_lr_table(
    adata,
    lr_table=predicted_db,
    groupby="cell_type",
    spatial_key="spatial",
    mode="cellbin",
    distance_kernels=["contact", "exp"],
    n_permutations=1000,
    null_models=["coordinate_permutation", "celltype_permutation", "matched_random_lr"],
)
```

## Model tracks

### MVP track: `universal_esmc300m_lgbm_v0`

Use one universal LightGBM model trained on normalized resources across species/kingdoms.

Why this first:

- one model is simpler to maintain,
- the density prior can be clade-aware even if the model is universal,
- leave-species-out validation will quickly reveal whether the universal model generalizes.

Required metadata:

- training resources and versions,
- species included,
- number of positives by species,
- number of pseudo-negatives/unlabeled pairs by species,
- density priors by species and clade,
- feature encoder version,
- ESMC model name/revision,
- LightGBM parameters,
- validation metrics.

### Baseline track: `human_mouse_esmc300m_lgbm_v0`

Keep a mammal-focused model as a baseline and for vertebrate use. It should not be the main answer to plant or insect use cases.

### Optional later track: `plant_esmc300m_lgbm_v0`

Add a plant-specific model only if enough curated plant LR labels are available and the universal model underperforms on plant leave-species/resource validation.

## Training labels and negatives

### Positives

Use only high-confidence curated pairs as positives:

- direct species-curated CellChatDB rows,
- direct OmniPath/CellPhoneDB-style LR rows with clear ligand/receptor roles,
- FlyPhoneDB2 rows if schema and provenance are confirmed,
- PlantPhoneDB rows if schema and provenance are confirmed,
- PlantCellChat rows if schema and provenance are confirmed.

Deduplicate positives by:

```text
species + ligand_protein_id + receptor_protein_id + resource_group
```

Keep a `support_count` and `support_resources` column.

### Do not use as positives by default

- orthology-transferred pairs,
- ML-predicted pairs,
- pairs with ambiguous ligand/receptor direction,
- complex pairs with missing required subunits,
- non-protein ligands unless there is a protein-coding ligand surrogate.

### Negative / unlabeled strategy

Do not label all unknown pairs as true negatives.

Use PU-style pseudo-negatives:

1. build ligand-like and receptor-like pools in the same species,
2. remove all known positives and homolog-near positives,
3. sample degree-matched pseudo-negatives,
4. include easy negatives only as a small fraction,
5. repeat sampling across folds and report variance.

Recommended pseudo-negative ratio:

```text
1 positive : 5 pseudo-negative/unlabeled pairs
```

Store sampling metadata:

- `negative_strategy`
- `negative_seed`
- `degree_matching`
- `excluded_homology_radius`
- `positive_resource_blacklist_for_fold`

## Leakage-safe validation

Random pair splits are not enough. They will overestimate performance.

Required validation splits:

1. **leave-resource-out**: train without one resource, test on that resource.
2. **leave-species-out**: train without one species, test on that species.
3. **leave-family-out**: hold out ligand/receptor protein families or homology clusters.
4. **leave-clade-out if possible**: e.g. train on animals, test on plants, or train without insect resources and test on Drosophila.

Report:

- PR-AUC,
- ROC-AUC only as secondary,
- top-K precision at K = 100, 500, 1000, 5000,
- calibration error,
- species-level density error,
- performance compared to orthology-only transfer,
- performance compared to expression-only random LR controls,
- failure cases by ligand/receptor family.

## Spatial validation on Stereo-seq/cellbin

The spatial validation goal is not to prove every predicted LR pair. It should test whether predicted LR-driven CCC scores are spatially enriched compared with matched null models.

### Inputs

- AnnData expression matrix.
- `adata.obsm["spatial"]` cell centroid coordinates.
- Optional cellbin polygons or cell area/radius metadata.
- Cell type labels in `adata.obs[groupby]`.
- Predicted LR table from `pc.predict_lr_dbfree(...)`.

### Distance kernels

Implement two simple kernels first:

```text
contact kernel:
    K(i, j) = 1 if distance(i, j) <= radius else 0

secreted/diffusion kernel:
    K(i, j) = exp(-distance(i, j) / sigma)
```

Default radius/sigma estimation:

- if cellbin cell area is available, estimate cell radius from area,
- otherwise use the median nearest-neighbor distance,
- allow explicit `radius` and `sigma` arguments.

### Validation statistics

For each predicted LR pair and sender/receiver cell-type pair:

```text
expression_score = mean_ligand_expression(sender) * mean_receptor_expression(receiver)
spatial_weight = mean K(sender_cell, receiver_cell)
spatial_ccc_score = expression_score * spatial_weight
```

Then compare top predicted LR edges to nulls.

### Null models

Required nulls:

1. **coordinate permutation**: keep expression and labels, permute spatial coordinates.
2. **cell-type label permutation**: keep coordinates and expression, permute labels within tissue regions if possible.
3. **matched random LR**: random ligand/receptor pairs matched by expression level, role score, and gene degree.
4. **score permutation**: keep candidate pairs, permute model scores.

### Metrics

Report:

- spatial enrichment z-score,
- empirical p-value,
- top-K spatial enrichment for K = 100, 500, 1000,
- distance-decay curve for top predicted pairs,
- reproducibility across sections/replicates,
- contact-kernel enrichment for membrane/contact-like pairs,
- diffusion-kernel enrichment for secreted-like pairs,
- comparison with curated DB when a curated DB exists for the same species.

### Important caveat

Spatial proximity is plausibility evidence, not biochemical validation. Secreted signaling can be long range, contact signaling should be local, and spatial co-localization can be driven by shared cell state or tissue architecture. Always include null models.

## Milestones

### Milestone 0 — Keep OmniPath as recommended install and tutorial path

This is already mostly done but should remain part of the goal.

Required work:

1. README recommends `pyccc[omnipath]`.
2. Installation docs recommend `pyccc[omnipath]`.
3. Tutorial shows:
   - `pc.load_cellchatdb("human")`,
   - `pc.load_cellchatdb("mouse")`,
   - `pc.load_omnipath_interactions(...)`,
   - `pc.filter_lr_table(...)`.
4. Tests mock OmniPath; CI must not depend on live network calls.

Acceptance criteria:

```bash
uv run pytest -q
uv run sphinx-build -W -b html docs docs/_build/html
uv build
```

### Milestone 1 — LR resource normalization for training

Files to add:

- `src/pyccc/lr_resources.py`
- `src/pyccc/training_data.py`
- `scripts/normalize_lr_resource.py`
- `tests/test_lr_resource_normalization.py`
- `docs/lr_resource_training.md`

Required work:

1. Add `load_training_lr_resources(...)`.
2. Add schema adapters for `cellchat`, `omnipath`, `cellphonedb`, `flyphonedb2`, `plantphonedb`, `plantcellchat`, and `generic`.
3. Normalize all resources to one schema.
4. Preserve provenance and license fields.
5. Deduplicate interactions while retaining support counts.
6. Validate ligand/receptor direction.
7. Handle complexes conservatively.

Acceptance criteria:

- local toy fixtures for every schema pass,
- no network required,
- unsupported columns fail with clear errors under `strict=True`,
- normalized output can be used to build a CellChatDB-compatible LR table.

### Milestone 2 — CDS/protein preparation

Files to add:

- `src/pyccc/sequence.py`
- `tests/test_sequence.py`
- `examples/non_model_species_cds_demo.py`

Required APIs:

```python
proteins = pc.load_cds_translations(
    "sample.cds.fa",
    gene_id_regex=r"gene=([^\s]+)",
    transcript_id_regex=r"transcript=([^\s]+)",
    genetic_code=1,
    select="longest",
)
```

```python
proteins = pc.load_protein_fasta(
    "sample.longest_protein.fa",
    gene_id_regex=r"gene=([^\s]+)",
)
```

Required output columns:

- `gene_id`
- `transcript_id`
- `protein_id`
- `cds_sequence`
- `protein_sequence`
- `cds_length`
- `protein_length`
- `stop_codon_count`
- `valid_translation`
- `selected_isoform`

Acceptance criteria:

- longest isoform selection works,
- invalid CDS records are flagged,
- expression gene IDs are matched without mutating `adata.var_names`,
- unmatched expression/CDS genes are reported.

### Milestone 3 — Embedding cache and feature encoder

Files to add:

- `src/pyccc/embeddings.py`
- `src/pyccc/pair_features.py`
- `tests/test_pair_features.py`
- `scripts/embed_proteome_esmc.py`

Required APIs:

```python
emb = pc.embed_proteins_esmc(
    proteins,
    model_name="biohub/esmc-300m-2024-12",
    pooling="mean",
    cache_dir=".pyccc-cache/esmc",
)
```

```python
X = pc.make_lr_pair_features(
    pairs=candidate_pairs,
    embeddings=emb,
    encoder="pca128_absdiff_hadamard_v1",
)
```

Acceptance criteria:

- tests use tiny fake embeddings,
- no Hugging Face download in default CI,
- embedding cache keys include sequence hash, model name, model revision,
  pooling, and backend,
- pair features are deterministic.

### Milestone 4 — Role classifier

Files to add:

- `src/pyccc/roles.py`
- `scripts/train_role_classifier.py`
- `tests/test_roles.py`
- `docs/predictor_model_card.md`

Required APIs:

```python
role_model = pc.train_protein_role_classifier(
    training_table=train,
    embeddings=emb,
    roles=["ligand_like", "receptor_like", "secreted_like", "membrane_like"],
    output_dir="models/universal_esmc300m_role_v0",
)
```

```python
roles = pc.predict_protein_roles(
    proteins=query_proteins,
    embeddings=query_embeddings,
    model="models/universal_esmc300m_role_v0",
)
```

Acceptance criteria:

- role prediction works from fixture embeddings,
- user can bypass role classifier with explicit ligand/receptor candidate lists,
- candidate generation is capped and chunked.

### Milestone 5 — LightGBM link predictor

Files to add:

- `src/pyccc/lr_prediction.py`
- `scripts/train_lr_predictor.py`
- `tests/test_lr_prediction.py`
- `docs/predictor_model_card.md`

Required APIs:

```python
model_card = pc.train_lr_link_predictor(
    training_table=train,
    embeddings=emb,
    model="lightgbm",
    feature_encoder="pca128_absdiff_hadamard_v1",
    negative_strategy="pu_degree_matched",
    validation_splits=["leave_species_out", "leave_resource_out", "leave_family_out"],
    output_dir="models/universal_esmc300m_lgbm_v0",
)
```

```python
scores = pc.score_lr_candidates(
    candidate_pairs=candidate_pairs,
    embeddings=query_embeddings,
    model="models/universal_esmc300m_lgbm_v0",
    batch_size=200000,
)
```

Acceptance criteria:

- the model can train on a tiny fixture dataset,
- saved model can reload and predict deterministically,
- validation report is written as JSON/Markdown,
- model card is generated automatically,
- no model weights are committed to git.

### Milestone 6 — Density calibration and predicted LR table builder

Files to add:

- `src/pyccc/density.py`
- `tests/test_density.py`

Required APIs:

```python
density = pc.estimate_lr_density_prior(
    training_table=train,
    groupby="clade",
    stat="median",
    min_species_edges=100,
)
```

```python
predicted_db = pc.build_predicted_lr_table(
    scored_pairs=scores,
    roles=query_roles,
    density_prior=density,
    species_hint="plant",
    min_score=0.50,
    max_pairs=50000,
)
```

Acceptance criteria:

- density is computed per species and clade,
- top-K thresholding is deterministic,
- output contains density metadata,
- hub ligands/receptors are capped,
- output is CellChatDB-compatible.

### Milestone 7 — End-to-end DB-free target-species workflow

Files to add:

- `examples/dbfree_non_model_species_demo.py`
- `docs/dbfree_ccc.md`
- `tests/test_dbfree_workflow.py`

Required API:

```python
predicted_db = pc.predict_lr_dbfree(
    adata,
    cds_fasta="target.cds.fa",
    gene_id_key="gene_id",
    species_name="target_species",
    species_hint="unknown",
    model="models/universal_esmc300m_lgbm_v0",
    density_prior="auto",
    max_pairs=50000,
)
```

Acceptance criteria:

- a toy AnnData + toy CDS fixture runs end-to-end,
- the predicted DB can be passed directly into `compute_communication`,
- outputs include prediction summary and warnings,
- docs clearly label the method as experimental.

### Milestone 8 — Spatial validation for Stereo-seq/cellbin

Files to add:

- `src/pyccc/spatial_validation.py`
- `examples/stereoseq_cellbin_spatial_validation.py`
- `docs/spatial_validation.md`
- `tests/test_spatial_validation.py`

Required API:

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

Required output:

- `summary` DataFrame by LR pair,
- `celltype_pair_summary` DataFrame,
- `null_distribution` metadata,
- `distance_decay` DataFrame,
- plots if plotting dependencies are installed.

Acceptance criteria:

- works with `adata.obsm["spatial"]`,
- does not require Stereo-seq-specific dependencies,
- supports cellbin radius/area metadata when available,
- implements coordinate, cell-type, matched-random-LR, and score-permutation nulls,
- reports empirical p-values and z-scores.

## Documentation plan

### `docs/lr_resources.md`

Include:

1. recommended install with `pyccc[omnipath]`,
2. CellChatDB human/mouse examples,
3. OmniPath examples,
4. resource filtering/provenance,
5. loading external LR tables,
6. how this relates to DB-free prediction.

### `docs/dbfree_ccc.md`

Include:

1. what “DB-free target-species CCC” means,
2. required inputs,
3. CDS/protein preparation,
4. ESMC embedding cache,
5. role prediction,
6. LightGBM LR link prediction,
7. density thresholding,
8. running `compute_communication`,
9. interpreting candidate LR tables,
10. limitations.

### `docs/predictor_model_card.md`

Include:

1. model name/version,
2. ESMC model name and revision,
3. training resources and dates,
4. species included,
5. positive label rules,
6. pseudo-negative rules,
7. feature encoder,
8. density prior computation,
9. validation splits,
10. metrics,
11. calibration,
12. intended use,
13. out-of-scope use,
14. known failure modes.

### `docs/spatial_validation.md`

Include:

1. spatial coordinates required in AnnData,
2. Stereo-seq/cellbin expectations,
3. distance kernels,
4. null models,
5. metrics,
6. interpretation caveats,
7. example plots.

## Testing requirements

Default CI must not require:

- internet,
- OmniPath live service,
- Hugging Face,
- GPU,
- ESMC weights,
- LightGBM unless `predict` tests are enabled,
- large reference resources.

Use markers:

```bash
uv run pytest -q -m "not slow and not network and not gpu"
uv run pytest -q -m predict
uv run pytest -q -m spatial
uv run pytest -q -m network
uv run pytest -q -m gpu
```

Required fixture tests:

- schema normalization for every resource adapter,
- CDS translation and longest isoform selection,
- fake ESMC embedding cache,
- pair feature construction,
- role candidate generation,
- LightGBM training on tiny fake data,
- density thresholding,
- predicted LR table compatibility with `compute_communication`,
- spatial validation null models on toy coordinates.

## Model quality gates

Before advertising the DB-free predictor as useful beyond demos, require:

1. leave-species-out PR-AUC better than expression/role-only baseline,
2. top-K precision better than density-matched random LR baseline,
3. calibration or rank-thresholding behavior reported honestly,
4. density of predicted edges close to the clade prior,
5. spatial enrichment above matched random LR controls on at least one spatial dataset,
6. reproducibility across at least two sections or samples when available,
7. clear failure mode documentation.

## Non-goals

- Do not make `torch`, `transformers`, or `lightgbm` default dependencies.
- Do not call the method experimentally validated LR discovery.
- Do not infer LR pairs silently inside `compute_communication`.
- Do not score all gene-gene pairs by default.
- Do not commit ESMC weights, downloaded databases, or large embeddings.
- Do not assume FlyPhoneDB2, PlantPhoneDB, or PlantCellChat have stable public APIs until confirmed.
- Do not invent curated pathway labels for predicted LR pairs.
- Do not treat spatial co-localization as direct biochemical validation.
- Do not require users to rename `adata.var_names`; preserve explicit gene ID mapping.

## Repository-level acceptance criteria

The goal is complete when:

1. `pyccc[omnipath]` remains the recommended install in README/docs.
2. LR resource loading has a stable tutorial and provenance story.
3. Resource normalization supports CellChatDB, OmniPath/CellPhoneDB-style tables, FlyPhoneDB2, PlantPhoneDB, PlantCellChat, and generic LR tables through local fixtures.
4. A training script can build a normalized cross-species LR training table.
5. ESMC embeddings can be cached from CDS/protein FASTA through an optional `predict` extra.
6. A two-vector LightGBM link predictor can be trained and serialized.
7. DB-free target-species prediction returns a CellChatDB-compatible LR table.
8. Predicted LR tables include model score, density threshold, confidence, provenance, and warnings.
9. The predicted LR table runs through `compute_communication` without special cases.
10. Stereo-seq/cellbin-style spatial validation reports enrichment against null models.
11. Default tests pass without network/GPU/model downloads.
12. Sphinx builds with warnings as errors.
13. `uv build` passes.

Run before declaring done:

```bash
uv run pytest -q
uv run sphinx-build -W -b html docs docs/_build/html
uv build
```

## Suggested implementation order

The simplest feasible order is:

1. keep OmniPath tutorial/documentation complete,
2. implement resource normalization with local fixtures,
3. implement CDS/protein loading,
4. implement fake-embedding tests and pair feature builder,
5. implement LightGBM training and prediction using tiny fixtures,
6. implement density prior thresholding,
7. implement `predict_lr_dbfree(...)` wrapper,
8. connect predicted LR table to `compute_communication`,
9. implement spatial validation,
10. only then train real models outside CI.

## References for design rationale

- ESM Cambrian / ESMC official announcement: https://www.evolutionaryscale.ai/blog/esm-cambrian
- Hugging Face ESMC-300M model page: https://huggingface.co/biohub/esmc-300m-2024-12
- OmniPath official overview: https://omnipathdb.org/
- CellPhoneDB documentation: https://cellphonedb.readthedocs.io/en/stable/
- LightGBM LGBMClassifier documentation: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
