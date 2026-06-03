# LR Resource Training Tables

`load_training_lr_resources` handles schema normalization. `build_lr_training_table`
adds protein sequences and labels for predictor training.

```python
resources = pc.load_training_lr_resources([
    {"path": "cellchat_human.tsv", "schema": "cellchat", "species": "human", "taxon_id": 9606},
    {"path": "cellchat_mouse.tsv", "schema": "cellchat", "species": "mouse", "taxon_id": 10090},
])

train = pc.build_lr_training_table(
    resources,
    protein_fasta_by_species={
        "human": "human.longest_protein.fa",
        "mouse": "mouse.longest_protein.fa",
    },
    positive_evidence={"curated_direct", "curated_inferred"},
    drop_complexes="partial",
)
```

The first implementation is conservative:

- direct curated and high-confidence inferred rows can be positive labels,
- orthology-transferred and ML-predicted rows remain explicitly marked and are
  not positives by default,
- complex LR rows without matching sequence evidence are dropped when
  `drop_complexes="partial"`,
- no downloaded databases, model weights, or embedding artifacts are committed.

Training scripts accept normalized TSV files:

```bash
uv run python scripts/normalize_lr_resource.py \
  --path cellchat_human.tsv \
  --schema cellchat \
  --species human \
  --taxon-id 9606 \
  --output normalized_cellchat_human.tsv

uv run python scripts/build_lr_training_table.py \
  --normalized-lr normalized_cellchat_human.tsv \
  --protein-fasta human=human.longest_protein.fa \
  --output-dir data/lr_training

uv run --extra predict python scripts/embed_proteome_esmc.py \
  --protein-table data/lr_training/training_proteins.tsv \
  --output data/lr_training/training_embeddings.tsv \
  --backend esmc \
  --model-name biohub/esmc-300m-2024-12 \
  --cache-dir .pyccc-cache/esmc

uv run --extra predict python scripts/train_role_classifier.py \
  --training-lr data/lr_training/training_interactions.tsv \
  --embeddings data/lr_training/training_embeddings.tsv \
  --output-dir models/universal_esmc300m_role_v0 \
  --model lightgbm \
  --validation-fraction 0.25

uv run --extra predict python scripts/train_lr_predictor.py \
  --training-lr data/lr_training/training_interactions.tsv \
  --embeddings data/lr_training/training_embeddings.tsv \
  --output-dir models/universal_esmc300m_lgbm_v0 \
  --model lightgbm \
  --density-groupby clade \
  --negative-ratio 5 \
  --easy-negative-fraction 0.05 \
  --excluded-homology-radius family_pair \
  --negative-repeats 3
```

Use `--backend hash` only for fixture tests or dry runs. Real model building
should use the ESMC-300M backend from the `predict` extra.

The role classifier trainer writes `model_card.json` and `model_card.md` with
one-vs-rest positive/negative counts, prevalence, stratified holdout PR-AUC,
ROC-AUC, top-K precision, and prevalence baseline deltas when each role has
enough positive and negative proteins. Small or single-class roles are reported
as skipped with a reason instead of inventing validation metrics.

The LR predictor trainer now writes `model_card.json` and `model_card.md` with:

- a random stratified split,
- leave-species-out folds when multiple species are present,
- leave-resource-out folds when multiple resources are present,
- leave-family-out folds when ligand/receptor family labels are present,
- leave-clade-out folds when a `clade` column is present and requested,
- PR-AUC, ROC-AUC, top-K precision, top-K recall, and top-K enrichment at
  K = 100, 500, 1000, and 5000.
- held-out probability calibration using isotonic regression by default,
- a final deployment model refit on all training pairs after validation.
- baseline comparisons against degree prior, embedding cosine, family-pair
  transfer, expression-only, role-only, density-matched random, and random
  scores for PR-AUC and top-K ranker metrics.
- ligand/receptor-family failure case summaries for held-out folds when
  family or homology-cluster labels are present.
- model-stack metadata: training resources, species/clades included, positive
  and pseudo-negative counts by species, embedding model name/revision/backend/
  pooling, and the serialized pair-model parameters.
- PU pseudo-negative sampling metadata, including positives/negatives by
  species, degree matching, the easy-negative count,
  `positive_resource_blacklist_for_fold`, and family-pair homology exclusion
  when ligand/receptor family or homology-cluster labels are present.
- repeated PU negative-sampling validation summaries, including PR-AUC/ROC-AUC
  mean, standard deviation, and variance by split when `--negative-repeats` is
  greater than one.
- `density_prior.tsv`, computed from curated positive LR rows by `clade` by
  default, for target-species thresholding.

Folds that cannot contain both positive and pseudo-negative labels are reported
as skipped instead of silently inflating the validation result. The PCA feature
encoder is fit inside each training fold for validation, then refit on all
pairs only for the final serialized model.

`--excluded-homology-radius family_pair` is a conservative v0 approximation of
homolog-near exclusion: when the training table has `ligand_family` and
`receptor_family` labels, pseudo-negatives with the same ligand-family plus
receptor-family combination as a curated positive are not sampled. If family
labels are absent, the model card still reports that the requested exclusion
mode was configured, but no sequence-radius inference is invented.

After training, inspect quality gates programmatically:

```python
gates = pc.evaluate_lr_model_quality_gates(
    "models/universal_esmc300m_lgbm_v0/model_card.json",
    required_splits=["leave_species_out", "leave_resource_out"],
    min_pr_auc_delta=0.0,
    top_k="top_100",
    min_top_k_delta=0.0,
)
print(gates)
print(gates.attrs["passed"])
```

For a generated predicted LR table, check density calibration separately:

```python
density_gates = pc.evaluate_predicted_lr_density_prior(predicted_db, max_fold_error=2.0)
print(density_gates.attrs["passed"])
```
