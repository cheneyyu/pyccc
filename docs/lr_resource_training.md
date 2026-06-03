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
uv run python scripts/embed_proteome_esmc.py \
  --protein-fasta human.longest_protein.fa \
  --output human.embeddings.tsv \
  --backend hash

uv run python scripts/train_role_classifier.py \
  --training-lr normalized_lr.tsv \
  --embeddings human.embeddings.tsv \
  --output-dir models/universal_esmc300m_role_v0

uv run --extra predict python scripts/train_lr_predictor.py \
  --training-lr normalized_lr.tsv \
  --embeddings human.embeddings.tsv \
  --output-dir models/universal_esmc300m_lgbm_v0 \
  --model lightgbm
```

Use `--backend hash` only for fixture tests or dry runs. Real model building
should use the ESMC backend from the `predict` extra.

The LR predictor trainer now writes `model_card.json` and `model_card.md` with:

- a random stratified split,
- leave-species-out folds when multiple species are present,
- leave-resource-out folds when multiple resources are present,
- leave-family-out folds when ligand/receptor family labels are present,
- leave-clade-out folds when a `clade` column is present and requested,
- PR-AUC, ROC-AUC, and top-K precision at K = 100, 500, 1000, and 5000.
- held-out probability calibration using isotonic regression by default,
- a final deployment model refit on all training pairs after validation.

Folds that cannot contain both positive and pseudo-negative labels are reported
as skipped instead of silently inflating the validation result. The PCA feature
encoder is fit inside each training fold for validation, then refit on all
pairs only for the final serialized model.
