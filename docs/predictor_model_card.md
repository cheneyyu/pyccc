# Predictor Model Card Template

Each trained LR predictor should write `model_card.json` and `model_card.md`.
The model card must include:

- model name, version, and revision,
- ESMC model name and revision,
- training resources and dates,
- species and clades included,
- positive label rules,
- pseudo-negative sampling rules,
- feature encoder,
- density prior computation,
- validation splits,
- PR-AUC and top-K precision,
- calibration method or rank-thresholding caveat,
- intended use,
- out-of-scope use,
- known failure modes.

The scaffolded trainer writes a minimal card:

```bash
uv run --extra predict python scripts/train_role_classifier.py \
  --training-lr data/lr_training/training_interactions.tsv \
  --embeddings data/lr_training/training_embeddings.tsv \
  --output-dir models/universal_esmc300m_role_v0 \
  --model lightgbm \
  --validation-fraction 0.25

uv run --extra predict python scripts/train_lr_predictor.py \
  --training-lr normalized_lr.tsv \
  --embeddings embeddings.tsv \
  --output-dir models/universal_esmc300m_lgbm_v0 \
  --model lightgbm \
  --density-groupby clade \
  --negative-ratio 5 \
  --easy-negative-fraction 0.05 \
  --excluded-homology-radius family_pair \
  --negative-repeats 3
```

Before advertising a DB-free predictor beyond demos, require leave-species-out
performance above expression/role-only baselines, top-K precision above
density-matched random controls, and honest calibration or rank-threshold
reporting.

Current generated cards include a machine-readable `validation_report` with a
random stratified split plus leakage-aware leave-species, leave-resource, and
leave-family folds when those metadata are available. Leave-clade folds are
also reported when a `clade` column is present and the split is requested.
Unusable folds are kept in the report with a skip reason.

Role classifier cards include one-vs-rest counts, role prevalence, stratified
holdout PR-AUC/ROC-AUC/top-K precision when enough labels are available, and a
prevalence PR-AUC baseline. Small or single-class roles are explicitly marked
as skipped.

Generated cards also record:

- `final_model_training = all_pairs_after_validation`,
- `validation_feature_encoder_fit = train_split_only`,
- training resources, species/clades included, positive-label and
  pseudo-negative counts by species,
- ESMC embedding model name/revision/pooling and final pair-model parameters,
- calibration method and calibration metrics,
- Brier score and 10-bin expected calibration error when calibration is usable.
- baseline PR-AUC comparisons for degree prior, embedding cosine,
  family-pair transfer, role-only, and random scores.
- PU pseudo-negative sampling counts by species, degree-matching metadata,
  easy-negative counts, and the configured homolog-near exclusion rule. The v0
  homology exclusion uses ligand/receptor family or homology-cluster labels
  when available; it does not infer new sequence homology clusters.
- repeated PU negative-sampling validation summaries with PR-AUC/ROC-AUC
  mean, standard deviation, and variance by split when `negative_repeats > 1`.
- `density_prior.tsv` next to the serialized LightGBM pair ranker, so
  DB-free prediction can apply a clade-aware density prior with
  `density_prior="auto"`.
- a capped reference index of curated positive LR pairs, used only to annotate
  predicted pairs with nearest-reference provenance.

Use `pc.evaluate_lr_model_quality_gates(...)` to turn the model card into a
check table. It reports model PR-AUC, baseline PR-AUC, optional top-K precision
deltas, pass/fail, and the reason for every requested split/baseline pair.
After building a predicted LR table, use
`pc.evaluate_predicted_lr_density_prior(...)` to check whether the selected edge
density remains close to the clade-aware prior recorded in the prediction
summary.

`family_pair_transfer` is a conservative orthology-transfer proxy: if
`ligand_family` and `receptor_family` or homology-cluster labels are present,
it scores a held-out pair when that ligand-family plus receptor-family
combination appears as a positive in the training split. It does not infer new
orthology mappings.
