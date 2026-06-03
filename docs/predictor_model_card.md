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
uv run --extra predict python scripts/train_lr_predictor.py \
  --training-lr normalized_lr.tsv \
  --embeddings embeddings.tsv \
  --output-dir models/universal_esmc300m_lgbm_v0 \
  --model lightgbm \
  --density-groupby clade
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

Generated cards also record:

- `final_model_training = all_pairs_after_validation`,
- `validation_feature_encoder_fit = train_split_only`,
- calibration method and calibration metrics,
- Brier score and 10-bin expected calibration error when calibration is usable.
- baseline PR-AUC comparisons for degree prior, embedding cosine, role-only,
  and random scores.
- `density_prior.tsv` next to the serialized LightGBM pair ranker, so
  DB-free prediction can apply a clade-aware density prior with
  `density_prior="auto"`.

Use `pc.evaluate_lr_model_quality_gates(...)` to turn the model card into a
check table. It reports model PR-AUC, baseline PR-AUC, optional top-K precision
deltas, pass/fail, and the reason for every requested split/baseline pair.
After building a predicted LR table, use
`pc.evaluate_predicted_lr_density_prior(...)` to check whether the selected edge
density remains close to the clade-aware prior recorded in the prediction
summary.
