# pyccc LR Link Predictor Model Card

- Model type: `lightgbm`
- Feature encoder: `pca128_absdiff_hadamard_v1`
- Final model training: `all_pairs_after_validation`
- Validation feature encoder fit: `train_split_only`
- Calibration method: `isotonic`
- Negative strategy: `pu_degree_matched`
- Negative ratio: `5`
- Easy negative fraction: `0.05`
- Excluded homology radius: `family_pair`
- Negative repeats: `2`
- Validation splits requested: leave_species_out, leave_resource_out, leave_family_out
- Random split PR-AUC: 0.8613
- Species included: human, mouse
- Training resources: CellChatDB

## Validation Summary

- `random_stratified`: PR-AUC 0.8613, strongest baseline `embedding_cosine` 0.2569, top-100 enrichment 5.76x, top-100 recall 0.1455
- `leave_species_out`: 2 usable folds, mean PR-AUC 0.8187, strongest baseline `embedding_cosine` 0.2542, top-100 enrichment 6.00x, top-100 recall 0.0759
- `leave_resource_out`: skipped (no usable folds)
- `leave_family_out`: skipped (No non-empty family labels are available.)

## Negative Sampling

- Positives: 2639
- Pseudo-negatives: 13195
- Easy pseudo-negatives: 661
- Degree matching: True
- Homology exclusion: `family_pair`
- Positive resource blacklist for folds: `CellChatDB`

## Negative Sampling Repeat Variance

- `random_stratified`: 2 usable repeats, PR-AUC 0.8660 +/- 0.0066
- `leave_species_out`: 2 usable repeats, PR-AUC 0.8144 +/- 0.0061

## Model Stack

- Embedding model: biohub/esmc-300m-2024-12
- Embedding revision: not recorded
- Embedding backend: esmc
- Embedding pooling: mean
- Embeddings: 5895
- Pair model parameters: `{"class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.03, "n_estimators": 300, "num_leaves": 31, "objective": "binary", "random_state": 0, "subsample": 0.8, "verbose": -1}`

## Calibration

- Brier score: 0.0490
- ECE 10-bin: 0.0000

## Intended Use

This model predicts candidate ligand-receptor pairs from protein embeddings for downstream pyccc scoring.

## Caveat

Predicted pairs are computational candidates and are not biochemical validation.