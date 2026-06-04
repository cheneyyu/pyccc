# pyccc LR Link Predictor Model Card

- Model type: `lightgbm`
- Feature encoder: `pca128_absdiff_hadamard_v1`
- Final model training: `all_pairs_after_validation`
- Validation feature encoder fit: `train_split_only`
- Calibration method: `isotonic`
- Negative strategy: `pu_degree_matched`
- Negative ratio: `3`
- Easy negative fraction: `0.05`
- Excluded homology radius: `family_pair`
- Negative repeats: `1`
- Validation splits requested: leave_species_out, leave_resource_out, leave_family_out
- Random split PR-AUC: 0.6981
- Species included: arabidopsis, human, maize, mouse, rice, tomato
- Training resources: CellChatDB, PlantCellChatDB

## Validation Summary

- `random_stratified`: PR-AUC 0.6981, strongest baseline `embedding_cosine` 0.3130, top-100 enrichment 4.00x, top-100 recall 0.0338
- `leave_species_out`: 6 usable folds, mean PR-AUC 0.5741, strongest baseline `embedding_cosine` 0.3425, top-100 enrichment 3.57x, top-100 recall 0.0652
- `leave_resource_out`: 2 usable folds, mean PR-AUC 0.3027, strongest baseline `embedding_cosine` 0.3365, top-100 enrichment 1.98x, top-100 recall 0.0140
- `leave_family_out`: skipped (No non-empty family labels are available.)

## Negative Sampling

- Positives: 11845
- Pseudo-negatives: 35535
- Easy pseudo-negatives: 1780
- Degree matching: True
- Homology exclusion: `family_pair`
- Positive resource blacklist for folds: `CellChatDB;PlantCellChatDB`

## Model Stack

- Embedding model: biohub/esmc-300m-2024-12
- Embedding revision: not recorded
- Embedding backend: esmc
- Embedding pooling: mean
- Embeddings: 17167
- Pair model parameters: `{"class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.03, "n_estimators": 300, "num_leaves": 31, "objective": "binary", "random_state": 0, "subsample": 0.8, "verbose": -1}`

## Calibration

- Brier score: 0.1246
- ECE 10-bin: 0.0000

## Intended Use

This model predicts candidate ligand-receptor pairs from protein embeddings for downstream pyccc scoring.

## Caveat

Predicted pairs are computational candidates and are not biochemical validation.