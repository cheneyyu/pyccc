# pyccc Protein Role Classifier

- classifier: `lightgbm`
- training proteins: 17167
- validation method: stratified_holdout
- embedding model: biohub/esmc-300m-2024-12
- embedding backend: esmc
- embedding pooling: mean

## Role Metrics

| role | positives | negatives | PR-AUC | baseline PR-AUC | delta | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ligand_like | 3825 | 13342 | 0.6295 | 0.2227 | 0.4068 | ok |
| receptor_like | 1809 | 15358 | 0.7871 | 0.1053 | 0.6818 | ok |
| secreted_like | 3825 | 13342 | 0.6295 | 0.2227 | 0.4068 | ok |
| membrane_like | 1809 | 15358 | 0.7871 | 0.1053 | 0.6818 | ok |

Predicted roles are sequence-derived candidate annotations used to reduce the LR search space.
They should not be interpreted as experimentally validated secretion or membrane-localization evidence.
