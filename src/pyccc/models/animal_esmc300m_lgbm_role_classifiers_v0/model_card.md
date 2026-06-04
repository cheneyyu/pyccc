# pyccc Protein Role Classifier

- classifier: `lightgbm`
- training proteins: 5895
- validation method: stratified_holdout
- embedding model: biohub/esmc-300m-2024-12
- embedding backend: esmc
- embedding pooling: mean

## Role Metrics

| role | positives | negatives | PR-AUC | baseline PR-AUC | delta | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ligand_like | 1031 | 4864 | 0.8481 | 0.1750 | 0.6731 | ok |
| receptor_like | 926 | 4969 | 0.9006 | 0.1574 | 0.7432 | ok |
| secreted_like | 1031 | 4864 | 0.8481 | 0.1750 | 0.6731 | ok |
| membrane_like | 926 | 4969 | 0.9006 | 0.1574 | 0.7432 | ok |

Predicted roles are sequence-derived candidate annotations used to reduce the LR search space.
They should not be interpreted as experimentally validated secretion or membrane-localization evidence.
