# Figure 5. DB-free LR prediction model

DB-free prediction converts target-species protein sequences into
provenance-rich LR candidate tables using ESMC-300M embeddings, LightGBM
ranking, and density-controlled edge selection.

Panels A-F show training resources, model architecture, pair-ranker validation,
density prior calibration, example predicted LR rows, and protein-role
classifier validation.

Data source: `docs/paper/source_data/figure5_*.tsv` plus model-card JSON files
under `models/`. Predicted LR rows are computational candidates and should not
be interpreted as experimentally validated biochemical interactions.
