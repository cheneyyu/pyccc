# Figure 1. pyccc overview

pyccc unifies CellChat-like CCC inference, differential analysis, visualization,
LR resource loading, and optional DB-free LR prediction in an AnnData-native
Python workflow.

Panel A shows the input problem setting and the pyccc solution surface. Panel B
shows the regular curated-LR workflow. Panel C shows CellChatDB, OmniPath,
external files, user tables, and DB-free predicted LR tables converging into a
CellChatDB-compatible LR table. Panel D shows the optional target-species
DB-free prediction branch. Panel E shows the minimal API surface.

Data source: API inventory in `docs/paper/source_data/figure1_api_inventory.tsv`.
The DB-free branch is an optional experimental LR-table generator, not the
default regular-mode benchmark path.
