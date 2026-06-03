# CellChat Visual Coverage

This checklist tracks pyccc's CellChat-style visual surface. It is intentionally
artifact-focused: each row names the CellChat-style task, the pyccc API that
covers it, and the remaining gap.

## Tutorial-Level Coverage

The official CellChat vignette uses the following major visual families:
aggregate circle networks, hierarchy plots, chord diagrams, pathway heatmaps,
LR contribution bars, LR bubble plots, signaling-role network/scatter/heatmaps,
pattern-number diagnostics, pattern river/dot plots, and network embedding
plots. pyccc now covers the core static CCC and pattern-interpretation figures
with Python-native Matplotlib/plotnine APIs. Remaining non-equivalent areas are
exact R object/layout parity, exact Seurat stacked-violin layout parity, and
exact `circlize` chord geometry.

Reference tutorials:

- CellChat single-dataset vignette:
  <https://github.com/sqjin/CellChat/blob/master/tutorial/CellChat-vignette.Rmd>
- Current CellChat tutorial index:
  <https://github.com/jinworks/CellChat>
- CellChat spatial transcriptomics vignette:
  <https://github.com/jinworks/CellChat/blob/main/tutorial/CellChat_analysis_of_spatial_transcriptomics_data.Rmd>

| CellChat-style task | pyccc API | Status | Remaining gap |
| --- | --- | --- | --- |
| Global weighted circle network | `pyccc.plotting.net_circle` | Covered | Not a byte-for-byte `netVisual_circle` port |
| Source-target network heatmap | `pyccc.plotting.net_heatmap` | Covered | Supports optional row/column clustering |
| Chord-like source-target network | `pyccc.plotting.net_chord` | Covered | Uses Matplotlib arcs, not exact circlize geometry |
| LR/pathway-mediated chord diagram | `pyccc.plotting.net_chord_gene` | Covered | Python-native approximation of `netVisual_chord_gene`; not exact circlize sector geometry |
| Individual LR pair hierarchy/circle/chord | `pyccc.plotting.net_individual` | Covered | Python-native equivalent of `netVisual_individual`; supports hierarchy, circle, and chord layouts for one selected LR pair |
| Sender-receiver hierarchy | `pyccc.plotting.net_hierarchy` | Covered | Needs richer receiver layout options |
| Ligand-receptor bubble | `pyccc.plotting.bubble`, `pyccc.ggplot.bubble` | Covered | Supports `max_pvalue`; plotnine version supports `facet_by` for pathway/annotation splits |
| Differential ligand-receptor bubble | `pyccc.plotting.diff_bubble`, `pyccc.ggplot.diff_bubble` | Covered | Supports `max_pvalue`; plotnine version supports `facet_by` |
| Pathway source-target heatmap | `pyccc.plotting.pathway_heatmap`, `pyccc.ggplot.pathway_heatmap` | Covered | Supports optional clustering; annotation tracks are limited |
| LR contribution within pathway | `pyccc.plotting.lr_contribution`, `pyccc.ggplot.lr_contribution`, `pyccc.plotting.lr_contribution_multi`, `pyccc.ggplot.lr_contribution_multi` | Covered | Single-pathway and multi-pathway contribution panels |
| CellChatDB annotation composition | `pyccc.plotting.annotation_bar`, `pyccc.ggplot.annotation_bar` | Covered | Summarizes annotation classes from CellChatDB |
| Outgoing/incoming role scatter | `pyccc.plotting.signaling_role_scatter`, `pyccc.ggplot.signaling_role_scatter` | Covered | Role definitions are pyccc's network summaries |
| Signaling role heatmap | `pyccc.plotting.signaling_role_heatmap` | Covered | Supports optional row/column clustering |
| Sender/receiver/mediator/influencer roles | `pyccc.plotting.signaling_role_network` | Covered | Exact CellChat centrality formulas may differ |
| Ranked signaling information flow | `pyccc.plotting.rank_signaling`, `pyccc.ggplot.rank_signaling` | Covered | Single-result ranked information flow |
| Ranked condition comparison | `pyccc.plotting.rank_signaling_compare`, `pyccc.ggplot.rank_signaling_compare` | Covered | Covers CellChat `rankNet(..., stacked = TRUE/FALSE)` style via `stacked` |
| Cell-group signaling changes scatter | `pyccc.signaling_changes`, `pyccc.plotting.signaling_changes_scatter`, `pyccc.ggplot.signaling_changes_scatter` | Covered | Python-native equivalent of `netAnalysis_signalingChanges_scatter` for pairwise comparisons |
| Side-by-side role heatmap comparison | `pyccc.plotting.signaling_role_heatmap_compare`, `pyccc.ggplot.signaling_role_heatmap_compare` | Covered | Python-native equivalent of the CellChat comparison tutorial's outgoing/incoming/all `netAnalysis_signalingRole_heatmap` side-by-side panels |
| Pairwise pathway joint embedding | `pyccc.pairwise_pathway_embedding`, `pyccc.plotting.pathway_embedding_pairwise`, `pyccc.ggplot.pathway_embedding_pairwise` | Covered | Python-native equivalent of `netVisual_embeddingPairwise`; exact CellChat merged object slots and zoom-in layout are not reproduced |
| Pathway joint-manifold distance ranking | `pyccc.pairwise_pathway_embedding`, `pyccc.rank_pathway_similarity`, `pyccc.plotting.pathway_similarity_rank`, `pyccc.ggplot.pathway_similarity_rank` | Covered | Python-native equivalent of `rankSimilarity`; exact merged CellChat object slots are not reproduced |
| Global interaction count/weight comparison | `pyccc.plotting.compare_interactions`, `pyccc.ggplot.compare_interactions` | Covered | More CellChat-specific styling presets could be added |
| Differential pathway ranking | `pyccc.plotting.diff_pathway_rank`, `pyccc.ggplot.diff_pathway_rank` | Covered | CellChat `rankNet`-style delta ranking rather than exact R output |
| Differential source-target ranking | `pyccc.plotting.diff_source_target_rank`, `pyccc.ggplot.diff_source_target_rank` | Covered | Adds pair-level delta ranking on top of CellChat's matrix views |
| Differential source-target heatmap | `pyccc.plotting.diff_heatmap` | Covered | Supports optional row/column clustering |
| Differential network circle | `pyccc.plotting.diff_network_circle` | Covered | Not a byte-for-byte CellChat port |
| Deterministic group-to-pathway river | `pyccc.plotting.pathway_river`, `pyccc.ggplot.pathway_river` | Covered | Distinct from latent-pattern river |
| Interactive group-to-pathway river | `pyccc.interactive_pathway_river` | Covered | Optional Plotly Sankey HTML for dense network inspection |
| Latent communication pattern extraction | `pyccc.compute_communication_patterns` | Covered | Pattern count is user-selected unless k-scan is used |
| Pattern number diagnostics | `pyccc.select_communication_pattern_number`, `pyccc.ggplot.pattern_number_plot` | Covered | Highlights an elbow-style recommended k; no consensus cophenetic score yet |
| Pattern dot plot | `pyccc.ggplot.pattern_dot` | Covered | Matplotlib equivalent not implemented |
| Pattern river plot | `pyccc.ggplot.pattern_river` | Covered | Uses plotnine Bezier polygon ribbons with local alluvial stacking |
| Interactive pattern river | `pyccc.interactive_pattern_river` | Covered | Optional Plotly Sankey HTML for exploring pattern links |
| Signaling pathway embedding and clustering | `pyccc.compute_pathway_similarity`, `pyccc.compute_pathway_embedding`, `pyccc.compute_pathway_clusters`, `pyccc.plotting.pathway_embedding`, `pyccc.ggplot.pathway_embedding` | Covered | Supports CellChat-style UMAP and spectral/kmeans grouping; exact R object slots and ggplot layout are not byte-for-byte identical |
| Signaling gene expression distribution | `pyccc.signaling_expression_frame`, `pyccc.plotting.signaling_gene_expression`, `pyccc.ggplot.signaling_gene_expression` | Covered | AnnData-native dot/violin/bar wrapper; exact Seurat stacked-violin layout is not byte-for-byte identical |
| Publication multi-panel gallery | `pyccc.plotting.key_plot_gallery`, `pyccc.plotting.masterpiece_gallery` | Covered | Panel selection is fixed |
| Multi-page visual report | `pyccc.save_cellchat_report` | Covered | PDF report includes visual panels and summary tables; template is fixed |
| Spatial communication plot | `pyccc.plotting.spatial_network` | Covered | Spatial overlay works with `adata.obsm["spatial"]` |
| Distance-aware spatial probabilities | `pyccc.compute_communication(..., spatial_key="spatial", distance_decay=...)` | Covered | Uses group-centroid exponential distance decay |
| Cofactor-aware visual summaries | `pyccc.compute_communication(..., cofactor_adjust=True)`, `pyccc.plotting.annotation_bar`, `pyccc.ggplot.annotation_bar` | Covered | Uses CellChat-style Hill agonist/antagonist and co-receptor factors; exact R parity tests are not implemented |
| Population-size-adjusted visuals | `pyccc.compute_communication(..., population_size=True)` plus all network/bubble plots | Covered | Uses group abundance weighting before plotting |
| Overexpressed gene/interactions gate | `pyccc.identify_overexpressed_genes`, `pyccc.identify_overexpressed_interactions`, `pyccc.compute_communication(..., de_gate=True)` | Covered | Supports fast effect-size and Wilcoxon gates; not a byte-for-byte CellChat DE wrapper |
| Truncated mean expression averaging | `pyccc.compute_communication(..., aggregate="truncated_mean", trim=0.1)` | Covered | Matches CellChat-style trimmed averaging option |
| Export to CellChat R plotting | `pyccc.export_cellchat` plus `pyccc_to_cellchat.R` | Covered | Builds a minimal CellChat `.rds` with `net`, `netP`, `LR`, `DB`, and `idents`; raw expression slots are empty |
| Export two-sample diff to merged CellChat | `pyccc.export_cellchat_merged` plus `pyccc_to_merged_cellchat.R` | Covered | Builds two minimal CellChat objects, merges them with CellChat `mergeCellChat`, and writes R-native comparison plot outputs |
| Full visual report parity | `pyccc.save_cellchat_report` | Covered | Visual PDF report exists; complete statistical-output parity is outside visual coverage |

## CellChat Alignment Defaults

For outputs intended to be directly compared with CellChat, use:

```python
pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=db,
    aggregate="tri_mean",
    score_method="cellchat",
    cofactor_adjust=True,  # when using CellChatDB metadata
)
```

The plotting defaults use a CellChat-like Spectral probability color scale for
Matplotlib and plotnine bubble/heatmap surfaces. This follows CellChat's
`netVisual_bubble` convention where dot color encodes communication probability
and dot size is used for probability or significance, while keeping pyccc's
Python-native figure layout and export path.

## Current Verification

The visual smoke suite draws Matplotlib and plotnine figures:

```bash
uv run pytest -q
```

The demo scripts export representative figures:

```bash
uv run python examples/quickstart.py
uv run python examples/masterpiece_demo.py
uv run python examples/plotnine_demo.py
uv run python examples/cellchat_official_plotnine_demo.py
```
