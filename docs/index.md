# pyccc

pyccc is an AnnData-native, CellChat-compatible cell-cell communication toolkit.
In regular mode, it reproduces the covered CellChat-style inference,
differential analysis, and plotting outputs while scaling faster on matching
workflows. It also provides DB-free ligand-receptor prediction for species
without curated LR databases. Results are stored in tidy pandas tables, plotting
is available through Matplotlib and plotnine APIs, and CellChat R export is
supported for R-native plotting.

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
lr_resources
spatial_validation
plotting_gallery
million_cell_scaling
reproducibility
cellchat_visual_coverage
```

```{toctree}
:maxdepth: 2
:caption: DB-free CCC

lr_resource_training
dbfree_ccc
dbfree_validation
predictor_model_card
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
