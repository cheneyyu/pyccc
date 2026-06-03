# pyccc Main Figure Legend

Figure 1. Publication-level summary of pyccc as an AnnData-native, CellChat-compatible CCC workflow.

Panel A shows the analysis path: sparse or backed AnnData input and CellChatDB ligand-receptor metadata are reduced to LR/cofactor gene summaries before scoring. Results remain in Python-native tables and can also be exported back to minimal CellChat R objects for R-native plotting.

Panel B shows numerical agreement with direct CellChat R on the official human skin tutorial benchmark. Pearson correlation, Spearman correlation, and top-20 overlap are all 1.0 across LR-source-target probabilities, global network weights, pathway information flow, LR information flow, and MIF LR contribution; maximum absolute differences are floating-point roundoff.

Panel C compares total time for within-sample CCC, two-condition differential CCC, and matched visual outputs on a real 1,000,000-cell sample from the CELLxGENE Human Immune Health Atlas. The benchmark uses raw.X for all methods because the h5ad main X matrix is scaled and contains negative values.

Panel D summarizes the tradeoff between GPU-friendly mean summaries and CellChat triMean similarity on the local CELLxGENE benchmark. The gated mean variants preserve higher rank agreement to triMean than plain clipped means while remaining faster than the triMean CPU reference in this environment.

Statistical notes. All reported communication scores are inferred group-level quantities from expression summaries and ligand-receptor metadata, not direct physical interaction measurements. The CellChat parity panel uses the same filtered LR table and metadata as the R reference. The 1M-cell runtime panel reports one local run on the stated environment and should be interpreted as an artifact-level benchmark rather than a universal performance guarantee.
