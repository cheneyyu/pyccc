# Real-Data DB-Free Spatial Validation

This workflow validates DB-free target-species CCC on real special-species
spatial transcriptomics. It does not retrain the DB-free predictor. It produces
source tables and a manuscript-style figure from ARTISTA axolotl and SOTA
soybean spatial sections.

The conservative claim is:

> DB-free LR candidates from ESMC-300M embedding, LightGBM protein role
> classifiers, a LightGBM pair ranker, and a clade-aware density prior are
> spatially more enriched than matched-random, score-shuffled, role-only,
> embedding-only, and expression-only controls.

Predicted LR edges are computational candidates, not validated biochemical
binding events.

## Data Sources

The committed manifests are the single source of truth:

- `configs/dbfree_validation/artista_axolotl.yaml`
- `configs/dbfree_validation/sota_soybean.yaml`

The ARTISTA manifest uses the ARTISTA download page for axolotl Stereo-seq
h5ad sections, including `Stage44.h5ad`, `Control_Juv.h5ad`, `5DPI_1.h5ad`,
and `30DPI.h5ad`.

The SOTA manifest uses the SOTA soybean download page for `SAM.spatial.h5ad`
and `Leaf.spatial.h5ad`.

Each manifest also records a target-species `protein_source`: URL, raw local
FASTA path, gene-ID regex, optional gene-ID replacements, and isoform selection
rule. `prepare_target_proteome.py` converts that raw source into the normalized
`protein_fasta` path used by prediction, with one longest protein per gene and
headers rewritten as `gene=<expression_gene_id>`. When
`restrict_to_expression` is enabled, that prediction FASTA is further limited
to genes present in the prepared spatial sections, avoiding unnecessary
ESMC-300M embedding of proteins that cannot enter the CCC analysis.

## Reproduce

Run the same four scripts for each manifest:

```bash
uv run python scripts/download_dbfree_validation_data.py \
  --manifest configs/dbfree_validation/artista_axolotl.yaml \
  --include-proteome \
  --smoke-only

uv run python scripts/prepare_dbfree_validation_data.py \
  --manifest configs/dbfree_validation/artista_axolotl.yaml \
  --smoke-only

uv run python scripts/prepare_target_proteome.py \
  --manifest configs/dbfree_validation/artista_axolotl.yaml

uv run python scripts/run_dbfree_spatial_validation.py \
  --manifest configs/dbfree_validation/artista_axolotl.yaml \
  --smoke-only \
  --n-permutations 100

uv run python scripts/make_dbfree_spatial_validation_figure.py \
  --results-dir results/dbfree_validation

uv run python scripts/check_dbfree_validation_acceptance.py \
  --manifest configs/dbfree_validation/artista_axolotl.yaml \
  --manifest configs/dbfree_validation/sota_soybean.yaml \
  --results-dir results/dbfree_validation
```

For final figure runs, omit `--smoke-only` and use `--n-permutations 1000`.

## Outputs

The pipeline writes:

- `results/dbfree_validation/download_manifest.tsv`
- `results/dbfree_validation/*/section_qc.tsv`
- `results/dbfree_validation/*/gene_protein_match.tsv`
- `results/dbfree_validation/*/predicted_lr.tsv`
- `results/dbfree_validation/*/prediction_summary.tsv`
- `results/dbfree_validation/*/spatial_validation_summary.tsv`
- `results/dbfree_validation/*/spatial_validation_top_k_enrichment.tsv`
- `results/dbfree_validation/*/spatial_validation_distance_decay.tsv`
- `results/dbfree_validation/*/spatial_validation_section_reproducibility.tsv`
- `results/dbfree_validation/baseline_comparison.tsv`
- `results/dbfree_validation/baseline_topk_enrichment.tsv`
- `figures/dbfree_spatial_validation_main.png`
- `figures/dbfree_spatial_validation_main.svg`
- `figures/dbfree_spatial_validation_main.pdf`
- `figures/dbfree_spatial_validation_main_legend.md`
- `figures/dbfree_spatial_validation_main_source_tables.tar.gz`
- `results/dbfree_validation/acceptance_report.tsv`

Final figure tables must not contain fixture warnings such as
`hash_embedding_backend`, `heuristic_role_model`, `heuristic_pair_ranker`, or
`default_unknown_density`.

## Acceptance Gates

For ARTISTA, a publishable run requires at least two of three selected main
sections to have top-500 or top-1000 DB-free enrichment z-score at least 2,
empirical p-value at most 0.05 against matched random LR, and improvement over
role-only or materially better top-K stability.

For SOTA, a feasibility run requires at least one of two organs to have
positive DB-free enrichment over matched random LR.

Gene/protein mapping gates are recorded in
`gene_protein_match_summary.tsv`: ARTISTA requires at least 60 percent of
expressed genes to map to target proteins, and SOTA requires at least
70 percent.

`check_dbfree_validation_acceptance.py` exits non-zero until all required
data, sequence, model, spatial, figure, and reproducibility gates pass. This is
expected during smoke tests before real target proteomes and trained models are
available.
