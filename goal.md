# pyccc publication-readiness goal

## Current Objective

Prepare pyccc as a publication-quality, ReadTheDocs-ready Python package for
CellChat-compatible cell-cell communication analysis.

## Scope

- Keep the supported inference path CPU/NumPy/SciPy based.
- Preserve CellChat-like scoring, grouped sparse aggregation, differential CCC,
  and export back to CellChat R objects.
- Document every public plotting function with minimal runnable examples and
  rendered outputs.
- Keep README concise: installation, core workflow, benchmark summary, and
  links to detailed docs. Do not place the manuscript main figure at the top of
  the GitHub landing page.
- Add ReadTheDocs configuration so the hosted docs can build from the GitHub
  repository.

## Acceptance Criteria

- `compute_communication` no longer contains an accelerator-specific backend.
- Public docs do not claim an accelerator backend.
- Matplotlib and plotnine plotting APIs have a gallery page with sample calls
  and generated figures.
- Sphinx documentation builds locally.
- Tests pass locally.
- The repository can be imported into ReadTheDocs from GitHub.

## Notes

Large-data benchmarks should focus on the three supported user strategies:
pyccc native plots, pyccc compute plus CellChat R plotting, and direct CellChat R.
The current million-cell benchmark uses the CELLxGENE Human Immune Health Atlas
and reports the same within-sample, two-condition differential, and matched
visualization workflow for each strategy.
