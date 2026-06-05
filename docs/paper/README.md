# Paper Artifacts

```{toctree}
:hidden:

submission_targets
```

This folder contains committed manuscript-planning artifacts that support the
repository's publication-style claims without requiring large local benchmark
directories.

## Contents

- `figures/main/`: six planned main figures as SVG, PDF, and PNG previews.
- `figures/supplementary/`: supplementary figure previews.
- `figures/qa_contact_sheet.png`: contact sheet for quick visual QA.
- `source_data/`: TSV source tables for numerical panels and asset manifests.
- `legends/`: per-figure legends and provenance/caveat notes.
- `manifests/manuscript_artifacts.tsv`: generated artifact manifest.
- `submission_targets.md`: journal-positioning notes.

## Regeneration

Regenerate the current manuscript figure package:

```bash
uv run python scripts/make_manuscript_figures.py --figure all
uv run python scripts/check_manuscript_artifacts.py
```

The check script verifies the required figure formats, source-data tables,
legends, QA contact sheet, artifact manifest, DB-free caveats, and the strict
4-core prepared 1M benchmark values.
