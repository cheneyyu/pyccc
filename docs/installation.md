# Installation

Install the core package:

```bash
pip install pyccc
```

For local development from this repository:

```bash
uv sync --extra dev --extra ggplot --extra interactive --extra docs
```

Optional extras:

- `ggplot`: plotnine-based plotting functions in `pyccc.ggplot`
- `interactive`: Plotly-based interactive river plots
- `docs`: Sphinx and MyST dependencies for documentation builds
- `liana`: LIANA import helpers

Core usage:

```python
import pyccc as pc

db = pc.load_cellchatdb("human", category="Secreted Signaling")
result = pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=db,
    gene_symbols_key="gene_symbols",
    score_method="cellchat",
)
```
