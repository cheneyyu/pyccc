# Installation

Clone the repository and create the local `uv` environment in one step:

```bash
git clone https://github.com/cheneyyu/pyccc.git
cd pyccc
uv sync --extra dev --extra ggplot --extra interactive --extra docs
uv run pytest -q
```

This installs the core package, test tools, plotnine-style plotting,
interactive Plotly outputs, and documentation dependencies.

Add LIANA support only when you need LIANA import helpers:

```bash
uv sync --extra liana --extra dev --extra ggplot --extra interactive --extra docs
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
