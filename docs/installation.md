# Installation

Install directly from GitHub:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

For development, clone the repository and create the local `uv` environment in
one step:

```bash
git clone https://github.com/cheneyyu/pyccc.git
cd pyccc
uv sync --extra dev --extra ggplot --extra interactive --extra omnipath --extra docs
uv run pytest -q
```

Add LIANA support only when you need LIANA import helpers:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath,liana] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

Add experimental DB-free prediction dependencies only when you need the
ESMC-300M embedding, LightGBM protein role classifiers, LightGBM pair ranker,
and clade-aware density prior workflow:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath,predict] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

Optional extras:

- `ggplot`: plotnine-based plotting functions in `pyccc.ggplot`
- `interactive`: Plotly-based interactive river plots
- `omnipath`: OmniPath ligand-receptor database loader
- `predict`: experimental ESMC-300M/LightGBM DB-free LR prediction dependencies
- `docs`: Sphinx and MyST dependencies for documentation builds
- `liana`: LIANA import helpers

Core usage:

```python
import pyccc as pc

db = pc.load_cellchatdb("human", category="Secreted Signaling")
mouse_db = pc.load_cellchatdb("mouse", category="Secreted Signaling")
omni = pc.load_omnipath_interactions("mouse", resources=["CellChatDB", "CellPhoneDB"])
result = pc.compute_communication(
    adata,
    groupby="cell_type",
    lr_table=db,
    gene_symbols_key="gene_symbols",
    score_method="cellchat",
)
```
