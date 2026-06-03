# Ligand-Receptor Resources

pyccc supports curated LR databases, OmniPath-backed resources, and local
resource exports used for experimental DB-free predictor training.

Recommended install:

```bash
python -m pip install "pyccc[ggplot,interactive,omnipath] @ git+https://github.com/cheneyyu/pyccc.git@main"
```

## CellChatDB

```python
import pyccc as pc

human = pc.load_cellchatdb("human")
mouse = pc.load_cellchatdb("mouse", category="Secreted Signaling")
ecm = pc.load_cellchatdb("human", category=["ECM-Receptor"])
```

`load_cellchatdb` returns a `CellChatDB` object that can be passed directly to
`compute_communication`.

## OmniPath

```python
omni = pc.load_omnipath_interactions(
    organism="mouse",
    resources=["CellChatDB", "CellPhoneDB", "LRdb"],
)
cellchat_like = pc.filter_lr_table(omni, resources="CellChatDB")
```

OmniPath is optional and mocked in tests; CI does not depend on live network
calls.

## Local Training Resources

For non-model species predictor work, normalize local files instead of relying
on brittle live download code:

```python
train_lr = pc.load_training_lr_resources([
    {"path": "cellchat_human.tsv", "schema": "cellchat", "species": "human", "taxon_id": 9606, "license": "CellChat license"},
    {"path": "flyphonedb2.tsv", "schema": "flyphonedb2", "species": "drosophila", "taxon_id": 7227},
    {"path": "plantphonedb.tsv", "schema": "plantphonedb", "species": "arabidopsis", "taxon_id": 3702},
    {"path": "plantcellchat.tsv", "schema": "plantcellchat", "species": "rice", "taxon_id": 4530},
])
```

Supported local schemas are `cellchat`, `omnipath`, `cellphonedb`,
`flyphonedb2`, `plantphonedb`, `plantcellchat`, and `generic`.

All resources normalize to:

- `ligand_gene`
- `receptor_gene`
- `species`
- `taxon_id`
- `resource`
- `evidence_type`
- `annotation`
- `pathway`

pyccc also preserves provenance fields such as `pmid`, `source_url`,
`curation_type`, `confidence_original`, `license`, `support_count`, and
`support_resources` when available. These can come from the resource spec or
from columns in the local table, including common aliases such as `PMID`,
`License`, `source_url`, `url`, `directed`, and `is_directed`.

When the same species/ligand/receptor/pathway row appears in multiple
resources, pyccc keeps one normalized row, records `support_count`, and
semicolon-merges provenance fields such as `resource`, `support_resources`,
`pmid`, `source_url`, `license`, and `evidence_type`. Explicitly undirected
rows are rejected in strict mode because predictor training expects directed
LR examples.

Normalize from the command line:

```bash
uv run python scripts/normalize_lr_resource.py \
  --path plantcellchat.tsv \
  --schema plantcellchat \
  --species rice \
  --taxon-id 4530 \
  --output normalized_plantcellchat.tsv
```

The normalized table can be converted back into a CellChatDB-compatible object:

```python
db = pc.training_lr_to_cellchatdb(train_lr)
```
