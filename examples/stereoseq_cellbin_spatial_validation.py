from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc


def main() -> None:
    adata = AnnData(
        np.array([[5, 0], [4, 0], [0, 3], [0, 4]], dtype=float),
        obs=pd.DataFrame(
            {"cell_type": ["sender", "sender", "receiver", "receiver"], "cell_area": [1.0, 1.0, 1.0, 1.0]},
            index=[f"cell{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["L1", "R1"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    predicted = pc.CellChatDB(
        pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "pathway": ["DB-free predicted"], "model_score": [0.9]}),
        name="toy_predicted",
    )
    report = pc.validate_spatial_lr_table(adata, predicted, groupby="cell_type", n_permutations=10)
    print(report.summary)


if __name__ == "__main__":
    main()
