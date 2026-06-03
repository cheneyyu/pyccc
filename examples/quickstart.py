import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc
import pyccc.plotting as cp

cp.set_theme(context="notebook", font_scale=0.85)


def make_demo_adata():
    x = np.array(
        [
            [5, 0, 0, 0, 1, 0],
            [4, 0, 0, 0, 1, 0],
            [0, 3, 2, 2, 0, 1],
            [0, 4, 3, 3, 0, 1],
            [0, 0, 4, 0, 5, 2],
            [0, 0, 5, 0, 4, 2],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "cell_type": ["Sender", "Sender", "Receiver", "Receiver", "Stroma", "Stroma"],
            "condition": ["ctrl", "stim", "ctrl", "stim", "ctrl", "stim"],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["TGFB1", "TGFBR1", "TGFBR2", "CXCR4", "CXCL12", "CD74"])
    return AnnData(x, obs=obs, var=var)


adata = make_demo_adata()
lr = pc.toy_lr_table()

cellchat_kwargs = dict(min_pct=0.0, aggregate="tri_mean", score_method="cellchat")
ctrl = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="ctrl", **cellchat_kwargs)
stim = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="stim", **cellchat_kwargs)
diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
cp.net_circle(stim, ax=axes[0, 0])
cp.net_heatmap(stim, ax=axes[0, 1])
cp.bubble(stim, ax=axes[0, 2], top_n=10)
cp.pathway_heatmap(stim, ax=axes[1, 0])
cp.signaling_role_network(stim, ax=axes[1, 1])
cp.pathway_river(stim, ax=axes[1, 2], top_n=6)
cp.compare_interactions(diff, ax=axes[2, 0])
cp.diff_heatmap(diff, ax=axes[2, 1])
cp.diff_bubble(diff, ax=axes[2, 2], top_n=10)
fig.savefig("pyccc_quickstart.png", dpi=160)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
cp.net_chord_gene(stim, pathways=["TGFb", "CXCL"], top_n=8, ax=ax, title="LR-mediated chord")
fig.savefig("pyccc_quickstart_lr_chord.png", dpi=180)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
cp.net_individual(stim, pathway="TGFb", layout="hierarchy", vertex_receiver=["Receiver"], ax=axes[0], title="Individual LR hierarchy")
cp.net_individual(stim, pathway="TGFb", layout="circle", ax=axes[1], title="Individual LR circle")
cp.net_individual(stim, pathway="TGFb", layout="chord", ax=axes[2], title="Individual LR chord")
fig.savefig("pyccc_quickstart_lr_individual.png", dpi=180)
plt.close(fig)
