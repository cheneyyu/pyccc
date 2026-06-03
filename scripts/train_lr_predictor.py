from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import pyccc as pc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pyccc LR link predictor from normalized LR resources and embeddings.")
    parser.add_argument("--training-lr", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "sklearn"])
    parser.add_argument("--density-groupby", default="clade")
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--easy-negative-fraction", type=float, default=0.05)
    parser.add_argument("--excluded-homology-radius", default="family_pair", choices=["family_pair", "none"])
    parser.add_argument("--negative-repeats", type=int, default=3)
    args = parser.parse_args()
    train = pd.read_csv(args.training_lr, sep="\t")
    embeddings = pd.read_csv(args.embeddings, sep="\t")
    embeddings["embedding"] = embeddings["embedding"].map(lambda value: np.asarray([float(x) for x in str(value).split(",")], dtype=np.float32))
    pc.train_lr_link_predictor(
        train,
        embeddings,
        model=args.model,
        density_groupby=args.density_groupby,
        output_dir=args.output_dir,
        negative_ratio=args.negative_ratio,
        easy_negative_fraction=args.easy_negative_fraction,
        excluded_homology_radius=args.excluded_homology_radius,
        negative_repeats=args.negative_repeats,
    )


if __name__ == "__main__":
    main()
