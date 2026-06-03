from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import pyccc as pc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pyccc protein role classifier from normalized LR resources and embeddings.")
    parser.add_argument("--training-lr", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "sklearn"])
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    args = parser.parse_args()
    train = pd.read_csv(args.training_lr, sep="\t")
    embeddings = pd.read_csv(args.embeddings, sep="\t")
    embeddings["embedding"] = embeddings["embedding"].map(lambda value: np.asarray([float(x) for x in str(value).split(",")], dtype=np.float32))
    pc.train_protein_role_classifier(
        train,
        embeddings,
        model=args.model,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
