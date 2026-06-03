from __future__ import annotations

import argparse

import pyccc as pc


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a local LR resource table for pyccc predictor training.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--schema", default="generic")
    parser.add_argument("--species", required=True)
    parser.add_argument("--taxon-id", required=True)
    parser.add_argument("--resource")
    parser.add_argument("--license", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pc.load_training_lr_resources(
        [
            {
                "path": args.path,
                "schema": args.schema,
                "species": args.species,
                "taxon_id": args.taxon_id,
                "resource": args.resource or args.schema,
                "license": args.license,
                "source_url": args.source_url,
            }
        ]
    )
    frame.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
