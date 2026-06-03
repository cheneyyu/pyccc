from __future__ import annotations

import argparse

import pyccc as pc


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a protein FASTA with ESMC or the deterministic hash backend.")
    parser.add_argument("--protein-fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=".pyccc-cache/esmc")
    parser.add_argument("--backend", default="auto", choices=["auto", "hash"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    proteins = pc.load_protein_fasta(args.protein_fasta)
    embeddings = pc.embed_proteins_esmc(proteins, cache_dir=args.cache_dir, backend=args.backend, batch_size=args.batch_size)
    out = embeddings.copy()
    out["embedding"] = out["embedding"].map(lambda value: ",".join(f"{float(x):.7g}" for x in value))
    out.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
