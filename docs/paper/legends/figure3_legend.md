# Figure 3. Million-cell scaling and sparse tri-mean optimization

pyccc completes a matched 1M-cell CellChat-like workflow in 11.2 seconds from
prepared input under strict 4-core affinity.

Panel A shows the benchmark design. Panel B shows total runtime from prepared
input read through matched visualizations. Panel C decomposes time by input read,
compute, export, and plotting components. Panel D summarizes the sparse
tri-mean optimization. Panel E reports peak CPU cores. Panel F states the timing
policy.

Data source: `docs/paper/source_data/figure3_runtime_benchmark.tsv`,
`figure3_runtime_decomposition.tsv`, and `figure3_cpu_affinity.tsv`.
The benchmark uses prepared 1M input, strict 4-core affinity, and regular mode
only. Sampling from the full 1.82M-cell h5ad and LR filtering/preparation are
not counted.
