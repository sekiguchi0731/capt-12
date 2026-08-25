# Prior-art comparison outputs

`results.csv` is intentionally schema-only until execution is authorized. It must not be interpreted as a completed or empty-result experiment. `method_contracts.json` is execution-independent and freezes method names, online-input/deployability status, and comparison roles before any result is inspected.

When the ongoing external experiment is complete, the privacy-matched runner must replace the schema-only CSV with measured rows and add certificate bundles, lower audits, figure-source CSVs, figures, resolved configs, runtime/memory records, and the deterministic review packet in this directory.
