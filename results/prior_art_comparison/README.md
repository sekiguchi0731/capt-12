# Prior-art comparison outputs

`results.csv` is intentionally schema-only until execution is authorized. It must not be interpreted as a completed or empty-result experiment. `method_contracts.json` is execution-independent and freezes method names, online-input/deployability status, and comparison roles before any result is inspected.

When execution is authorized, the dedicated runner writes measured rows,
certificate bundles, figures, resolved configs, runtime records, and the review
packet under ignored `outputs/prior_art_comparison/`. This checked-in directory
remains an execution-independent schema/contract fixture; publishing measured
rows here is a separate reviewed action.
