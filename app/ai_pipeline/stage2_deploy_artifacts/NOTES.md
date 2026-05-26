# Selected Stage2 Strategy

The selected experiment is:

```text
hierarchical league feature sets
target-wise top3 combinations
100-trial random search
objective = minimum OOF after MAE
```

Categorical league context was represented by splitting `transfer_path`:

```text
source_league = left side of " -> "
destination_league = right side of " -> "
```

Targets with negative MAE improvement are retained in the config for transparency, but `apply_stage2=false` so production prediction falls back to Stage1.
