# Stage2 Deploy Artifacts

Stage2 is a transfer-aware correction layer on top of Stage1.

```text
final_after_pred = stage1_pred + stage2_delta_pred
```

For targets where cross-validated Stage2 correction did not improve after MAE, `apply_stage2=false` in `stage2_model_config.json`; serving should keep the Stage1 prediction as the final prediction for those targets.

## Files

- `models/*.joblib`: selected Stage2 delta models refit on all usable rows for each target.
- `stage2_model_config.json`: target-to-model mapping, selected features, CV metrics, and fallback flags.
- `stage2_feature_schema.json`: required raw columns and derived feature notes.
- `stage2_selected_models_metrics.csv`: selected model metrics from the hierarchical top3 + 100trial experiment.
- `predict_stage2.py`: helper functions for loading artifacts and producing final after predictions.

## Required Input

Each input row should include:

- Stage1 prediction columns listed in `stage2_feature_schema.json`
- `transfer_path`, formatted like `Liga Portugal -> La Liga`
- `position_code`
- `player_age_before`
- `before_stat_minutes_played_total_num`

`source_league` and `destination_league` are derived automatically from `transfer_path`.

## Usage

```python
import pandas as pd
from predict_stage2 import predict_dataframe

df = pd.read_csv("stage2_input.csv")
predictions = predict_dataframe(df)
```

The output is long-form: one row per input row and target.
