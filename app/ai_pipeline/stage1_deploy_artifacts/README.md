# Stage1 Deploy Artifacts

This folder contains deployable Stage1 single-target prediction artifacts.

## Contents

- `models/*.pkl`: fitted target-specific sklearn pipelines.
- `stage1_model_config.json`: model path, target, features, alpha blending, confidence policy, and metrics.
- `stage1_feature_schema.json`: required input feature schema by target.
- `predict_stage1.py`: lightweight prediction helper.
- `requirements.txt`: minimal Python dependencies.

## Prediction Policy

Final prediction:

```text
prediction = alpha_model * model_prediction + alpha_current * current_season_same_stat
```

Confidence is based on current-season minutes only:

```text
high   : current_minutes >= minutes_threshold
medium : 0.7 * minutes_threshold <= current_minutes < minutes_threshold
low    : current_minutes < 0.7 * minutes_threshold or missing current_minutes
```

The training/evaluation cohort used both current and next-season minute filters. In live prediction,
next-season minutes are unknown, so the exported predictor only uses current minutes for confidence.

## Excluded Target

`stat_clearances_total_per90_next` is intentionally excluded.
