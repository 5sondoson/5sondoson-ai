from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ARTIFACT_DIR / "stage2_model_config.json"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def split_transfer_path(value: object) -> tuple[str, str]:
    text = "missing" if pd.isna(value) else str(value)
    if " -> " not in text:
        return text or "missing", "missing"
    source, destination = text.split(" -> ", 1)
    return source or "missing", destination or "missing"


def prepare_stage2_input(row: dict[str, Any] | pd.Series, model_cfg: dict[str, Any]) -> pd.DataFrame:
    data = dict(row)
    stage1_col = model_cfg["stage1_pred_col"]
    if stage1_col not in data:
        raise KeyError(f"Missing required stage1 prediction column: {stage1_col}")

    source, destination = split_transfer_path(data.get("transfer_path"))
    data["source_league"] = source
    data["destination_league"] = destination
    data["stage1_pred"] = data[stage1_col]

    required = model_cfg["input_features"]
    missing = [col for col in required if col not in data]
    if missing:
        raise KeyError(f"Missing required Stage2 input columns for {model_cfg['target']}: {missing}")
    return pd.DataFrame([{col: data[col] for col in required}])


def predict_one(row: dict[str, Any] | pd.Series, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = load_config() if config is None else config
    outputs: list[dict[str, Any]] = []
    for model_cfg in config["models"]:
        stage1_pred = float(dict(row)[model_cfg["stage1_pred_col"]])
        if model_cfg.get("apply_stage2", True):
            bundle = joblib.load(ARTIFACT_DIR / model_cfg["model_file"])
            pipeline = bundle["pipeline"]
            x = prepare_stage2_input(row, model_cfg)
            stage2_delta_pred = float(pipeline.predict(x)[0])
            final_pred = stage1_pred + stage2_delta_pred
            applied = True
        else:
            stage2_delta_pred = 0.0
            final_pred = stage1_pred
            applied = False
        outputs.append(
            {
                "target": model_cfg["target"],
                "target_short_name": model_cfg["target_short_name"],
                "stage1_pred_col": model_cfg["stage1_pred_col"],
                "stage1_pred": stage1_pred,
                "stage2_delta_pred": stage2_delta_pred,
                "final_after_pred": final_pred,
                "stage2_applied": applied,
                "model_type": model_cfg["model_type"],
                "feature_set": model_cfg["feature_set"],
                "cv_after_mae_improvement_pct": model_cfg["cv_metrics"]["after_mae_improvement_pct"],
            }
        )
    return outputs


def predict_dataframe(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    config = load_config() if config is None else config
    rows = []
    for idx, row in df.iterrows():
        for output in predict_one(row, config=config):
            rows.append({"row_index": idx, **output})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    config = load_config()
    print(f"Loaded {len(config['models'])} Stage2 model configs from {CONFIG_PATH}")
