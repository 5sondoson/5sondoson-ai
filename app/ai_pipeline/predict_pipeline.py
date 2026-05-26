from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
STAGE1_DIR = ROOT_DIR / "stage1_deploy_artifacts"
STAGE2_DIR = ROOT_DIR / "stage2_deploy_artifacts"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage1 = _load_module("stage1_predict", STAGE1_DIR / "predict_stage1.py")
stage2 = _load_module("stage2_predict", STAGE2_DIR / "predict_stage2.py")


def _as_frame(data: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame([data])


def add_stage1_predictions(data: pd.DataFrame | dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stage1 and append stage1_pred_* columns for Stage2."""
    df = _as_frame(data)
    stage1_long = stage1.predict_all(df, artifact_dir=STAGE1_DIR)

    enriched = df.copy()
    if "row_index" not in stage1_long.columns:
        # predict_all concatenates one target block at a time; each block preserves
        # the input row order. Reconstruct that row identity for Stage2 joining.
        stage1_long = stage1_long.copy()
        stage1_config = stage1.load_config(STAGE1_DIR / "stage1_model_config.json")
        row_index = []
        for _target in stage1_config["targets"]:
            row_index.extend(df.index.tolist())
        stage1_long["row_index"] = row_index[: len(stage1_long)]

    for idx, group in stage1_long.groupby("row_index"):
        for _, pred_row in group.iterrows():
            col = f"stage1_pred_{pred_row['target']}"
            enriched.loc[idx, col] = float(pred_row["prediction"])

    return enriched, stage1_long


def predict_one(row: dict[str, Any] | pd.Series) -> dict[str, pd.DataFrame]:
    """Run Stage1 and Stage2 for a single player row."""
    enriched, stage1_long = add_stage1_predictions(dict(row))
    stage2_long = stage2.predict_dataframe(enriched, config=stage2.load_config())
    return {
        "stage1_predictions": stage1_long.reset_index(drop=True),
        "stage2_predictions": stage2_long.reset_index(drop=True),
        "stage2_input_row": enriched.reset_index(drop=True),
    }


def predict_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the end-to-end Stage1 -> Stage2 pipeline for a DataFrame."""
    enriched, stage1_long = add_stage1_predictions(df)
    stage2_long = stage2.predict_dataframe(enriched, config=stage2.load_config())
    return {
        "stage1_predictions": stage1_long.reset_index(drop=True),
        "stage2_predictions": stage2_long.reset_index(drop=True),
        "stage2_input_rows": enriched.reset_index(drop=True),
    }


if __name__ == "__main__":
    print(f"Stage1 artifacts: {STAGE1_DIR}")
    print(f"Stage2 artifacts: {STAGE2_DIR}")
    print("Use predict_one(row) or predict_dataframe(df) to run Stage1 -> Stage2.")
