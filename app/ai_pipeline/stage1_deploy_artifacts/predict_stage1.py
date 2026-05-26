from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ARTIFACT_DIR / "stage1_model_config.json"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def assign_confidence(current_minutes: float | int | None, threshold: int) -> tuple[str, str | None, float | None]:
    if current_minutes is None or pd.isna(current_minutes):
        return "low", "missing_current_minutes", None
    ratio = float(current_minutes) / float(threshold) if threshold > 0 else None
    if ratio is not None and ratio >= 1.0:
        return "high", None, ratio
    if ratio is not None and ratio >= 0.7:
        return "medium", "below_minutes_threshold", ratio
    return "low", "far_below_minutes_threshold", ratio


def _as_frame(data: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame([data])


def predict_target(
    data: pd.DataFrame | dict[str, Any],
    target: str,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> pd.DataFrame:
    artifact_dir = Path(artifact_dir)
    config = load_config(artifact_dir / "stage1_model_config.json")
    if target not in config["targets"]:
        raise KeyError(f"Unknown target: {target}")

    target_cfg = config["targets"][target]
    df = _as_frame(data)

    features = target_cfg["selected_features"]
    x = df.reindex(columns=features)

    model = joblib.load(artifact_dir / target_cfg["model_path"])
    model_pred = model.predict(x)

    current_feature = target_cfg["current_feature"]
    alpha_model = float(target_cfg["alpha_model"])
    alpha_current = float(target_cfg["alpha_current"])

    warnings = []
    if current_feature in df.columns:
        current_values = pd.to_numeric(df[current_feature], errors="coerce").to_numpy(dtype=float)
        missing_current = np.isnan(current_values)
        blended = alpha_model * model_pred + alpha_current * np.nan_to_num(current_values, nan=0.0)
        blended[missing_current] = model_pred[missing_current]
        warnings = ["missing_current_feature_value" if flag else None for flag in missing_current]
    else:
        blended = model_pred
        warnings = ["missing_current_feature"] * len(df)

    minute_values = pd.to_numeric(df.get("stat_minutes_played_total", pd.Series([np.nan] * len(df))), errors="coerce")
    confidence_rows = [
        assign_confidence(value, int(target_cfg["minutes_threshold"]))
        for value in minute_values.to_numpy()
    ]

    rows = []
    for idx, pred in enumerate(blended):
        confidence, minutes_warning, minutes_ratio = confidence_rows[idx]
        warning_parts = [part for part in [warnings[idx], minutes_warning] if part]
        rows.append(
            {
                "target": target,
                "position": target_cfg["position"],
                "prediction": float(pred),
                "unit": target_cfg["unit"],
                "confidence": confidence,
                "warning": "|".join(warning_parts) if warning_parts else None,
                "current_minutes": None if pd.isna(minute_values.iloc[idx]) else float(minute_values.iloc[idx]),
                "minutes_threshold": int(target_cfg["minutes_threshold"]),
                "minutes_ratio": minutes_ratio,
                "alpha_model": alpha_model,
                "alpha_current": alpha_current,
                "model_prediction": float(model_pred[idx]),
            }
        )
    return pd.DataFrame(rows)


def predict_all(data: pd.DataFrame | dict[str, Any], artifact_dir: str | Path = ARTIFACT_DIR) -> pd.DataFrame:
    config = load_config(Path(artifact_dir) / "stage1_model_config.json")
    outputs = [predict_target(data, target, artifact_dir) for target in config["targets"]]
    return pd.concat(outputs, ignore_index=True)
