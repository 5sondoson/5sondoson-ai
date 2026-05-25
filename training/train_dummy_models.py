"""임의(dummy) 모델 학습 스크립트.

목적: AI팀이 진짜 모델 주기 전에 같은 형태의 .joblib 파일을 만들어서
서버가 ModelPredictor로 정상 로딩되는지 검증.

이 파일은 일회용. 실제 운영에서는 AI팀이 학습한 모델 .joblib을 받아서 교체.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "app" / "models"
DUMMY_VERSION = "v0.0.1-dummy"

# 백엔드 PR #9 스펙 기준 포지션별 출력 키
POSITION_OUTPUT_KEYS = {
    "FW": ["goals", "shots", "dribbles", "key_passes", "pass_accuracy"],
    "MF": ["passes", "key_passes", "tackles", "pass_accuracy"],
    "DF": ["aerials_won", "blocked_shots", "pass_accuracy"],
    "GK": ["saves", "cleansheets", "pass_accuracy"],
}

LEAGUES = ["EPL", "LA", "SA", "BL", "L1"]
POSITIONS = ["FW", "MF", "DF", "GK"]

# 학습 시 사용할 피처 (포지션 무관, 공통)
COMMON_FEATURES = [
    "stat_minutes_played_total",
    "stat_goals_total",
    "stat_shots_total",
    "stat_assists_total",
    "stat_passes_total",
    "stat_key_passes_total",
    "stat_tackles_total",
    "stat_aerials_won_total",
    "stat_clearances_total",
    "stat_blocked_shots_total",
    "stat_saves_total",
    "stat_cleansheets_total",
    "stat_accurate_passes_pct",
    "age",
    "height",
    "weight",
]


def build_pipeline(feature_names: list[str], n_outputs: int) -> Pipeline:
    """sklearn Pipeline (전처리 + 모델).

    lambda 쓰지 말 것 (pickle 불가). 명시적 컬럼 리스트 사용.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_names),
        ]
    )
    base_model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    if n_outputs > 1:
        model = MultiOutputRegressor(base_model)
    else:
        model = base_model
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def make_dummy_data(n_samples: int, feature_names: list[str], n_outputs: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.uniform(0, 100, size=(n_samples, len(feature_names))),
        columns=feature_names,
    )
    if n_outputs == 1:
        y = rng.uniform(0, 1, size=n_samples)
    else:
        y = rng.uniform(0, 1, size=(n_samples, n_outputs))
    return X, y


def save_with_metadata(pipeline, save_path: Path, name: str,
                       feature_names: list[str], output_keys: list[str]):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    metadata = {
        "model_name": name,
        "version": DUMMY_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_features": feature_names,
        "output_keys": output_keys,
        "note": "Dummy model. Replace with real model from AI team.",
    }
    save_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  생성: {save_path.name}")


def train_performance():
    """리그 5 x 포지션 4 = 20개."""
    print("\n[Performance] 리그/포지션별 dummy 모델 학습")
    for league in LEAGUES:
        for position in POSITIONS:
            output_keys = POSITION_OUTPUT_KEYS[position]
            n_outputs = len(output_keys)
            X, y = make_dummy_data(200, COMMON_FEATURES, n_outputs,
                                   seed=hash(f"{league}_{position}") % 10000)
            pipeline = build_pipeline(COMMON_FEATURES, n_outputs)
            pipeline.fit(X, y)
            save_with_metadata(
                pipeline,
                MODELS_DIR / "performance" / f"{league}_{position}_{DUMMY_VERSION}.joblib",
                name=f"{league}_{position}_performance",
                feature_names=COMMON_FEATURES,
                output_keys=output_keys,
            )


def train_market_value():
    """포지션 4개."""
    print("\n[Market Value] 포지션별 dummy 모델 학습")
    features = COMMON_FEATURES + ["target_league_idx"]
    for position in POSITIONS:
        X, _ = make_dummy_data(200, features, 1,
                              seed=hash(f"mv_{position}") % 10000)
        rng = np.random.default_rng(hash(position) % 10000)
        y = rng.uniform(1_000_000, 100_000_000, size=200)
        pipeline = build_pipeline(features, 1)
        pipeline.fit(X, y)
        save_with_metadata(
            pipeline,
            MODELS_DIR / "market_value" / f"market_value_{position}_{DUMMY_VERSION}.joblib",
            name=f"market_value_{position}",
            feature_names=features,
            output_keys=["market_value_eur"],
        )


def train_similarity():
    """포지션 4개. 유사도 모델은 임시로 회귀 모델로 대체."""
    print("\n[Similarity] 포지션별 dummy 모델 학습")
    features = ["player_id", "candidate_idx"]
    for position in POSITIONS:
        X, _ = make_dummy_data(200, features, 1,
                              seed=hash(f"sim_{position}") % 10000)
        rng = np.random.default_rng(hash(f"sim_{position}") % 10000)
        y = rng.uniform(0.5, 0.99, size=200)
        pipeline = build_pipeline(features, 1)
        pipeline.fit(X, y)
        save_with_metadata(
            pipeline,
            MODELS_DIR / "similar" / f"similarity_{position}_{DUMMY_VERSION}.joblib",
            name=f"similarity_{position}",
            feature_names=features,
            output_keys=["similarity"],
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Dummy 모델 학습 시작")
    print("=" * 60)
    train_performance()
    train_market_value()
    train_similarity()
    n_files = sum(1 for _ in MODELS_DIR.rglob("*.joblib"))
    print()
    print("=" * 60)
    print(f"완료! 생성된 .joblib 파일: {n_files}개")
    print("=" * 60)
