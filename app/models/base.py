"""모든 모델 predictor의 베이스 클래스.

핵심:
- 실제 모델이든 mock이든 같은 인터페이스로 동작
- 모델 교체 시 코드 수정 없이 파일 교체만으로 가능
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    name: str
    version: str
    output_keys: list[str]

    @abstractmethod
    def predict(self, features: dict) -> dict[str, float]:
        """단일 선수 피처 dict -> 예측 결과 dict."""
        ...

    @property
    def is_mock(self) -> bool:
        return self.version.endswith("-mock") or "dummy" in self.version


class ModelPredictor(BasePredictor):
    """실제 sklearn Pipeline 모델을 감싸는 predictor."""

    def __init__(self, name: str, version: str, pipeline, output_keys: list[str]):
        self.name = name
        self.version = version
        self.pipeline = pipeline
        self.output_keys = output_keys

    @classmethod
    def load(cls, path: str | Path, output_keys: list[str]) -> "ModelPredictor":
        """파일에서 모델 로딩. 파일명 패턴: {name}_v{x.y.z}.joblib"""
        path = Path(path)
        pipeline = joblib.load(path)
        stem = path.stem
        if "_v" in stem:
            name_part, version_part = stem.rsplit("_v", 1)
            name = name_part
            version = f"v{version_part}"
        else:
            name = stem
            version = "unknown"
        return cls(name=name, version=version, pipeline=pipeline, output_keys=output_keys)

    def predict(self, features: dict) -> dict[str, float]:
        X = pd.DataFrame([features])
        preds = self.pipeline.predict(X)
        if isinstance(preds, np.ndarray):
            if preds.ndim == 1:
                values = [float(preds[0])]
            else:
                values = [float(v) for v in preds[0]]
        else:
            values = [float(preds)]
        # output_keys 개수와 안 맞으면 첫 키만 사용
        if len(values) != len(self.output_keys):
            return {self.output_keys[0]: values[0]}
        return dict(zip(self.output_keys, values))


class MockPredictor(BasePredictor):
    """모델이 아직 없을 때 그럴듯한 값을 반환하는 가짜 predictor.

    같은 입력 -> 같은 출력 (seed_from_input 기반). 백엔드 테스트 시 헷갈리지 않게.
    """

    def __init__(
        self,
        name: str,
        output_keys: list[str],
        value_ranges: dict | None = None,
        seed_from_input: str | None = None,
    ):
        self.name = name
        self.version = "v0.0.0-mock"
        self.output_keys = output_keys
        self.value_ranges = value_ranges or {}
        self.seed_from_input = seed_from_input

    def predict(self, features: dict) -> dict[str, float]:
        if self.seed_from_input and self.seed_from_input in features:
            seed = int(abs(hash(str(features[self.seed_from_input]))) % (2**31))
        else:
            seed = 42
        rng = np.random.default_rng(seed)
        return {
            key: float(rng.uniform(*self.value_ranges.get(key, (0.0, 1.0))))
            for key in self.output_keys
        }
