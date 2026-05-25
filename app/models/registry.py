"""모델 레지스트리.

서버 시작 시 모든 모델 파일을 메모리에 로딩.
파일 없으면 자동으로 MockPredictor로 대체.

백엔드 PR #9 적응도 점수 산출 스펙에 맞춰 포지션별 출력 키 정의.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.models.base import BasePredictor, MockPredictor, ModelPredictor

logger = logging.getLogger(__name__)


# 포지션별 예측 출력 키.
# 백엔드 PR #9의 적응도 산출 지표 + 리그 적응 효율성(pass_accuracy)에 맞춤.
POSITION_OUTPUT_KEYS = {
    "FW": ["goals", "shots", "dribbles", "key_passes", "pass_accuracy"],
    "MF": ["passes", "key_passes", "tackles", "pass_accuracy"],
    "DF": ["aerials_won", "blocked_shots", "pass_accuracy"],
    "GK": ["saves", "cleansheets", "pass_accuracy"],
}

LEAGUES = ["EPL", "LA", "SA", "BL", "L1"]
POSITIONS = ["FW", "MF", "DF", "GK"]

# Mock 값 범위
MOCK_VALUE_RANGES = {
    "goals": (0.05, 0.8),
    "shots": (0.5, 4.0),
    "dribbles": (0.3, 3.0),
    "passes": (20.0, 90.0),
    "key_passes": (0.3, 2.5),
    "tackles": (0.5, 4.0),
    "aerials_won": (0.5, 5.0),
    "blocked_shots": (0.1, 2.0),
    "saves": (1.0, 5.0),
    "cleansheets": (0.1, 0.5),
    "pass_accuracy": (0.6, 0.95),
    "market_value_eur": (1_000_000, 100_000_000),
}


class ModelRegistry:
    """모델을 디렉토리에서 자동으로 발견해서 로딩."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        # (league, position) -> predictor
        self.performance: dict[tuple[str, str], BasePredictor] = {}
        # position -> predictor (시장가치는 포지션별)
        self.market_value: dict[str, BasePredictor] = {}
        # position -> predictor (유사도는 포지션별 임베딩)
        self.similarity: dict[str, BasePredictor] = {}

        self._load_all()

    def _load_all(self):
        self._load_performance()
        self._load_market_value()
        self._load_similarity()
        n_real = sum(1 for p in self._all_predictors() if not p.is_mock)
        n_mock = sum(1 for p in self._all_predictors() if p.is_mock)
        logger.info(f"모델 로딩 완료: 실제={n_real}, mock={n_mock}")

    def _load_performance(self):
        """리그 5 x 포지션 4 = 20개 모델."""
        perf_dir = self.models_dir / "performance"
        for league in LEAGUES:
            for position in POSITIONS:
                output_keys = POSITION_OUTPUT_KEYS[position]
                pattern = f"{league}_{position}_v*.joblib"
                files = list(perf_dir.glob(pattern)) if perf_dir.exists() else []
                if files:
                    latest = sorted(files)[-1]
                    try:
                        predictor = ModelPredictor.load(latest, output_keys=output_keys)
                        logger.info(f"  [실제] performance: {league}/{position}")
                    except Exception as e:
                        logger.warning(f"  [실패] {latest.name}: {e}")
                        predictor = self._make_mock_performance(league, position, output_keys)
                else:
                    predictor = self._make_mock_performance(league, position, output_keys)
                    logger.info(f"  [mock]  performance: {league}/{position}")
                self.performance[(league, position)] = predictor

    def _make_mock_performance(self, league, position, output_keys):
        return MockPredictor(
            name=f"{league}_{position}_performance",
            output_keys=output_keys,
            value_ranges={k: MOCK_VALUE_RANGES.get(k, (0.0, 1.0)) for k in output_keys},
            seed_from_input="player_id",
        )

    def _load_market_value(self):
        """포지션 4개 모델."""
        mv_dir = self.models_dir / "market_value"
        for position in POSITIONS:
            pattern = f"market_value_{position}_v*.joblib"
            files = list(mv_dir.glob(pattern)) if mv_dir.exists() else []
            if files:
                latest = sorted(files)[-1]
                try:
                    predictor = ModelPredictor.load(latest, output_keys=["market_value_eur"])
                    logger.info(f"  [실제] market_value: {position}")
                except Exception as e:
                    logger.warning(f"  [실패] {latest.name}: {e}")
                    predictor = self._make_mock_market_value(position)
            else:
                predictor = self._make_mock_market_value(position)
                logger.info(f"  [mock]  market_value: {position}")
            self.market_value[position] = predictor

    def _make_mock_market_value(self, position):
        return MockPredictor(
            name=f"market_value_{position}",
            output_keys=["market_value_eur"],
            value_ranges={"market_value_eur": MOCK_VALUE_RANGES["market_value_eur"]},
            seed_from_input="player_id",
        )

    def _load_similarity(self):
        """포지션 4개 유사도 모델.

        실제로는 임베딩 + KNN 인덱스가 들어갈 예정.
        mock은 임의의 player_id 와 유사도를 반환.
        """
        sim_dir = self.models_dir / "similar"
        for position in POSITIONS:
            pattern = f"similarity_{position}_v*.joblib"
            files = list(sim_dir.glob(pattern)) if sim_dir.exists() else []
            if files:
                latest = sorted(files)[-1]
                try:
                    predictor = ModelPredictor.load(latest, output_keys=["similarity"])
                    logger.info(f"  [실제] similarity: {position}")
                except Exception as e:
                    logger.warning(f"  [실패] {latest.name}: {e}")
                    predictor = self._make_mock_similarity(position)
            else:
                predictor = self._make_mock_similarity(position)
                logger.info(f"  [mock]  similarity: {position}")
            self.similarity[position] = predictor

    def _make_mock_similarity(self, position):
        return MockPredictor(
            name=f"similarity_{position}",
            output_keys=["similarity"],
            value_ranges={"similarity": (0.5, 0.99)},
            seed_from_input="player_id",
        )

    # ===== 외부 접근 API =====

    def get_performance(self, league: str, position: str) -> BasePredictor:
        return self.performance[(league, position)]

    def get_market_value(self, position: str) -> BasePredictor:
        return self.market_value[position]

    def get_similarity(self, position: str) -> BasePredictor:
        return self.similarity[position]

    def _all_predictors(self):
        yield from self.performance.values()
        yield from self.market_value.values()
        yield from self.similarity.values()

    def status(self) -> dict:
        return {
            "performance": {
                f"{lg}/{pos}": {"version": p.version, "is_mock": p.is_mock}
                for (lg, pos), p in self.performance.items()
            },
            "market_value": {
                pos: {"version": p.version, "is_mock": p.is_mock}
                for pos, p in self.market_value.items()
            },
            "similarity": {
                pos: {"version": p.version, "is_mock": p.is_mock}
                for pos, p in self.similarity.items()
            },
        }
