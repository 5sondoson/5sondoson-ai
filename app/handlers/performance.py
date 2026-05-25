"""POST /predict/performance 핸들러.

백엔드 AiPredictionClient.fetchPerformancePredictions 호출에 대응한다.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from app.features.store import MockFeatureStore
from app.models.registry import ModelRegistry, POSITION_OUTPUT_KEYS
from app.schemas.api import PerformancePrediction, PerformanceRequest
from app.schemas.enums import League

logger = logging.getLogger(__name__)


# 레지스트리 출력 키 -> 백엔드 응답 필드 매핑.
# 백엔드 DTO 표기를 따라 typo(aeriels, cleensheets) 그대로 사용한다.
_OUTPUT_KEY_TO_FIELD: dict[str, str] = {
    "goals": "pred_goals_total_per90",
    "shots": "pred_shots_total_per90",
    "dribbles": "pred_successful_dribbles_per90",
    "key_passes": "pred_key_passes_per90",
    "passes": "pred_passes_total_per90",
    "tackles": "pred_tackles_total_per90",
    "aerials_won": "pred_aeriels_won_per90",
    "blocked_shots": "pred_blocked_shots_per90",
    "pass_accuracy": "pred_accurate_passes_pct",
    "cleansheets": "pred_cleensheets_total",
}


class PerformanceHandler:
    def __init__(self, registry: ModelRegistry, feature_store: MockFeatureStore):
        self.registry = registry
        self.feature_store = feature_store

    def handle(self, request: PerformanceRequest) -> list[PerformancePrediction]:
        t0 = time.time()
        league = request.destination_league
        results: list[PerformancePrediction] = []

        for player_id in request.player_ids:
            try:
                results.append(self._predict_one(player_id, league))
            except Exception as e:
                logger.exception(f"performance prediction failed for player_id={player_id}")
                # 실패도 1:1 대응 유지 (모든 pred_* 가 None)
                results.append(PerformancePrediction(player_id=player_id))

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "performance: league=%s requested=%d latency_ms=%d",
            league.value, len(request.player_ids), latency_ms,
        )
        return results

    def _predict_one(self, player_id: int, league: League) -> PerformancePrediction:
        if not self.feature_store.exists(player_id):
            raise ValueError(f"player_id={player_id} not found")

        info = self.feature_store.get_player_info(player_id, season_id=0)
        features = self.feature_store.get_features(player_id, season_id=0)
        position = info["position"]
        output_keys = POSITION_OUTPUT_KEYS[position]

        predictor = self.registry.get_performance(league.value, position)
        raw = predictor.predict(features)

        kwargs: dict[str, Optional[float]] = {"player_id": player_id}
        for key in output_keys:
            field = _OUTPUT_KEY_TO_FIELD.get(key)
            if field and key in raw:
                kwargs[field] = float(raw[key])
        return PerformancePrediction(**kwargs)
