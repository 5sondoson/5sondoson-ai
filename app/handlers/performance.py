"""POST /predictions/performance 핸들러."""
from __future__ import annotations

import logging
import time

from app.features.store import MockFeatureStore
from app.models.registry import ModelRegistry, POSITION_OUTPUT_KEYS
from app.schemas.api import (
    FailedPlayer,
    PerformanceRequest,
    PerformanceResponse,
    PerformanceStats,
    PlayerPerformancePrediction,
    Position,
    ResponseMeta,
)

logger = logging.getLogger(__name__)


class PerformanceHandler:
    def __init__(self, registry: ModelRegistry, feature_store: MockFeatureStore):
        self.registry = registry
        self.feature_store = feature_store

    def handle(self, request: PerformanceRequest) -> PerformanceResponse:
        t0 = time.time()
        predictions: list[PlayerPerformancePrediction] = []
        failed: list[FailedPlayer] = []
        used_versions: dict[str, str] = {}
        any_mock = False

        for player_input in request.players:
            try:
                pred, versions, is_mock = self._predict_one(
                    player_input.player_id,
                    player_input.season_id,
                    request.target_leagues,
                )
                predictions.append(pred)
                used_versions.update(versions)
                any_mock = any_mock or is_mock
            except Exception as e:
                logger.exception(f"performance prediction failed for player_id={player_input.player_id}")
                failed.append(FailedPlayer(
                    player_id=player_input.player_id,
                    reason=str(e),
                ))

        latency_ms = int((time.time() - t0) * 1000)
        return PerformanceResponse(
            predictions=predictions,
            failed=failed,
            meta=ResponseMeta(
                requested=len(request.players),
                succeeded=len(predictions),
                failed_count=len(failed),
                latency_ms=latency_ms,
                model_versions=used_versions,
                is_mock=any_mock,
            ),
        )

    def _predict_one(self, player_id, season_id, target_leagues):
        info = self.feature_store.get_player_info(player_id, season_id)
        if not self.feature_store.exists(player_id):
            raise ValueError(f"player_id={player_id} not found")

        features = self.feature_store.get_features(player_id, season_id)
        position = info["position"]
        output_keys = POSITION_OUTPUT_KEYS[position]

        by_league = {}
        versions = {}
        any_mock = False

        for league in target_leagues:
            predictor = self.registry.get_performance(league.value, position)
            raw_result = predictor.predict(features)
            # output_keys 만 사용 (안전 장치)
            stats = {k: raw_result.get(k, 0.0) for k in output_keys}
            by_league[league] = PerformanceStats(stats=stats)
            versions[f"performance:{league.value}:{position}"] = predictor.version
            any_mock = any_mock or predictor.is_mock

        prediction = PlayerPerformancePrediction(
            player_id=player_id,
            position=Position(position),
            by_league=by_league,
        )
        return prediction, versions, any_mock
