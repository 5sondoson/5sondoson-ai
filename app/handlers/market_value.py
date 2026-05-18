"""POST /predictions/market-value 핸들러."""
from __future__ import annotations

import logging
import time

from app.features.store import MockFeatureStore
from app.models.registry import ModelRegistry
from app.schemas.api import (
    FailedPlayer,
    MarketValueRequest,
    MarketValueResponse,
    PlayerMarketValuePrediction,
    Position,
    ResponseMeta,
)

logger = logging.getLogger(__name__)


class MarketValueHandler:
    def __init__(self, registry: ModelRegistry, feature_store: MockFeatureStore,
                 performance_handler=None):
        self.registry = registry
        self.feature_store = feature_store
        # performance_hints가 없으면 내부적으로 퍼포먼스 추론도 같이 수행
        self.performance_handler = performance_handler

    def handle(self, request: MarketValueRequest) -> MarketValueResponse:
        t0 = time.time()
        predictions: list[PlayerMarketValuePrediction] = []
        failed: list[FailedPlayer] = []
        used_versions: dict[str, str] = {}
        any_mock = False

        for player_input in request.players:
            try:
                pred, versions, is_mock = self._predict_one(
                    player_input.player_id,
                    player_input.season_id,
                    request.target_leagues,
                    request.performance_hints,
                )
                predictions.append(pred)
                used_versions.update(versions)
                any_mock = any_mock or is_mock
            except Exception as e:
                logger.exception(f"market value prediction failed for player_id={player_input.player_id}")
                failed.append(FailedPlayer(
                    player_id=player_input.player_id,
                    reason=str(e),
                ))

        latency_ms = int((time.time() - t0) * 1000)
        return MarketValueResponse(
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

    def _predict_one(self, player_id, season_id, target_leagues, performance_hints):
        info = self.feature_store.get_player_info(player_id, season_id)
        if not self.feature_store.exists(player_id):
            raise ValueError(f"player_id={player_id} not found")

        features = self.feature_store.get_features(player_id, season_id)
        position = info["position"]
        age = info["age"]

        by_league = {}
        versions = {}
        any_mock = False

        predictor = self.registry.get_market_value(position)
        versions[f"market_value:{position}"] = predictor.version

        for league in target_leagues:
            # 퍼포먼스 hint가 있으면 사용, 없으면 features만
            if performance_hints and player_id in performance_hints:
                perf_hint = performance_hints[player_id].get(league.value, {})
            else:
                perf_hint = {}

            mv_input = {
                **features,
                "age": age,
                "player_id": player_id,
                "target_league": league.value,
                **perf_hint,
            }
            result = predictor.predict(mv_input)
            by_league[league] = float(result.get("market_value_eur", 0.0))
            any_mock = any_mock or predictor.is_mock

        prediction = PlayerMarketValuePrediction(
            player_id=player_id,
            position=Position(position),
            by_league=by_league,
        )
        return prediction, versions, any_mock
