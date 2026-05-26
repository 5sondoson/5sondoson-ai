"""POST /predict/market-value 핸들러.

백엔드 AiPredictionClient.fetchMarketValuePredictions 호출에 대응한다.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from app.features.store import MockFeatureStore
from app.models.registry import LEAGUES, ModelRegistry
from app.schemas.api import MarketValuePrediction, MarketValueRequest
from app.schemas.enums import League

logger = logging.getLogger(__name__)


def _mock_change_rate(player_id: int) -> float:
    """현재 가치 대비 변화율을 player_id 기반으로 결정적으로 생성.

    실제 모델 통합 전 mock 용. -0.2 ~ +0.5 범위.
    """
    h = int(hashlib.md5(f"mv_change_{player_id}".encode()).hexdigest(), 16)
    return round(-0.2 + (h % 700) / 1000, 4)


class MarketValueHandler:
    def __init__(
        self,
        registry: ModelRegistry,
        feature_store: MockFeatureStore,
        ai_pipeline=None,
    ):
        self.registry = registry
        self.feature_store = feature_store
        self.ai_pipeline = ai_pipeline

    def handle(self, request: MarketValueRequest) -> list[MarketValuePrediction]:
        t0 = time.time()
        league = request.destination_league
        results: list[MarketValuePrediction] = []

        for player_id in request.player_ids:
            try:
                results.append(self._predict_one(player_id, league))
            except Exception as e:
                logger.exception(f"market value prediction failed for player_id={player_id}")
                results.append(MarketValuePrediction(player_id=player_id))

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "market_value: league=%s requested=%d latency_ms=%d",
            league.value, len(request.player_ids), latency_ms,
        )
        return results

    def _predict_one(self, player_id: int, league: League) -> MarketValuePrediction:
        if not self.feature_store.exists(player_id):
            raise ValueError(f"player_id={player_id} not found")

        info = self.feature_store.get_player_info(player_id, season_id=0)
        features = self.feature_store.get_features(player_id, season_id=0)
        position = info["position"]

        predictor = self.registry.get_market_value(position)
        mv_input = {
            **features,
            "age": info["age"],
            "player_id": player_id,
            "target_league": league.value,
            "target_league_idx": LEAGUES.index(league.value) if league.value in LEAGUES else 0,
        }
        raw = predictor.predict(mv_input)
        predicted_mv: Optional[int] = int(raw["market_value_eur"]) if "market_value_eur" in raw else None
        return MarketValuePrediction(
            player_id=player_id,
            predicted_mv=predicted_mv,
            mv_change_rate=_mock_change_rate(player_id),
        )
