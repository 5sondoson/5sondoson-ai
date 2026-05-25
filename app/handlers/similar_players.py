"""POST /predict/similar-players 핸들러.

백엔드 AiPredictionClient.fetchSimilarPlayersPredictions 호출에 대응한다.
"""
from __future__ import annotations

import hashlib
import logging
import time

from app.features.store import MockFeatureStore
from app.models.registry import ModelRegistry
from app.schemas.api import (
    SimilarPlayerEntry,
    SimilarPlayersPrediction,
    SimilarPlayersRequest,
)
from app.schemas.enums import League

logger = logging.getLogger(__name__)


class SimilarPlayersHandler:
    DEFAULT_TOP_K = 5

    def __init__(self, registry: ModelRegistry, feature_store: MockFeatureStore):
        self.registry = registry
        self.feature_store = feature_store

    def handle(self, request: SimilarPlayersRequest) -> list[SimilarPlayersPrediction]:
        t0 = time.time()
        league = request.destination_league
        results: list[SimilarPlayersPrediction] = []

        for player_id in request.player_ids:
            try:
                results.append(self._predict_one(player_id, league))
            except Exception as e:
                logger.exception(f"similar players prediction failed for player_id={player_id}")
                results.append(
                    SimilarPlayersPrediction(player_id=player_id, similar_players=[])
                )

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "similar_players: league=%s requested=%d latency_ms=%d",
            league.value, len(request.player_ids), latency_ms,
        )
        return results

    def _predict_one(self, player_id: int, league: League) -> SimilarPlayersPrediction:
        if not self.feature_store.exists(player_id):
            raise ValueError(f"player_id={player_id} not found")

        info = self.feature_store.get_player_info(player_id, season_id=0)
        position = info["position"]
        predictor = self.registry.get_similarity(position)

        candidates = self._mock_candidates(
            player_id, league.value, position, self.DEFAULT_TOP_K, predictor,
        )
        return SimilarPlayersPrediction(
            player_id=player_id,
            similar_players=candidates,
        )

    def _mock_candidates(self, player_id, league, position, top_k, predictor):
        """결정적으로 top_k 개 후보 생성. 실제 모델은 KNN 인덱스 조회."""
        entries: list[SimilarPlayerEntry] = []
        for i in range(top_k):
            seed_str = f"{player_id}_{league}_{position}_{i}"
            h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
            similar_id = 10000 + (h % 90000)
            base = predictor.predict({"player_id": player_id, "candidate_idx": i})
            sim_score = base.get("similarity", 0.7) - (i * 0.03)
            sim_score = max(0.1, min(0.99, sim_score))
            entries.append(SimilarPlayerEntry(
                similar_player_id=similar_id,
                similarity_score=round(sim_score, 4),
            ))
        # 가장 유사한 선수가 먼저 오도록 내림차순 정렬
        return sorted(entries, key=lambda e: e.similarity_score, reverse=True)
