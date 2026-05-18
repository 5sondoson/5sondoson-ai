"""POST /predictions/similar-players 핸들러."""
from __future__ import annotations

import hashlib
import logging
import time

from app.features.store import MockFeatureStore
from app.models.registry import ModelRegistry
from app.schemas.api import (
    FailedPlayer,
    PlayerSimilarPrediction,
    Position,
    ResponseMeta,
    SimilarPlayerEntry,
    SimilarPlayersRequest,
    SimilarPlayersResponse,
)

logger = logging.getLogger(__name__)


class SimilarPlayersHandler:
    def __init__(self, registry: ModelRegistry, feature_store: MockFeatureStore):
        self.registry = registry
        self.feature_store = feature_store

    def handle(self, request: SimilarPlayersRequest) -> SimilarPlayersResponse:
        t0 = time.time()
        predictions: list[PlayerSimilarPrediction] = []
        failed: list[FailedPlayer] = []
        used_versions: dict[str, str] = {}
        any_mock = False

        for player_input in request.players:
            try:
                pred, versions, is_mock = self._predict_one(
                    player_input.player_id,
                    player_input.season_id,
                    request.target_leagues,
                    request.top_k,
                )
                predictions.append(pred)
                used_versions.update(versions)
                any_mock = any_mock or is_mock
            except Exception as e:
                logger.exception(f"similar players prediction failed for player_id={player_input.player_id}")
                failed.append(FailedPlayer(
                    player_id=player_input.player_id,
                    reason=str(e),
                ))

        latency_ms = int((time.time() - t0) * 1000)
        return SimilarPlayersResponse(
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

    def _predict_one(self, player_id, season_id, target_leagues, top_k):
        info = self.feature_store.get_player_info(player_id, season_id)
        if not self.feature_store.exists(player_id):
            raise ValueError(f"player_id={player_id} not found")

        position = info["position"]
        predictor = self.registry.get_similarity(position)
        versions = {f"similarity:{position}": predictor.version}
        any_mock = predictor.is_mock

        by_league = {}
        for league in target_leagues:
            # mock: player_id + league + position 기반 결정적 후보 생성
            candidates = self._mock_candidates(player_id, league.value, position, top_k, predictor)
            by_league[league] = candidates

        prediction = PlayerSimilarPrediction(
            player_id=player_id,
            position=Position(position),
            by_league=by_league,
        )
        return prediction, versions, any_mock

    def _mock_candidates(self, player_id, league, position, top_k, predictor):
        """결정적으로 top_k개 후보 생성. 실제 모델은 KNN 인덱스 조회."""
        entries = []
        for i in range(top_k):
            seed_str = f"{player_id}_{league}_{position}_{i}"
            h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
            similar_id = 10000 + (h % 90000)
            # similarity는 i가 클수록 약간 낮아지게
            base = predictor.predict({"player_id": player_id, "candidate_idx": i})
            sim_score = base.get("similarity", 0.7) - (i * 0.03)
            sim_score = max(0.1, min(0.99, sim_score))
            entries.append(SimilarPlayerEntry(
                similar_player_id=similar_id,
                similarity_score=round(sim_score, 4),
            ))
        return entries
