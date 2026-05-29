"""POST /predict/similar-players 핸들러.

ai_pipeline.has_similar 일 때는 AI 팀 recommend_similar_players 로 cosine 추천,
없으면 기존 dummy 경로(로컬 개발)로 fallback.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

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

    def __init__(
        self,
        registry: ModelRegistry,
        feature_store: MockFeatureStore,
        ai_pipeline: Optional[Any] = None,
    ):
        self.registry = registry
        self.feature_store = feature_store
        self.ai_pipeline = ai_pipeline
        self._pool_path: Optional[str] = None

    def _ensure_candidate_pool(self) -> Optional[str]:
        """백엔드 DB 5대리그 선수로 후보 풀 CSV 를 1회 빌드/캐싱하고 경로 반환.

        풀을 못 만들면(Mock 등) None — 이 경우 AI 팀 기본 후보 풀로 fallback.
        """
        if self._pool_path and Path(self._pool_path).exists():
            return self._pool_path
        try:
            df = self.feature_store.get_big5_candidate_pool()
        except Exception:
            logger.exception("백엔드 후보 풀 빌드 실패 — 기본 풀로 fallback")
            return None
        if df is None or df.empty:
            return None
        path = os.path.join(tempfile.gettempdir(), "backend_candidate_pool.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        self._pool_path = path
        logger.info("백엔드 후보 풀 빌드 완료 (%d명, %s)", len(df), path)
        return path

    def handle(self, request: SimilarPlayersRequest) -> list[SimilarPlayersPrediction]:
        t0 = time.time()
        league = request.destination_league
        if self.ai_pipeline is not None and getattr(self.ai_pipeline, "has_similar", False):
            results = self._handle_real(request)
            mode = "real"
        else:
            results = self._handle_dummy(request)
            mode = "dummy"
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "similar_players: league=%s requested=%d latency_ms=%d mode=%s",
            league.value, len(request.player_ids), latency_ms, mode,
        )
        return results

    # ---------------- 실모델 경로 ----------------

    def _handle_real(self, request: SimilarPlayersRequest) -> list[SimilarPlayersPrediction]:
        rows: list[dict[str, Any]] = []
        valid_pids: list[int] = []
        by_pid: dict[int, list[SimilarPlayerEntry]] = {pid: [] for pid in request.player_ids}

        for pid in request.player_ids:
            try:
                if not self.feature_store.exists(pid):
                    raise ValueError(f"player_id={pid} not found")
                info = self.feature_store.get_player_info(pid)
                feats = self.feature_store.get_features(pid)
                if info is None or feats is None:
                    raise ValueError(f"player_id={pid} no data")
                merged = {**feats, **info, "player_id": pid}
                rows.append(merged)
                valid_pids.append(pid)
            except Exception:
                logger.exception("similar 입력 빌드 실패 player_id=%d", pid)

        if rows:
            try:
                pool_path = self._ensure_candidate_pool()
                results_per_row = self.ai_pipeline.predict_similar(
                    rows, request.destination_league, top_k=self.DEFAULT_TOP_K,
                    candidate_pool_path=pool_path,
                )
                # 백엔드 DB 후보 풀이면 player_id 가 곧 백엔드 id (별도 매핑 불필요)
                for pid, entries in zip(valid_pids, results_per_row):
                    by_pid[pid] = [
                        SimilarPlayerEntry(similar_player_id=sid, similarity_score=score)
                        for sid, score in entries
                    ]
            except Exception:
                logger.exception("AI similar_player 호출 실패")

        return [
            SimilarPlayersPrediction(player_id=pid, similar_players=by_pid[pid])
            for pid in request.player_ids
        ]

    # ---------------- dummy 경로 ----------------

    def _handle_dummy(self, request: SimilarPlayersRequest) -> list[SimilarPlayersPrediction]:
        results: list[SimilarPlayersPrediction] = []
        for pid in request.player_ids:
            try:
                results.append(self._predict_one_dummy(pid, request.destination_league))
            except Exception:
                logger.exception("similar players prediction failed for player_id=%d", pid)
                results.append(
                    SimilarPlayersPrediction(player_id=pid, similar_players=[])
                )
        return results

    def _predict_one_dummy(self, player_id: int, league: League) -> SimilarPlayersPrediction:
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
        return sorted(entries, key=lambda e: e.similarity_score, reverse=True)
