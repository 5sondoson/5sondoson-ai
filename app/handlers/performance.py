"""POST /predict/performance 핸들러.

ai_pipeline 이 주입돼 있으면 AI 팀의 Stage1+Stage2 실모델을 사용하고,
없으면 기존 dummy registry 경로(로컬 개발용)로 fallback 한다.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pandas as pd

from app.features.store import MockFeatureStore
from app.models.registry import ModelRegistry, POSITION_OUTPUT_KEYS
from app.schemas.api import PerformancePrediction, PerformanceRequest
from app.schemas.enums import League

logger = logging.getLogger(__name__)


# dummy 모델 (Mock 모드) 출력 키 -> 백엔드 응답 필드 매핑.
_DUMMY_OUTPUT_TO_FIELD: dict[str, str] = {
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

# AI 팀 Stage2 출력의 target_short_name -> 백엔드 응답 필드 매핑.
# (stage2_predictions DF 의 target 컬럼은 'delta_*' 형식이라 target_short_name 사용)
_AI_TARGET_TO_FIELD: dict[str, str] = {
    "goals": "pred_goals_total_per90",
    "shots": "pred_shots_total_per90",
    "successful_dribbles": "pred_successful_dribbles_per90",
    "key_passes": "pred_key_passes_per90",
    "passes": "pred_passes_total_per90",
    "tackles": "pred_tackles_total_per90",
    "aerials_won": "pred_aeriels_won_per90",
    "blocked_shots": "pred_blocked_shots_per90",
    "accurate_passes_%": "pred_accurate_passes_pct",
    "cleansheets": "pred_cleensheets_total",
}


class PerformanceHandler:
    def __init__(
        self,
        registry: ModelRegistry,
        feature_store: MockFeatureStore,
        ai_pipeline: Optional[Any] = None,
    ):
        self.registry = registry
        self.feature_store = feature_store
        self.ai_pipeline = ai_pipeline

    def handle(self, request: PerformanceRequest) -> list[PerformancePrediction]:
        t0 = time.time()
        league = request.destination_league
        if self.ai_pipeline is not None:
            results, cache_hit = self._handle_real(request)
        else:
            results = self._handle_dummy(request)
            cache_hit = 0
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "performance: league=%s requested=%d cache_hit=%d latency_ms=%d mode=%s",
            league.value, len(request.player_ids), cache_hit, latency_ms,
            "real" if self.ai_pipeline else "dummy",
        )
        return results

    # ---------------- 실모델 경로 ----------------

    def _handle_real(
        self, request: PerformanceRequest,
    ) -> tuple[list[PerformancePrediction], int]:
        by_pid: dict[int, dict[str, Any]] = {pid: {"player_id": pid} for pid in request.player_ids}

        # 1) 백엔드 캐시(player_performance_predictions) 조회 — destination_league 는 enum NAME 으로 비교
        cached = self.feature_store.get_cached_performance(
            request.player_ids, request.destination_league.name,
        )
        for pid, fields in cached.items():
            by_pid[pid].update({k: v for k, v in fields.items() if v is not None})

        # 2) miss 인 pid 만 Stage1+Stage2 실행
        miss_pids = [pid for pid in request.player_ids if pid not in cached]
        rows: list[dict[str, Any]] = []
        valid_pids: list[int] = []
        for pid in miss_pids:
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
                logger.exception("AI 입력 row 빌드 실패 player_id=%d", pid)

        if rows:
            try:
                stage2_df = self.ai_pipeline.predict(rows, request.destination_league)
                for _, pred_row in stage2_df.iterrows():
                    ridx = int(pred_row["row_index"])
                    if ridx >= len(valid_pids):
                        continue
                    pid = valid_pids[ridx]
                    field = _AI_TARGET_TO_FIELD.get(pred_row["target_short_name"])
                    if field is None:
                        continue
                    value = pred_row["final_after_pred"]
                    if pd.isna(value):
                        continue
                    by_pid[pid][field] = float(value)
            except Exception:
                logger.exception("AI pipeline 호출 실패 — 모든 결과 null 로 반환")

        return (
            [PerformancePrediction(**by_pid[pid]) for pid in request.player_ids],
            len(cached),
        )

    # ---------------- dummy 경로 (Mock 모드) ----------------

    def _handle_dummy(self, request: PerformanceRequest) -> list[PerformancePrediction]:
        results: list[PerformancePrediction] = []
        for player_id in request.player_ids:
            try:
                results.append(self._predict_one_dummy(player_id, request.destination_league))
            except Exception:
                logger.exception("performance prediction failed for player_id=%d", player_id)
                results.append(PerformancePrediction(player_id=player_id))
        return results

    def _predict_one_dummy(self, player_id: int, league: League) -> PerformancePrediction:
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
            field = _DUMMY_OUTPUT_TO_FIELD.get(key)
            if field and key in raw:
                kwargs[field] = float(raw[key])
        return PerformancePrediction(**kwargs)
