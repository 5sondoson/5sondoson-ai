"""POST /predict/market-value 핸들러.

ai_pipeline 이 있고 market_value 모델까지 로드돼 있으면 실모델 경로로,
없으면 기존 dummy 경로(로컬 개발)로 fallback.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Optional

from app.features.store import MockFeatureStore
from app.models.registry import LEAGUES, ModelRegistry
from app.schemas.api import MarketValuePrediction, MarketValueRequest
from app.schemas.enums import League

# 캐시 테이블 컬럼명 → AI 모델의 target_short_name 매핑 (Stage2 final_after_pred 와 동일한 의미)
_CACHE_FIELD_TO_SHORT: dict[str, str] = {
    "pred_goals_total_per90": "goals",
    "pred_shots_total_per90": "shots",
    "pred_successful_dribbles_per90": "successful_dribbles",
    "pred_key_passes_per90": "key_passes",
    "pred_passes_total_per90": "passes",
    "pred_tackles_total_per90": "tackles",
    "pred_aeriels_won_per90": "aerials_won",
    "pred_blocked_shots_per90": "blocked_shots",
    "pred_accurate_passes_pct": "accurate_passes_%",
    "pred_cleansheets_total": "cleansheets",
}

logger = logging.getLogger(__name__)


def _mock_change_rate(player_id: int) -> float:
    """현 가치 대비 변화율 mock — player_id 기반 결정적."""
    h = int(hashlib.md5(f"mv_change_{player_id}".encode()).hexdigest(), 16)
    return round(-0.2 + (h % 700) / 1000, 4)


def _change_rate(predicted: Optional[int], current: Optional[int]) -> Optional[float]:
    if predicted is None or not current:
        return None
    return round((predicted - current) / current, 4)


class MarketValueHandler:
    def __init__(
        self,
        registry: ModelRegistry,
        feature_store: MockFeatureStore,
        ai_pipeline: Optional[Any] = None,
    ):
        self.registry = registry
        self.feature_store = feature_store
        self.ai_pipeline = ai_pipeline

    def handle(self, request: MarketValueRequest) -> list[MarketValuePrediction]:
        t0 = time.time()
        league = request.destination_league
        if self.ai_pipeline is not None and getattr(self.ai_pipeline, "has_market_value", False):
            results, cache_hit = self._handle_real(request)
            mode = "real"
        else:
            results = self._handle_dummy(request)
            cache_hit = 0
            mode = "dummy"
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "market_value: league=%s requested=%d cache_hit=%d latency_ms=%d mode=%s",
            league.value, len(request.player_ids), cache_hit, latency_ms, mode,
        )
        return results

    # ---------------- 실모델 경로 ----------------

    def _handle_real(
        self, request: MarketValueRequest,
    ) -> tuple[list[MarketValuePrediction], int]:
        rows: list[dict[str, Any]] = []
        valid_pids: list[int] = []
        current_mv: dict[int, Optional[int]] = {}
        by_pid: dict[int, dict[str, Any]] = {pid: {"player_id": pid} for pid in request.player_ids}

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
                # 현 가치는 PlayerSeasonRecord.market_value 우선, 없으면 None
                cmv = feats.get("market_value") or feats.get("current_market_value")
                current_mv[pid] = int(cmv) if cmv else None
            except Exception:
                logger.exception("market_value 입력 빌드 실패 player_id=%d", pid)

        # performance 캐시(player_performance_predictions) 를 stage2 final_after_pred 대용으로 사용
        cached_perf = self.feature_store.get_cached_performance(
            valid_pids, request.destination_league.name,
        )
        cached_by_pid: dict[int, dict[str, float]] = {}
        for pid, fields in cached_perf.items():
            mapped: dict[str, float] = {}
            for cache_field, value in fields.items():
                if value is None:
                    continue
                short_name = _CACHE_FIELD_TO_SHORT.get(cache_field)
                if short_name:
                    mapped[short_name] = value
            if mapped:
                cached_by_pid[pid] = mapped

        if rows:
            try:
                preds = self.ai_pipeline.predict_market_value(
                    rows, request.destination_league, cached_by_pid=cached_by_pid,
                )
                for i, pid in enumerate(valid_pids):
                    predicted = preds[i] if i < len(preds) else None
                    by_pid[pid]["predicted_mv"] = predicted
                    by_pid[pid]["mv_change_rate"] = _change_rate(predicted, current_mv.get(pid))
            except Exception:
                logger.exception("AI market_value 호출 실패")

        return (
            [MarketValuePrediction(**by_pid[pid]) for pid in request.player_ids],
            len(cached_by_pid),
        )

    # ---------------- dummy 경로 ----------------

    def _handle_dummy(self, request: MarketValueRequest) -> list[MarketValuePrediction]:
        results: list[MarketValuePrediction] = []
        for pid in request.player_ids:
            try:
                results.append(self._predict_one_dummy(pid, request.destination_league))
            except Exception:
                logger.exception("market value prediction failed for player_id=%d", pid)
                results.append(MarketValuePrediction(player_id=pid))
        return results

    def _predict_one_dummy(self, player_id: int, league: League) -> MarketValuePrediction:
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
        predicted_mv = int(raw["market_value_eur"]) if "market_value_eur" in raw else None
        return MarketValuePrediction(
            player_id=player_id,
            predicted_mv=predicted_mv,
            mv_change_rate=_mock_change_rate(player_id),
        )
