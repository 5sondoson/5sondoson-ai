"""백엔드(5sondoson-be) AiPredictionClient 가 호출하는 형태와 일치하는 요청/응답 스키마.

출처: src/main/java/com/osondoson/backend/admin/ai/dto/

특징:
- JSON 직렬화는 camelCase (백엔드 Java record 표기와 일치).
- pred_* 필드는 모두 Optional — 실패/미해당 시 None 으로 전달.
- 백엔드 DTO 의 typo 도 그대로 따른다(aeriels, cleensheets) — 시스템 전체 일관성을 위함.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from app.schemas.enums import League


class ApiModel(BaseModel):
    """모든 요청/응답이 상속하는 베이스.

    내부 코드는 snake_case, JSON 입출력은 camelCase 로 변환된다.
    """
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


# ============================================================
# 1. 퍼포먼스 예측: POST /predict/performance
# ============================================================

class PerformanceRequest(ApiModel):
    """백엔드 AiPerformanceRequest 와 동일."""
    player_ids: list[int]
    destination_league: League


class PerformancePrediction(ApiModel):
    """백엔드 AiPerformancePrediction 와 동일."""
    player_id: int
    pred_goals_total_per90: Optional[float] = None
    pred_shots_total_per90: Optional[float] = None
    pred_successful_dribbles_per90: Optional[float] = None
    pred_key_passes_per90: Optional[float] = None
    pred_passes_total_per90: Optional[float] = None
    pred_tackles_total_per90: Optional[float] = None
    pred_aeriels_won_per90: Optional[float] = None
    pred_blocked_shots_per90: Optional[float] = None
    pred_accurate_passes_pct: Optional[float] = None
    pred_cleensheets_total: Optional[float] = None


# ============================================================
# 2. 시장가치 예측: POST /predict/market-value
# ============================================================

class MarketValueRequest(ApiModel):
    """백엔드 AiMarketValueRequest 와 동일."""
    player_ids: list[int]
    destination_league: League


class MarketValuePrediction(ApiModel):
    """백엔드 AiMarketValuePrediction 와 동일.

    predicted_mv: 예측 시장가치(EUR, 정수).
    mv_change_rate: 현재 시장가치 대비 변화율.
    """
    player_id: int
    predicted_mv: Optional[int] = None
    mv_change_rate: Optional[float] = None


# ============================================================
# 3. 유사 선수 추천: POST /predict/similar-players
# ============================================================

class SimilarPlayersRequest(ApiModel):
    """백엔드 AiSimilarPlayersRequest 와 동일."""
    player_ids: list[int]
    destination_league: League


class SimilarPlayerEntry(ApiModel):
    """백엔드 SimilarPlayerEntry 와 동일."""
    similar_player_id: int
    similarity_score: float


class SimilarPlayersPrediction(ApiModel):
    """백엔드 AiSimilarPlayersPrediction 와 동일."""
    player_id: int
    similar_players: list[SimilarPlayerEntry]


# ============================================================
# 에러 응답
# ============================================================

class ErrorResponse(ApiModel):
    error_code: str
    message: str
    details: dict = {}
