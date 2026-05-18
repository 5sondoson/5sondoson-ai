"""API 요청/응답 스키마.

이 파일이 AI 서버와 백엔드 사이의 계약(contract).
백엔드 PR #9 스펙에 맞춰 설계.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 공통 enum / 기본 타입
# ============================================================

class Position(str, Enum):
    FW = "FW"
    MF = "MF"
    DF = "DF"
    GK = "GK"


class TargetLeague(str, Enum):
    PREMIER_LEAGUE = "premier_league"
    LA_LIGA = "la_liga"
    SERIE_A = "serie_a"
    BUNDESLIGA = "bundesliga"
    LIGUE_1 = "ligue_1"


class PlayerInput(BaseModel):
    """배치 요청 안에 들어가는 선수 1명의 정보."""
    player_id: int = Field(..., description="선수 ID")
    season_id: int = Field(..., description="기준 시즌 ID")


class FailedPlayer(BaseModel):
    """처리 실패한 선수 정보."""
    player_id: int
    reason: str


# ============================================================
# 1. 퍼포먼스 예측: POST /predictions/performance
# ============================================================

class PerformanceRequest(BaseModel):
    players: list[PlayerInput] = Field(..., max_length=100,
                                       description="선수 리스트 (보통 50명)")
    target_leagues: list[TargetLeague] = Field(
        default_factory=lambda: list(TargetLeague),
        description="예측할 대상 리그. 비우면 5대리그 전체",
    )


class PerformanceStats(BaseModel):
    """포지션별 예측 스탯. 키는 포지션마다 다름.

    FW: goals, shots, dribbles, pass_accuracy, key_passes
    MF: passes, key_passes, tackles, pass_accuracy
    DF: aerials_won, blocked_shots, pass_accuracy
    GK: saves, cleansheets, pass_accuracy
    """
    stats: dict[str, float] = Field(..., description="포지션별 예측 스탯")


class PlayerPerformancePrediction(BaseModel):
    player_id: int
    position: Position
    by_league: dict[TargetLeague, PerformanceStats] = Field(
        ..., description="리그별 예측 퍼포먼스"
    )


class PerformanceResponse(BaseModel):
    predictions: list[PlayerPerformancePrediction]
    failed: list[FailedPlayer] = Field(default_factory=list)
    meta: "ResponseMeta"


# ============================================================
# 2. 시장가치 예측: POST /predictions/market-value
# ============================================================

class MarketValueRequest(BaseModel):
    players: list[PlayerInput] = Field(..., max_length=100)
    target_leagues: list[TargetLeague] = Field(
        default_factory=lambda: list(TargetLeague),
    )
    # 백엔드가 퍼포먼스 예측 결과를 알고 있으면 같이 전달 (선택).
    # 없으면 AI 서버가 내부적으로 다시 추론.
    performance_hints: Optional[dict[int, dict]] = Field(
        default=None,
        description="player_id -> 퍼포먼스 예측 결과 (있으면 재사용)"
    )


class PlayerMarketValuePrediction(BaseModel):
    player_id: int
    position: Position
    by_league: dict[TargetLeague, float] = Field(
        ..., description="리그별 예측 시장가치 (EUR)"
    )


class MarketValueResponse(BaseModel):
    predictions: list[PlayerMarketValuePrediction]
    failed: list[FailedPlayer] = Field(default_factory=list)
    meta: "ResponseMeta"


# ============================================================
# 3. 유사 선수 추천: POST /predictions/similar-players
# ============================================================

class SimilarPlayersRequest(BaseModel):
    players: list[PlayerInput] = Field(..., max_length=100)
    target_leagues: list[TargetLeague] = Field(
        default_factory=lambda: list(TargetLeague),
    )
    top_k: int = Field(default=5, ge=1, le=20,
                       description="리그당 추천할 유사 선수 수")


class SimilarPlayerEntry(BaseModel):
    similar_player_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class PlayerSimilarPrediction(BaseModel):
    player_id: int
    position: Position
    by_league: dict[TargetLeague, list[SimilarPlayerEntry]]


class SimilarPlayersResponse(BaseModel):
    predictions: list[PlayerSimilarPrediction]
    failed: list[FailedPlayer] = Field(default_factory=list)
    meta: "ResponseMeta"


# ============================================================
# 공통 메타
# ============================================================

class ResponseMeta(BaseModel):
    requested: int = Field(..., description="요청된 선수 수")
    succeeded: int = Field(..., description="성공한 선수 수")
    failed_count: int = Field(..., description="실패한 선수 수")
    latency_ms: int = Field(..., description="처리 시간 (밀리초)")
    model_versions: dict[str, str] = Field(default_factory=dict)
    is_mock: bool = Field(..., description="mock 모델이 하나라도 사용됐는지")


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict = Field(default_factory=dict)


# Forward reference 해소
PerformanceResponse.model_rebuild()
MarketValueResponse.model_rebuild()
SimilarPlayersResponse.model_rebuild()
