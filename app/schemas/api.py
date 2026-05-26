"""백엔드(5sondoson-be) AiPredictionClient 가 호출하는 형태와 일치하는 요청/응답 스키마.

출처: src/main/java/com/osondoson/backend/admin/ai/dto/

특징:
- JSON 직렬화는 camelCase (백엔드 Java record 표기와 일치).
- pred_* 필드는 모두 Optional — 실패/미해당 시 None 으로 전달.
- 백엔드 DTO 의 typo 도 그대로 따른다(aeriels, cleensheets) — 시스템 전체 일관성을 위함.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
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
    """이적 후 퍼포먼스 예측 요청.

    백엔드 어드민 배치가 선수를 50명 청크로 잘라 호출한다.
    """
    player_ids: list[int] = Field(
        description="예측 대상 player_id 목록. 한 요청당 최대 50개 권장.",
        examples=[[1, 2, 3, 4, 5]],
    )
    destination_league: League = Field(
        description="이적 목적지 리그(5대 리그 중 하나). 값은 백엔드 League enum 의 코드(EPL/LA/BL/SA/L1).",
        examples=["EPL"],
    )


class PerformancePrediction(ApiModel):
    """선수 1명의 이적 후 퍼포먼스 예측 결과.

    모든 pred_* 필드는 per90 정규화 값(90분당 통계)이고 선수 포지션과 무관한 컬럼은 null.
    필드명의 `aeriels`(공중볼)/`cleensheets`(클린시트) 는 백엔드 DTO 와 동일하게 의도된 표기.
    """
    player_id: int = Field(description="선수 ID(백엔드 players.id).", examples=[1])
    pred_goals_total_per90: Optional[float] = Field(
        default=None, description="90분당 예측 골 수.", examples=[0.45],
    )
    pred_shots_total_per90: Optional[float] = Field(
        default=None, description="90분당 예측 슈팅 수.", examples=[1.21],
    )
    pred_successful_dribbles_per90: Optional[float] = Field(
        default=None, description="90분당 예측 성공 드리블 수.", examples=[2.76],
    )
    pred_key_passes_per90: Optional[float] = Field(
        default=None, description="90분당 예측 키패스 수.", examples=[2.42],
    )
    pred_passes_total_per90: Optional[float] = Field(
        default=None, description="90분당 예측 패스 수.", examples=[45.7],
    )
    pred_tackles_total_per90: Optional[float] = Field(
        default=None, description="90분당 예측 태클 수.", examples=[1.68],
    )
    pred_aeriels_won_per90: Optional[float] = Field(
        default=None,
        description="90분당 예측 공중볼 경합 승리(필드명 `aeriels` 는 백엔드 DTO 표기 그대로).",
        examples=[1.57],
    )
    pred_blocked_shots_per90: Optional[float] = Field(
        default=None, description="90분당 예측 슈팅 블록 수.", examples=[0.39],
    )
    pred_accurate_passes_pct: Optional[float] = Field(
        default=None, description="예측 패스 정확도(%, 0~100).", examples=[81.06],
    )
    pred_cleensheets_total: Optional[float] = Field(
        default=None,
        description="시즌 누적 예측 클린시트 수(필드명 `cleensheets` 는 백엔드 DTO 표기 그대로).",
        examples=[7.7],
    )


# ============================================================
# 2. 시장가치 예측: POST /predict/market-value
# ============================================================

class MarketValueRequest(ApiModel):
    """이적 후 시장가치 예측 요청."""
    player_ids: list[int] = Field(
        description="예측 대상 player_id 목록.",
        examples=[[1, 2, 3, 4, 5]],
    )
    destination_league: League = Field(
        description="이적 목적지 리그.",
        examples=["EPL"],
    )


class MarketValuePrediction(ApiModel):
    """선수 1명의 이적 후 시장가치 예측 결과."""
    player_id: int = Field(description="선수 ID.", examples=[1])
    predicted_mv: Optional[int] = Field(
        default=None,
        description="이적 후 예측 시장가치(EUR, 정수). 모델은 log(EUR) 출력 → exp 변환된 값.",
        examples=[6706048],
    )
    mv_change_rate: Optional[float] = Field(
        default=None,
        description="현재 시장가치 대비 변화율((predicted - current) / current). 현재 시장가치가 없으면 null.",
        examples=[3.19],
    )


# ============================================================
# 3. 유사 선수 추천: POST /predict/similar-players
# ============================================================

class SimilarPlayersRequest(ApiModel):
    """유사 선수 추천 요청."""
    player_ids: list[int] = Field(
        description="추천 대상 player_id 목록.",
        examples=[[57]],
    )
    destination_league: League = Field(
        description="이적 목적지 리그(추천 후보 풀은 이 리그에 속한 선수들로 한정).",
        examples=["EPL"],
    )


class SimilarPlayerEntry(ApiModel):
    """유사 선수 후보 1명."""
    similar_player_id: int = Field(description="후보 선수의 player_id.", examples=[1116])
    similarity_score: float = Field(
        description="cosine 유사도(-1.0 ~ 1.0). 보통 0.9 이상이 강한 유사로 본다.",
        examples=[0.98],
    )


class SimilarPlayersPrediction(ApiModel):
    """선수 1명의 유사 선수 top-K 결과(K=5)."""
    player_id: int = Field(description="요청한 원본 선수 ID.", examples=[57])
    similar_players: list[SimilarPlayerEntry] = Field(
        description="유사도 내림차순으로 정렬된 상위 5명. 실패한 경우 빈 리스트.",
    )


# ============================================================
# 에러 응답
# ============================================================

class ErrorResponse(ApiModel):
    """전역 에러 핸들러가 5xx 응답으로 반환하는 형식."""
    error_code: str = Field(description="짧은 에러 코드.", examples=["INTERNAL_ERROR"])
    message: str = Field(description="사람이 읽을 수 있는 에러 메시지.")
    details: dict = Field(default_factory=dict, description="추가 디버깅 정보(있을 때만).")
