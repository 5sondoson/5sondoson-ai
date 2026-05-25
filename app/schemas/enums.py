"""백엔드 enum 과 일치하는 League/Position + AI 모델 호출용 매핑.

백엔드 정의 출처:
- com.osondoson.backend.enums.league.League
- com.osondoson.backend.enums.position.Position
"""
from __future__ import annotations

from enum import Enum


class Position(str, Enum):
    """백엔드 Position 과 동일한 식별자(FW/MF/DF/GK).

    display_name 은 백엔드 enum 의 displayName 과 일치하며,
    AI 모델은 이 값을 소문자로 변환한 role_name (attacker 등)을 사용한다.
    """
    FW = "FW"
    MF = "MF"
    DF = "DF"
    GK = "GK"

    @property
    def display_name(self) -> str:
        return _POSITION_DISPLAY[self]

    @property
    def role_name(self) -> str:
        """AI 모델이 사용하는 역할명(소문자: attacker/midfielder/defender/goalkeeper)."""
        return self.display_name.lower()


_POSITION_DISPLAY: dict[Position, str] = {
    Position.FW: "Attacker",
    Position.MF: "Midfielder",
    Position.DF: "Defender",
    Position.GK: "Goalkeeper",
}


class League(str, Enum):
    """백엔드 League 의 value(.getValue()) 와 동일한 식별자.

    is_source = 비5대(우리 선수의 출신 리그)
    is_destination = 5대(이적 대상 리그)

    display_name 은 AI 모델의 transfer_path 문자열에 사용한다.
    """
    EREDIVISIE = "ERE"
    PRIMEIRA_LIGA = "PRL"
    PRO_LEAGUE = "BPL"

    PREMIER_LEAGUE = "EPL"
    LA_LIGA = "LA"
    BUNDESLIGA = "BL"
    SERIE_A = "SA"
    LIGUE_1 = "L1"

    @property
    def display_name(self) -> str:
        return _LEAGUE_DISPLAY[self]

    @property
    def is_source(self) -> bool:
        return self in _SOURCE_LEAGUES

    @property
    def is_destination(self) -> bool:
        return self not in _SOURCE_LEAGUES


_LEAGUE_DISPLAY: dict[League, str] = {
    League.EREDIVISIE: "Eredivisie",
    League.PRIMEIRA_LIGA: "Primeira Liga",
    League.PRO_LEAGUE: "Pro League",
    League.PREMIER_LEAGUE: "Premier League",
    League.LA_LIGA: "La Liga",
    League.BUNDESLIGA: "Bundesliga",
    League.SERIE_A: "Serie A",
    League.LIGUE_1: "Ligue 1",
}

_SOURCE_LEAGUES: set[League] = {
    League.EREDIVISIE,
    League.PRIMEIRA_LIGA,
    League.PRO_LEAGUE,
}
