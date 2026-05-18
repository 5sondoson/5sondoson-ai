"""피처 스토어.

선수 ID로부터 피처(통계 데이터) + 메타 정보(포지션, 나이 등)를 조회.

지금은 mock. 나중에 백엔드 DB 조회 또는 별도 캐시로 교체.
"""
from __future__ import annotations

import hashlib


POSITION_BY_HASH = ["FW", "MF", "DF", "GK"]


class MockFeatureStore:
    """player_id 기반 결정적 mock.

    같은 player_id -> 항상 같은 피처/포지션.
    """

    def get_player_info(self, player_id: int, season_id: int) -> dict:
        """선수의 메타 정보."""
        # player_id 해시로 포지션 결정 (재현 가능)
        h = int(hashlib.md5(str(player_id).encode()).hexdigest(), 16)
        position = POSITION_BY_HASH[h % 4]
        age = 18 + (h % 18)  # 18 ~ 35
        return {
            "player_id": player_id,
            "name": f"Player_{player_id}",
            "position": position,
            "age": age,
            "current_league": "primeira_liga",
            "height": 170 + (h % 25),
            "weight": 65 + (h % 25),
        }

    def get_features(self, player_id: int, season_id: int) -> dict:
        """모델 입력으로 사용할 피처 dict."""
        h = int(hashlib.md5(f"{player_id}_{season_id}".encode()).hexdigest(), 16)
        # 적당히 다양한 값 생성
        return {
            "player_id": player_id,
            "season_id": season_id,
            "stat_minutes_played_total": 1000 + (h % 2500),
            "stat_goals_total": h % 20,
            "stat_shots_total": h % 80,
            "stat_assists_total": h % 15,
            "stat_passes_total": 200 + (h % 1500),
            "stat_key_passes_total": h % 30,
            "stat_tackles_total": h % 60,
            "stat_aerials_won_total": h % 40,
            "stat_clearances_total": h % 50,
            "stat_blocked_shots_total": h % 25,
            "stat_saves_total": h % 100,
            "stat_cleansheets_total": h % 15,
            "stat_accurate_passes_pct": 0.6 + (h % 35) / 100,
            "age": 18 + (h % 18),
            "height": 170 + (h % 25),
            "weight": 65 + (h % 25),
        }

    def exists(self, player_id: int) -> bool:
        """선수가 DB에 있는지. mock에서는 음수만 없는 걸로 처리."""
        return player_id > 0
