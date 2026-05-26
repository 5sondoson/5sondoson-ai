"""피처 스토어.

선수 ID 로 피처(통계 데이터) + 메타 정보(포지션, 나이 등)를 조회.

- MockFeatureStore: 로컬 개발용 결정적 mock
- DbFeatureStore:   운영용 백엔드 RDS(MySQL) read-only 조회 (최신 시즌 자동 선택)

main.py 의 lifespan 에서 DB_HOST 환경변수 유무로 스토어를 분기한다.
"""
from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


POSITION_BY_HASH = ["FW", "MF", "DF", "GK"]


class MockFeatureStore:
    """player_id 기반 결정적 mock.

    같은 player_id -> 항상 같은 피처/포지션.
    """

    def get_player_info(self, player_id: int, season_id: int = 0) -> dict:
        """선수의 메타 정보."""
        h = int(hashlib.md5(str(player_id).encode()).hexdigest(), 16)
        position = POSITION_BY_HASH[h % 4]
        age = 18 + (h % 18)  # 18 ~ 35
        return {
            "player_id": player_id,
            "name": f"Player_{player_id}",
            "position": position,
            "age": age,
            "current_league": "PRL",
            "height": 170 + (h % 25),
            "weight": 65 + (h % 25),
        }

    def get_features(self, player_id: int, season_id: int = 0) -> dict:
        """모델 입력으로 사용할 피처 dict — 백엔드 PlayerSeasonRecord 컬럼명과 일치."""
        h = int(hashlib.md5(f"{player_id}_{season_id}".encode()).hexdigest(), 16)
        return {
            "player_id": player_id,
            "season_id": season_id,
            "stat_minutes_played_total": 1000 + (h % 2500),
            "stat_goals_total": h % 20,
            "stat_shots_total_total": h % 80,
            "stat_assists_total": h % 15,
            "stat_passes_total": 200 + (h % 1500),
            "stat_key_passes_total": h % 30,
            "stat_tackles_total": h % 60,
            "stat_aeriels_won_total": h % 40,
            "stat_clearances_total": h % 50,
            "stat_blocked_shots_total": h % 25,
            "stat_saves_total": h % 100,
            "stat_cleansheets_total": h % 15,
            "stat_accurate_passes_percentage_total": 0.6 + (h % 35) / 100,
        }

    def exists(self, player_id: int) -> bool:
        """선수가 DB에 있는지. mock 에서는 음수만 없는 걸로 처리."""
        return player_id > 0

    def get_cached_performance(
        self,
        player_ids: list[int],
        destination_league: str,
    ) -> dict[int, dict]:
        """Mock 에서는 캐시 없음 — 항상 miss."""
        return {}


def _build_db_url() -> str:
    """환경변수에서 MySQL SQLAlchemy URL 조립."""
    user = os.getenv("DB_USER")
    pw = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME")
    return f"mysql+pymysql://{user}:{pw}@{host}:{port}/{name}?charset=utf8mb4"


class DbFeatureStore:
    """백엔드 RDS(MySQL) read-only 조회.

    백엔드 PlayerSeasonRecord/Player 엔티티의 실제 컬럼명을 사용한다.
    season 식별은 season_start_year DESC 로 최신 시즌을 자동 선택.

    DB_HOST 환경변수가 설정된 운영 환경에서만 활성화된다.
    """

    def __init__(self):
        # 지연 import: sqlalchemy 가 설치되어 있는 환경(EC2)에서만 동작
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine

        connect_args: dict = {"connect_timeout": 10}
        if os.getenv("DB_SSL_REQUIRED", "false").lower() == "true":
            # RDS SSL 강제 시 (true 만 주면 RDS 의 공용 CA 사용)
            connect_args["ssl"] = {"ssl_disabled": False}

        self.engine: Engine = create_engine(
            _build_db_url(),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args=connect_args,
        )
        self._test_connection()

    def _test_connection(self):
        from sqlalchemy import text
        from sqlalchemy.exc import OperationalError
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("DB 연결 OK (%s)", os.getenv("DB_HOST"))
        except OperationalError as e:
            logger.error("DB 연결 실패: %s", e)
            raise

    def get_player_info(self, player_id: int, season_id: int = 0) -> dict | None:
        """Player 엔티티 기준 메타 정보.

        Position/League 는 백엔드 EnumType.STRING 으로 저장됨
        (DB 값 예: "FW", "PREMIER_LEAGUE"). 우리 코드의 enum value 와 다를 수 있으므로
        호출 측에서 필요 시 매핑한다.
        """
        from sqlalchemy import text
        query = text("""
            SELECT
                id           AS player_id,
                name,
                position,
                age,
                height,
                weight,
                league       AS current_league
            FROM players
            WHERE id = :player_id
        """)
        with self.engine.connect() as conn:
            row = conn.execute(query, {"player_id": player_id}).first()
            return dict(row._mapping) if row else None

    def get_features(self, player_id: int, season_id: int = 0) -> dict | None:
        """선수의 최신 시즌 PlayerSeasonRecord 전체 컬럼 조회.

        AI Stage1/2 모델은 ~75 개 피처가 필요하므로 SELECT * 로 전부 가져와
        AI 어댑터 layer 에서 필요한 것만 골라 쓴다.

        우리 신규 API contract 에서 season_id 를 받지 않으므로 인자는 무시한다.
        """
        from sqlalchemy import text
        query = text("""
            SELECT *
            FROM player_season_records
            WHERE player_id = :player_id
            ORDER BY season_start_year DESC
            LIMIT 1
        """)
        with self.engine.connect() as conn:
            row = conn.execute(query, {"player_id": player_id}).first()
            return dict(row._mapping) if row else None

    def exists(self, player_id: int) -> bool:
        from sqlalchemy import text
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM players WHERE id = :pid LIMIT 1"),
                {"pid": player_id},
            ).first()
            return row is not None

    def get_cached_performance(
        self,
        player_ids: list[int],
        destination_league: str,
    ) -> dict[int, dict]:
        """캐시 테이블 player_performance_predictions 에서 (player_id, destination_league) 기준 조회.

        반환: player_id -> dict (백엔드 응답 필드와 동일 키만 포함).
        miss 인 player_id 는 결과 dict 에 없음.
        """
        if not player_ids:
            return {}
        from sqlalchemy import bindparam, text
        query = text("""
            SELECT
                player_id,
                pred_goals_total_per90,
                pred_shots_total_per90,
                pred_successful_dribbles_per90,
                pred_key_passes_per90,
                pred_passes_total_per90,
                pred_tackles_total_per90,
                pred_aeriels_won_per90,
                pred_blocked_shots_per90,
                pred_accurate_passes_pct,
                pred_cleansheets_total
            FROM player_performance_predictions
            WHERE destination_league = :league
              AND player_id IN :pids
        """).bindparams(bindparam("pids", expanding=True))
        with self.engine.connect() as conn:
            rows = conn.execute(
                query,
                {"league": destination_league, "pids": list(player_ids)},
            ).fetchall()
        out: dict[int, dict] = {}
        for r in rows:
            d = dict(r._mapping)
            pid = int(d.pop("player_id"))
            out[pid] = {k: (float(v) if v is not None else None) for k, v in d.items()}
        return out
