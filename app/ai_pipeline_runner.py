"""AI 팀 predict_pipeline 의 thin wrapper.

DB 에서 가져온 선수 행 + 목적지 리그 → Stage1+Stage2 실행 → 최종 예측 DataFrame.
필요 시 market_value 모델까지 호출해 EUR 예측을 돌려준다.

AI 팀 산출물 구조(`app/ai_pipeline/`) 의 코드는 import 가 까다로워서
`importlib` 로 동적 로드한다. import 시점에 모델 파일(.pkl/.joblib)이
디스크에 있어야 함 — S3 다운로드는 lifespan 에서 먼저 끝나야 한다.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from app.schemas.enums import League, Position

logger = logging.getLogger(__name__)

AI_PIPELINE_DIR = Path(__file__).parent / "ai_pipeline"
PREDICT_PIPELINE_PATH = AI_PIPELINE_DIR / "predict_pipeline.py"
MARKET_VALUE_DIR = AI_PIPELINE_DIR / "market_value"
MARKET_VALUE_MODEL = MARKET_VALUE_DIR / "market_value_with_stage2_perf_model_sample10.pkl"
MARKET_VALUE_FEATURES = MARKET_VALUE_DIR / "market_value_with_stage2_perf_features_sample10.pkl"
SIMILAR_PLAYER_PATH = AI_PIPELINE_DIR / "similar_player_deploy_artifacts" / "predict_similar_players.py"

# Stage2 target_short_name -> market_value 모델의 pred_after_* 컬럼명
_TARGET_TO_PRED_AFTER: dict[str, str] = {
    "goals": "pred_after_goals",
    "shots": "pred_after_shots",
    "successful_dribbles": "pred_after_successful_dribbles",
    "key_passes": "pred_after_key_passes",
    "passes": "pred_after_passes",
    "tackles": "pred_after_tackles",
    "aerials_won": "pred_after_aerials_won",  # market_value 모델은 "aerials" 표기(백엔드 DB typo 와 다름)
    "blocked_shots": "pred_after_blocked_shots",
    "cleansheets": "pred_after_cleansheets",
    "accurate_passes_%": "pred_after_accurate_passes_pct",
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AiPredictionPipeline:
    """Stage1 + Stage2 를 한 번에 실행하는 wrapper.

    handlers 에서 사용:
      stage2_df = pipeline.predict(rows, League.PREMIER_LEAGUE)
      # final_after_pred 와 stage2_applied 가 target 별로 들어있음
    """

    def __init__(self):
        # 모델 binary 가 디스크에 없으면 dummy fallback 으로 위임.
        stage1_models = AI_PIPELINE_DIR / "stage1_deploy_artifacts" / "models"
        if not stage1_models.exists() or not any(stage1_models.glob("*.pkl")):
            raise RuntimeError(
                f"Stage1 모델 파일 없음 ({stage1_models}). "
                f"S3 다운로드 또는 로컬 배치가 선행되어야 합니다."
            )
        self._pipeline = _load_module("predict_pipeline", PREDICT_PIPELINE_PATH)
        logger.info("AiPredictionPipeline 로드 완료 (%s)", PREDICT_PIPELINE_PATH)

        # market_value 모델은 있으면 로드, 없으면 무시 (Stage2까지는 정상 동작).
        self._mv_model = None
        self._mv_features: list[str] = []
        if MARKET_VALUE_MODEL.exists() and MARKET_VALUE_FEATURES.exists():
            self._mv_model = joblib.load(MARKET_VALUE_MODEL)
            self._mv_features = joblib.load(MARKET_VALUE_FEATURES)
            logger.info("market_value 모델 로드 완료 (%d features)", len(self._mv_features))
        else:
            logger.warning("market_value 모델 파일 없음 — 시장가치 예측은 dummy fallback")

        # similar_players 모듈 (recommend_similar_players)
        self._similar_module = None
        if SIMILAR_PLAYER_PATH.exists():
            self._similar_module = _load_module("predict_similar_players", SIMILAR_PLAYER_PATH)
            logger.info("similar_player 모듈 로드 완료")
        else:
            logger.warning("predict_similar_players.py 없음 — 유사선수는 dummy fallback")

    def predict(
        self,
        player_rows: list[dict[str, Any]],
        destination_league: League,
    ) -> pd.DataFrame:
        """선수 행들 + 목적지 리그 → stage2_predictions long DataFrame.

        주요 컬럼: target / target_short_name / stage1_pred / stage2_delta_pred /
                   final_after_pred / stage2_applied / row_index.
        """
        if not player_rows:
            return pd.DataFrame()

        rows = [
            self._build_input_row(row, destination_league)
            for row in player_rows
        ]
        df = pd.DataFrame(rows)
        result = self._pipeline.predict_dataframe(df)
        return result["stage2_predictions"]

    @property
    def has_market_value(self) -> bool:
        return self._mv_model is not None

    @property
    def has_similar(self) -> bool:
        return self._similar_module is not None

    def predict_similar(
        self,
        player_rows: list[dict[str, Any]],
        destination_league: League,
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """선수별 top_k 유사선수 후보 반환.

        반환: rows 와 같은 길이의 list. 각 항목은 [(similar_player_name, similarity_score), ...].
        후보 풀의 player_id 는 백엔드 DB id 와 체계가 달라, 이름을 키로 반환하고
        핸들러에서 백엔드 player_id 로 변환한다. 실패한 row 는 빈 list.
        """
        if not self._similar_module or not player_rows:
            return [[] for _ in player_rows]

        results: list[list[tuple[str, float]]] = []
        dest_display = destination_league.display_name
        for row in player_rows:
            try:
                built = self._build_input_row(row, destination_league)
                ret = self._similar_module.recommend_similar_players(
                    built,
                    destination_league=dest_display,
                    top_k=top_k,
                )
                recs = ret["recommendations"]
                entries: list[tuple[str, float]] = []
                for _, r in recs.iterrows():
                    name = r.get("player_name")
                    sim = r.get("similarity")
                    if name is None or pd.isna(name) or pd.isna(sim):
                        continue
                    entries.append((str(name), float(sim)))
                results.append(entries)
            except Exception:
                logger.exception("similar_player 호출 실패 player_id=%s", row.get("player_id"))
                results.append([])
        return results

    def predict_market_value(
        self,
        player_rows: list[dict[str, Any]],
        destination_league: League,
        cached_by_pid: Optional[dict[int, dict[str, float]]] = None,
    ) -> list[Optional[int]]:
        """Stage1+Stage2 결과를 활용해 market_value 모델로 EUR 예측.

        cached_by_pid 가 있으면 해당 player_id 는 Stage1+Stage2 호출을 스킵하고
        캐시 값을 stage2 final_after_pred 로 간주해 pred_after_* 입력에 사용한다.
        형식: {player_id: {target_short_name: value}} (e.g. {123: {"goals": 0.45}}).

        반환 길이는 player_rows 와 동일. 예측 실패한 row 는 None.
        """
        if not self._mv_model or not player_rows:
            return [None] * len(player_rows)

        cached_by_pid = cached_by_pid or {}
        pid_to_idx: dict[int, int] = {}
        miss_rows: list[dict[str, Any]] = []
        miss_local_to_global: list[int] = []
        for i, row in enumerate(player_rows):
            pid = row.get("player_id")
            if pid is not None:
                pid_to_idx[int(pid)] = i
            if pid is None or int(pid) not in cached_by_pid:
                miss_local_to_global.append(i)
                miss_rows.append(row)

        # row_index → pred_after_* dict (player_rows 전역 인덱스 기준)
        pred_after_by_idx: dict[int, dict[str, float]] = {}

        # 캐시 hit: 캐시된 short_name → pred_after_* 입력
        for pid, short_to_val in cached_by_pid.items():
            gidx = pid_to_idx.get(int(pid))
            if gidx is None:
                continue
            for short_name, value in short_to_val.items():
                field = _TARGET_TO_PRED_AFTER.get(short_name)
                if field and value is not None:
                    pred_after_by_idx.setdefault(gidx, {})[field] = float(value)

        # 캐시 miss: Stage1+Stage2 호출 후 stage2_df 에서 pred_after_* 추출
        if miss_rows:
            stage2_df = self.predict(miss_rows, destination_league)
            for _, r in stage2_df.iterrows():
                local_idx = int(r["row_index"])
                if local_idx >= len(miss_local_to_global):
                    continue
                gidx = miss_local_to_global[local_idx]
                field = _TARGET_TO_PRED_AFTER.get(r["target_short_name"])
                value = r["final_after_pred"]
                if field and not pd.isna(value):
                    pred_after_by_idx.setdefault(gidx, {})[field] = float(value)

        # market_value 모델 입력 구성 — league 컬럼은 enum 이름("PREMIER_LEAGUE") 사용
        prepared: list[dict[str, Any]] = []
        for i, row in enumerate(player_rows):
            built = self._build_input_row(row, destination_league)
            built["league"] = destination_league.name
            built.update(pred_after_by_idx.get(i, {}))
            prepared.append(built)

        df = pd.DataFrame(prepared).reindex(columns=self._mv_features)
        try:
            # 모델 출력은 log(EUR). 실제 EUR 로 변환해 반환.
            log_eur = self._mv_model.predict(df)
            return [int(round(np.exp(float(v)))) for v in log_eur]
        except Exception:
            logger.exception("market_value 모델 예측 실패")
            return [None] * len(player_rows)

    @staticmethod
    def _build_input_row(
        row: dict[str, Any],
        destination_league: League,
    ) -> dict[str, Any]:
        """DB 행을 AI 모델 입력 형태로 변환.

        - 백엔드 DB 의 league 컬럼은 EnumType.STRING 으로 저장 → enum 이름
          ("PREMIER_LEAGUE") 가 들어있음. AI 모델 transfer_path 는 사람이름
          ("Premier League") 사용하므로 변환 필요.
        - Stage2 에 필요한 `transfer_path`, `position_code`, `player_age_before`,
          `before_stat_minutes_played_total_num` 을 만들어 넣는다.
        - Stage1 도 `player_age` / `player_height` / `player_weight` / `player_age_sq`
          같은 derived 컬럼을 요구하므로 함께 채운다.
        """
        merged = dict(row)
        src_db = row.get("current_league") or row.get("league")
        try:
            source_display = League[src_db].display_name if src_db else "missing"
        except KeyError:
            source_display = "missing"
        dest_display = destination_league.display_name

        merged["transfer_path"] = f"{source_display} -> {dest_display}"
        # 백엔드 DB 의 position 은 "FW"/"MF"/"DF"/"GK" enum 이름.
        # AI 팀 모델은 "attacker"/"midfielder"/... role 명을 사용한다.
        pos_db = row.get("position")
        if pos_db:
            try:
                merged["position_code"] = Position(pos_db).role_name
            except ValueError:
                merged["position_code"] = pos_db
        merged["player_age_before"] = row.get("age")
        merged["before_stat_minutes_played_total_num"] = row.get(
            "stat_minutes_played_total"
        )
        merged["player_age"] = row.get("age")
        merged["player_height"] = row.get("height")
        merged["player_weight"] = row.get("weight")
        age = row.get("age")
        if age is not None:
            try:
                merged["player_age_sq"] = float(age) ** 2
            except (TypeError, ValueError):
                pass
        # AI 팀이 알려준 market_value alias (학습 컬럼명에 맞춤)
        minutes = row.get("stat_minutes_played_total")
        appearances = row.get("stat_appearances_total")
        lineups = row.get("stat_lineups_total")
        if minutes is not None:
            merged["minutes"] = minutes
        if appearances is not None:
            merged["appearances"] = appearances
        if lineups is not None:
            merged["lineups"] = lineups
        try:
            if appearances and float(appearances) > 0 and lineups is not None:
                merged["lineup_rate"] = float(lineups) / float(appearances)
        except (TypeError, ValueError):
            pass
        return merged
