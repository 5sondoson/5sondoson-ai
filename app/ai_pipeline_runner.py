"""AI 팀 predict_pipeline 의 thin wrapper.

DB 에서 가져온 선수 행 + 목적지 리그 → Stage1+Stage2 실행 → 최종 예측 DataFrame.

AI 팀 산출물 구조(`app/ai_pipeline/`) 의 코드는 import 가 까다로워서
`importlib` 로 동적 로드한다. import 시점에 모델 파일(.pkl/.joblib)이
디스크에 있어야 함 — S3 다운로드는 lifespan 에서 먼저 끝나야 한다.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.enums import League

logger = logging.getLogger(__name__)

AI_PIPELINE_DIR = Path(__file__).parent / "ai_pipeline"
PREDICT_PIPELINE_PATH = AI_PIPELINE_DIR / "predict_pipeline.py"


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
        self._pipeline = _load_module("predict_pipeline", PREDICT_PIPELINE_PATH)
        logger.info("AiPredictionPipeline 로드 완료 (%s)", PREDICT_PIPELINE_PATH)

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
        merged["position_code"] = row.get("position")
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
        return merged
