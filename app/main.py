"""5sondoson AI 추론 서버 엔트리포인트."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.features.store import DbFeatureStore, MockFeatureStore
from app.handlers.market_value import MarketValueHandler
from app.handlers.performance import PerformanceHandler
from app.handlers.similar_players import SimilarPlayersHandler
from app.models.registry import ModelRegistry
from app.schemas.api import (
    ErrorResponse,
    MarketValuePrediction,
    MarketValueRequest,
    PerformancePrediction,
    PerformanceRequest,
    SimilarPlayersPrediction,
    SimilarPlayersRequest,
)

# ===== 로깅 =====
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app.main")


# ===== Lifespan =====
state: dict = {}


def download_models_from_s3():
    """서버 시작 시 S3에서 모델 산출물 다운로드.

    MODEL_BUCKET 환경변수가 없으면 스킵 (로컬 개발).

    S3 prefix -> 로컬 대상 디렉토리 매핑:
    - models/        → app/models/        (dummy 모델, 옛 흐름)
    - ai_pipeline/   → app/ai_pipeline/   (AI 팀 실모델)
    """
    bucket = os.getenv("MODEL_BUCKET")
    if not bucket:
        logger.info("MODEL_BUCKET 미설정. 로컬 모델 사용.")
        return

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("boto3 미설치. S3 다운로드 스킵.")
        return

    app_dir = Path(__file__).parent
    prefixes = [
        ("models/", app_dir / "models"),
        ("ai_pipeline/", app_dir / "ai_pipeline"),
    ]
    allowed_ext = (".joblib", ".pkl", ".json", ".csv")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    total = 0

    for prefix, local_root in prefixes:
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(allowed_ext):
                        continue
                    rel_path = key[len(prefix):]
                    if not rel_path:
                        continue
                    local_path = local_root / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                        continue
                    s3.download_file(bucket, key, str(local_path))
                    total += 1
        except ClientError as e:
            logger.error("S3 다운로드 실패 (prefix=%s): %s", prefix, e)
            logger.warning("로컬 파일로 계속 진행")
    logger.info("S3 다운로드 완료: %d개 파일", total)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("서버 시작 중...")
    download_models_from_s3()

    models_dir = Path(__file__).parent / "models"
    state["registry"] = ModelRegistry(models_dir=models_dir)

    # 환경변수 DB_HOST 유무로 피처 스토어 분기.
    # 운영(EC2): DB_HOST 설정 → 백엔드 RDS 조회
    # 로컬:     DB_HOST 미설정 → Mock 사용
    if os.getenv("DB_HOST"):
        state["feature_store"] = DbFeatureStore()
        logger.info("DbFeatureStore 사용 (백엔드 RDS)")
    else:
        state["feature_store"] = MockFeatureStore()
        logger.info("MockFeatureStore 사용 (로컬 개발)")

    # AI 팀 실모델 파이프라인 로드 시도. 실패하면 dummy 사용.
    state["ai_pipeline"] = None
    try:
        from app.ai_pipeline_runner import AiPredictionPipeline
        state["ai_pipeline"] = AiPredictionPipeline()
        logger.info("AiPredictionPipeline 활성화 (Stage1+Stage2 실모델)")
    except Exception as e:
        logger.warning("AiPredictionPipeline 로드 실패 → dummy 모델 사용. %s", e)

    state["performance_handler"] = PerformanceHandler(
        registry=state["registry"],
        feature_store=state["feature_store"],
        ai_pipeline=state["ai_pipeline"],
    )
    state["market_value_handler"] = MarketValueHandler(
        registry=state["registry"],
        feature_store=state["feature_store"],
        ai_pipeline=state["ai_pipeline"],
    )
    state["similar_players_handler"] = SimilarPlayersHandler(
        registry=state["registry"],
        feature_store=state["feature_store"],
        ai_pipeline=state["ai_pipeline"],
    )

    logger.info("서버 준비 완료")
    yield
    logger.info("서버 종료")
    state.clear()


# ===== FastAPI 앱 =====
API_DESCRIPTION = """
축구 선수가 다른 리그로 이적했을 때를 예측하는 추론 서버.

### 엔드포인트
- **POST /predict/performance** — 이적 후 퍼포먼스(per90 스탯) 예측
- **POST /predict/market-value** — 이적 후 시장가치(EUR) 예측
- **POST /predict/similar-players** — 이적 후보 유사 선수 top-5 추천

### 호출 규약
- 모든 입출력은 JSON, 필드명은 **camelCase**.
- 선수는 한 요청당 **최대 50명** 단위로 청크해서 보낸다 (백엔드 어드민 배치 기본값).
- 부분 실패는 허용 — 해당 선수의 `pred*` 필드를 `null` 로 반환한다.
- 동일 (player_id, destinationLeague) 가 백엔드 캐시 테이블에 있으면 Stage1+Stage2 를 건너뛰고 캐시 값을 반환한다 (대폭 빠름).

### 필드명 메모
- `aeriels`, `cleensheets` 는 백엔드 DTO 와 동일하게 의도된 표기다 (오타 아님).
- pred_* 값은 모두 90분당 정규화 통계(per90), 단 `pred_cleensheets_total` 은 시즌 누적.
"""

tags_metadata = [
    {
        "name": "Health",
        "description": "서버 상태 / 모델 로딩 상태 점검.",
    },
    {
        "name": "Predictions",
        "description": "백엔드 어드민 배치가 호출하는 메인 예측 엔드포인트 3개.",
    },
]

app = FastAPI(
    title="5sondoson AI Server",
    version="0.1.0",
    description=API_DESCRIPTION,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ===== 전역 에러 핸들러 =====
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error on {request.url}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message=str(exc),
        ).model_dump(),
    )


# ===== 헬스/메타 엔드포인트 =====
@app.get("/", tags=["Health"], summary="서비스 메타 정보")
def root():
    """버전과 문서 위치를 짧게 알려준다."""
    return {"name": "5sondoson-ai", "version": "0.1.0", "docs": "/docs"}


@app.get("/health", tags=["Health"], summary="서버 헬스체크")
def health():
    """ALB/배포 스크립트가 사용. `ready=true` 면 예측 엔드포인트 호출 가능."""
    ready = "registry" in state
    return {"status": "ok" if ready else "starting", "ready": ready}


@app.get("/models/status", tags=["Health"], summary="모델 로딩 상태")
def models_status():
    """28개 dummy 모델(performance 20 + market_value 4 + similarity 4)의 로딩 상태."""
    if "registry" not in state:
        raise HTTPException(503, "registry not loaded yet")
    return state["registry"].status()


# ===== 메인 엔드포인트 3개 (백엔드 AiPredictionClient 가 호출하는 경로) =====
@app.post(
    "/predict/performance",
    response_model=list[PerformancePrediction],
    tags=["Predictions"],
    summary="이적 후 퍼포먼스(per90 스탯) 예측",
    description=(
        "Stage1(잔류 가정 예측) + Stage2(이적 보정 delta) 를 거쳐 90분당 통계를 예측한다. "
        "해당 (player_id, destinationLeague) 가 백엔드 캐시에 있으면 캐시 값을 즉시 반환한다."
    ),
)
def predict_performance(request: PerformanceRequest):
    if "performance_handler" not in state:
        raise HTTPException(503, "server not ready")
    return state["performance_handler"].handle(request)


@app.post(
    "/predict/market-value",
    response_model=list[MarketValuePrediction],
    tags=["Predictions"],
    summary="이적 후 시장가치(EUR) 예측",
    description=(
        "Stage1+Stage2 결과를 입력으로 market_value 모델이 log(EUR) 을 출력하고, "
        "서버에서 exp 변환해 정수 EUR 로 반환한다. mvChangeRate 는 현재 시장가치 대비 변화율."
    ),
)
def predict_market_value(request: MarketValueRequest):
    if "market_value_handler" not in state:
        raise HTTPException(503, "server not ready")
    return state["market_value_handler"].handle(request)


@app.post(
    "/predict/similar-players",
    response_model=list[SimilarPlayersPrediction],
    tags=["Predictions"],
    summary="이적 목적지 리그에서 유사 선수 top-5 추천",
    description=(
        "선수의 Stage1+Stage2 예측 벡터와 목적지 리그 후보 풀(Big Five)을 cosine 유사도로 비교해 "
        "상위 5명을 유사도 내림차순으로 반환한다."
    ),
)
def predict_similar_players(request: SimilarPlayersRequest):
    if "similar_players_handler" not in state:
        raise HTTPException(503, "server not ready")
    return state["similar_players_handler"].handle(request)
