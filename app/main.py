"""5sondoson AI 추론 서버 엔트리포인트."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.features.store import MockFeatureStore
from app.handlers.market_value import MarketValueHandler
from app.handlers.performance import PerformanceHandler
from app.handlers.similar_players import SimilarPlayersHandler
from app.models.registry import ModelRegistry
from app.schemas.api import (
    ErrorResponse,
    MarketValueRequest,
    MarketValueResponse,
    PerformanceRequest,
    PerformanceResponse,
    SimilarPlayersRequest,
    SimilarPlayersResponse,
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
    """서버 시작 시 S3에서 모델 다운로드.

    MODEL_BUCKET 환경변수가 없으면 스킵 (로컬 개발).
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

    logger.info(f"S3에서 모델 다운로드: s3://{bucket}/models/")
    s3 = boto3.client("s3")
    local_dir = Path(__file__).parent / "models"

    try:
        paginator = s3.get_paginator("list_objects_v2")
        n_downloaded = 0
        for page in paginator.paginate(Bucket=bucket, Prefix="models/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not (key.endswith(".joblib") or key.endswith(".json")):
                    continue
                rel_path = key.replace("models/", "", 1)
                local_path = local_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                    continue
                s3.download_file(bucket, key, str(local_path))
                n_downloaded += 1
        logger.info(f"S3 다운로드 완료: {n_downloaded}개 파일")
    except ClientError as e:
        logger.error(f"S3 다운로드 실패: {e}")
        logger.warning("로컬 모델로 계속 진행")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("서버 시작 중...")
    download_models_from_s3()

    models_dir = Path(__file__).parent / "models"
    state["registry"] = ModelRegistry(models_dir=models_dir)
    state["feature_store"] = MockFeatureStore()
    state["performance_handler"] = PerformanceHandler(
        registry=state["registry"],
        feature_store=state["feature_store"],
    )
    state["market_value_handler"] = MarketValueHandler(
        registry=state["registry"],
        feature_store=state["feature_store"],
    )
    state["similar_players_handler"] = SimilarPlayersHandler(
        registry=state["registry"],
        feature_store=state["feature_store"],
    )

    logger.info("서버 준비 완료")
    yield
    logger.info("서버 종료")
    state.clear()


# ===== FastAPI 앱 =====
app = FastAPI(
    title="5sondoson AI Server",
    version="0.1.0",
    description="축구 선수 이적 후 퍼포먼스/시장가치/유사 선수 예측",
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
@app.get("/")
def root():
    return {"name": "5sondoson-ai", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
def health():
    ready = "registry" in state
    return {"status": "ok" if ready else "starting", "ready": ready}


@app.get("/models/status")
def models_status():
    if "registry" not in state:
        raise HTTPException(503, "registry not loaded yet")
    return state["registry"].status()


# ===== 메인 엔드포인트 3개 =====
@app.post("/predictions/performance", response_model=PerformanceResponse)
def predict_performance(request: PerformanceRequest):
    if "performance_handler" not in state:
        raise HTTPException(503, "server not ready")
    return state["performance_handler"].handle(request)


@app.post("/predictions/market-value", response_model=MarketValueResponse)
def predict_market_value(request: MarketValueRequest):
    if "market_value_handler" not in state:
        raise HTTPException(503, "server not ready")
    return state["market_value_handler"].handle(request)


@app.post("/predictions/similar-players", response_model=SimilarPlayersResponse)
def predict_similar_players(request: SimilarPlayersRequest):
    if "similar_players_handler" not in state:
        raise HTTPException(503, "server not ready")
    return state["similar_players_handler"].handle(request)
