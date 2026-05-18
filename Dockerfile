FROM python:3.11-slim

# 시스템 패키지 (lightgbm/xgboost 실행에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 (캐싱 활용: requirements.txt 안 바뀌면 재설치 안 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드
COPY app/ ./app/
COPY training/ ./training/

# 빌드 시점에 dummy 모델 생성 (S3 모델 없을 때 fallback)
RUN python training/train_dummy_models.py

# CPU 추론 최적화: 수치 연산 라이브러리(OpenMP/OpenBLAS/MKL)의
# 내부 스레드 풀을 프로세스당 1개로 고정한다.
# uvicorn 을 여러 워커(프로세스)로 띄우므로, 각 프로세스가 또
# 멀티스레드로 돌면 CPU 코어를 과점유(oversubscription)해 오히려 느려진다.
# sklearn 트리 모델 추론은 단일 스레드로도 충분히 빠르고, 병렬성은
# 워커 프로세스 수로 확보한다.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
