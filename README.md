# 5sondoson-ai

축구 선수가 다른 리그로 이적했을 때의 **퍼포먼스 / 시장가치 / 유사 선수**를 예측하는 AI 추론 서버.

백엔드(Spring Boot) 어드민 배치가 선수를 50명씩 청크로 잘라 HTTP POST 로 호출하면,
2단계 모델(Stage1 잔류 가정 예측 + Stage2 이적 보정 delta) 을 거쳐 결과를 돌려준다.

---

## 주요 특징

- **2-stage 캐스케이드 모델** — Stage1 으로 잔류 가정의 다음 시즌 통계를 예측하고, Stage2 가 이적 보정 delta 를 더해 최종 per90 통계를 만든다. 그 위에 market_value 모델(log EUR 회귀), similar_player 모듈(cosine 유사도) 이 동작한다.
- **세 가지 예측 엔드포인트** — performance(per90 스탯 10개) / market-value(EUR + 변화율) / similar-players(top-5).
- **백엔드 RDS read-only 연동** — 운영 환경에서는 백엔드 MySQL 을 직접 조회한다(`DbFeatureStore`). 로컬은 자동으로 `MockFeatureStore` 로 대체.
- **모델 파일 교체만으로 갱신** — `BasePredictor` / `ModelPredictor` / `MockPredictor` 추상화로 코드 수정 없이 .pkl/.joblib 만 S3 에 올리면 적용된다.
- **GitHub Actions 자동 배포** — `main` 푸시 → pytest → SSH 로 EC2 배포.

---

## 아키텍처

```
   ┌───────────────────┐    POST /predict/*     ┌──────────────────────┐
   │  Backend          │  ───────────────────▶  │  AI Server (FastAPI) │
   │  (Spring Boot)    │  ◀───────────────────  │  EC2 + Docker        │
   │  Admin Batch      │       JSON             │                      │
   └─────────┬─────────┘                        └────────┬─────────────┘
             │                                           │
             │  read-only SELECT                         │  download
             ▼                                           ▼
   ┌────────────────────┐                       ┌────────────────────┐
   │  RDS (MySQL)       │                       │  S3                │
   │  players,          │                       │  ai_pipeline/      │
   │  player_season_*,  │                       │  models/           │
   │  ...               │                       │  (.pkl / .joblib)  │
   └────────────────────┘                       └────────────────────┘
```

`MODEL_BUCKET` 환경변수가 있으면 컨테이너 부팅 시 S3 에서 모델을 다운로드하고, `DB_HOST` 가 있으면 RDS 모드로 동작한다. 둘 다 없으면 dummy/mock 으로 자동 전환되어 로컬에서 그대로 띄울 수 있다.

---

## API

| Method | Path | 설명 |
|--------|------|----|
| POST | `/predict/performance` | 이적 후 per90 스탯 예측 (10개 필드) |
| POST | `/predict/market-value` | 이적 후 시장가치(EUR) + 변화율 |
| POST | `/predict/similar-players` | 목적지 리그 후보 풀에서 cosine top-5 |
| GET | `/health` | 헬스체크 (`ready=true` 면 호출 가능) |
| GET | `/models/status` | 모델 로딩 상태 |
| GET | `/docs` | Swagger UI |

### 호출 규약

- JSON, 필드명은 **camelCase**.
- 한 요청당 **최대 50명** 청크 권장 (백엔드 어드민 배치 기본값, read timeout 300s).
- 부분 실패는 해당 선수의 `pred*` 필드만 `null` 로 반환한다 — 전체 요청이 실패하지 않는다.
- `destinationLeague` 는 리그 코드: `EPL` / `LA` / `BL` / `SA` / `L1`.
- 필드명 `aeriels` / `cleensheets` 는 백엔드 DTO 와 동일하게 의도된 표기다 (오타 아님).
- 호출은 항상 fresh 계산 — performance 응답을 캐싱하지 않는다. (단 market-value 는 Stage1+Stage2 중간 결과만 백엔드 캐시에서 재사용해 빠르게 동작하고, market_value 모델 호출은 매번 수행한다.)

자세한 필드 의미와 예시값은 서버의 `/docs` (Swagger UI) 에서 확인.

---

## 빠른 시작 (로컬)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training\train_dummy_models.py
uvicorn app.main:app --reload --port 8000
```

브라우저에서 http://127.0.0.1:8000/docs 로 Swagger UI 확인.

> Windows 에서는 `localhost` 가 IPv6(`::1`) 로 먼저 풀리고 uvicorn 은 기본적으로 IPv4(`127.0.0.1`) 에 바인딩되므로 항상 `127.0.0.1` 을 사용한다.

---

## 배포 / 운영

- **CI/CD**: [.github/workflows/deploy.yml](./.github/workflows/deploy.yml) — `main` 푸시 시 pytest 후 SSH 로 EC2 배포.
- **인프라**: AWS EC2 (ap-northeast-2), Elastic IP, S3(모델 보관), 백엔드 RDS read-only.
- **환경변수 분기**:
  - `MODEL_BUCKET` 설정 시 S3 에서 모델 다운로드, 없으면 `app/models/` 로컬 산출물 사용
  - `DB_HOST` 설정 시 `DbFeatureStore` 활성 (백엔드 RDS), 없으면 `MockFeatureStore`

---

## 폴더 구조

```
app/
├── main.py                FastAPI 엔트리포인트 (lifespan, CORS, 라우팅)
├── ai_pipeline_runner.py  AI 팀 Stage1+Stage2+market_value+similar 통합 어댑터
├── ai_pipeline/           AI 팀 산출물 (predict_pipeline.py, 모델, config)
├── handlers/              엔드포인트별 비즈니스 로직 (performance/market_value/similar_players)
├── features/              피처 스토어 (MockFeatureStore / DbFeatureStore)
├── models/                dummy 모델 registry + BasePredictor 추상화
└── schemas/               요청/응답 Pydantic 모델 + League/Position enum
tests/                     pytest 통합 테스트
training/                  dummy 모델 학습 스크립트
```

---

## 문서

- 작업 컨벤션: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 코드 작업 가이드 (Claude/Cursor 등): [CLAUDE.md](./CLAUDE.md)
- API 스펙: 서버 띄우고 `/docs`
