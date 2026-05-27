# CLAUDE.md

## 프로젝트 개요

**5sondoson-ai** 는 축구 선수가 다른 리그로 이적했을 때를 예측하는 AI 추론 서버다.
백엔드(Spring Boot) 어드민 배치 파이프라인이 선수를 50명씩 청크로 나눠
HTTP POST 로 호출한다(read timeout 300초). 부분 실패를 허용한다 — 해당 선수의
`pred*` 필드만 `null` 로 반환한다.

### 엔드포인트 3개 (+ 헬스)

- `POST /predict/performance` — 이적 후 per90 스탯 예측 (10개 필드)
- `POST /predict/market-value` — 이적 후 시장가치(EUR) + 변화율
- `POST /predict/similar-players` — 목적지 리그 후보 풀에서 cosine top-5
- `GET /health`, `GET /models/status`, `GET /docs` (Swagger UI)

### 도메인 모델

- 대상 리그 5개: `EPL`/`LA`/`BL`/`SA`/`L1` (백엔드 League enum value).
- 출신 리그(non-Big-Five): `ERE`/`PRL`/`BPL`.
- 포지션 4개: `FW`/`MF`/`DF`/`GK` (백엔드 표기). AI 모델은 `attacker`/`midfielder`/`defender`/`goalkeeper` (`Position.role_name`) 를 사용.
- 응답 필드명의 `aeriels` / `cleensheets` 는 **백엔드 DTO 와 동일하게 의도된 표기**다. 오타라고 판단해 절대 수정하지 말 것.

### 모델 파이프라인

- **운영**: AI 팀 산출물을 `app/ai_pipeline/` 에 통합 (`predict_pipeline.py` + Stage1/Stage2/market_value/similar deploy artifacts). 동적 로드는 `app/ai_pipeline_runner.py::AiPredictionPipeline` 어댑터가 담당.
  - Stage1 = 잔류 가정 다음 시즌 통계 예측
  - Stage2 = 이적 보정 delta 를 더해 최종 per90 통계
  - market_value 모델은 stage2 결과를 `pred_after_*` 컬럼으로 받아 **log(EUR)** 출력 → 서버에서 `np.exp` 변환해 정수 EUR 반환
  - similar_player 모듈은 stage1+stage2 결과로 query_vector 만들어 cosine 후보 매칭
- **Fallback**: AI 팀 산출물이 없을 때만 dummy 모델(`app/models/`) 로 자동 대체. 구성: 퍼포먼스 5리그×4포지션=20, 시장가치 4, 유사도 4 = 총 28개 (`MockPredictor`).
- 모델 산출물은 S3 (`MODEL_BUCKET`) 로 관리. 서버 시작 시 다운로드한 뒤 디스크에서 로드.

### 피처 스토어

- 운영: `DbFeatureStore` 가 백엔드 RDS(MySQL) 를 read-only 조회 (`players`, `player_season_records`, `player_performance_predictions` 등).
- 로컬: `DB_HOST` 환경변수가 없으면 자동으로 `MockFeatureStore` 로 대체.

### 캐시 정책

- performance / similar-players: 호출은 항상 fresh 계산, 응답을 캐싱하지 않는다.
- market-value: 백엔드 `player_performance_predictions` 의 캐시된 `pred_*_per90` 값을 Stage2 `final_after_pred` 대용으로 재사용해 Stage1+Stage2 를 스킵 가능. market_value 모델 호출은 매번 fresh — 응답 의미는 캐시 미사용과 동일, 단지 빠를 뿐.

### 아키텍처 원칙

- 모델은 sklearn 기반(`.pkl`/`.joblib`), CPU 추론.
- `BasePredictor` / `ModelPredictor` / `MockPredictor` 추상화 — 파일 교체만으로 모델 갱신.
- 버전 문자열에 `dummy` 또는 `-mock` 포함 시 `is_mock=True` 로 분류.
- 배포: FastAPI + Docker + AWS EC2 + GitHub Actions(SSH).

## 작업 컨벤션

이 레포에서 작업할 때는 [CONTRIBUTING.md](./CONTRIBUTING.md) 의 컨벤션을 반드시 따른다.

- 작업 흐름: `main` 에서 이슈 생성 → 브랜치 → 작업 → PR → 머지 → `main` 갱신 후 반복
- 브랜치: `feat/<작업 내용>` / `chore/<작업 내용>` / `fix/<작업 내용>` / `docs/<작업 내용>`
- 커밋: Conventional Commits, 한 문장, 논리 단위마다 분리
- 이슈/PR 제목: `feat: 작업 내용`
- PR 연관 이슈 칸에 `- Closes #N` 으로 자동 종료
- 리뷰어 지정 안 함, 스크린샷 첨부 안 함
- 커밋·PR·이슈 본문에 작업 도구를 언급하지 않는다
- 언어는 한국어
