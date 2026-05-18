# CLAUDE.md

## 프로젝트 개요

**5sondoson-ai** 는 축구 선수가 다른 리그로 이적했을 때를 예측하는 AI 추론 서버다.
백엔드(Spring Boot) 어드민 배치 파이프라인이 선수를 50명씩 청크로 나눠
HTTP POST 로 호출한다(read timeout 300초). 부분 실패를 허용한다(`failed` 배열).

### 엔드포인트 3개

- `POST /predictions/performance` — 이적 후 퍼포먼스(스탯) 예측
- `POST /predictions/market-value` — 이적 후 시장가치(EUR) 예측
- `POST /predictions/similar-players` — 유사 선수 추천

### 도메인 모델

- 대상 리그 5개: `premier_league`, `la_liga`, `serie_a`, `bundesliga`, `ligue_1`
- 포지션 4개: `FW`, `MF`, `DF`, `GK`
- 포지션마다 예측 출력 키가 다르다 (예: FW = goals/shots/dribbles/key_passes/pass_accuracy).
- 모델 구성: 퍼포먼스 = 리그5 × 포지션4 = 20개, 시장가치 = 포지션4 = 4개,
  유사도 = 포지션4 = 4개. **총 28개**.

### 아키텍처 원칙

- 모델은 sklearn 기반 트리 모델(CPU 추론), `.joblib` 로 직렬화.
- 모델 파일은 S3 로 관리한다. 서버 시작 시 다운로드하고,
  파일이 없으면 `MockPredictor` 로 자동 대체한다.
- 실모델 교체는 코드 수정 없이 **파일 교체만**으로 가능해야 한다
  (`BasePredictor` / `ModelPredictor` / `MockPredictor` 추상화).
- 버전 문자열에 `dummy` 또는 `-mock` 이 포함되면 `is_mock=True` 로 분류된다.
- 피처 조회는 현재 mock(`MockFeatureStore`). 추후 DB 조회 구현체로 교체 예정.
- 배포: FastAPI + Docker + AWS EC2 + GitHub Actions(OIDC).

## 작업 컨벤션

이 레포에서 작업할 때는 [CONTRIBUTING.md](./CONTRIBUTING.md) 의 컨벤션을 반드시 따른다.

- 작업 흐름: `main` 에서 이슈 생성 → 브랜치 → 작업 → PR → 머지 → `main` 갱신 후 반복
- 브랜치: `feat/<작업 내용>` `chore/<작업 내용>` `fix/<작업 내용>`
- 커밋: Conventional Commits, 한 문장, 논리 단위마다 분리
- 이슈/PR 제목: `feat: 작업 내용`
- PR 연관 이슈 칸에 `- Closes #N` 으로 자동 종료
- 리뷰어 지정 안 함, 스크린샷 첨부 안 함
- 커밋·PR·이슈 본문에 작업 도구를 언급하지 않는다
- 언어는 한국어
