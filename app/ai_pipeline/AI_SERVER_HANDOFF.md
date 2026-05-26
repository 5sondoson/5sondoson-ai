# AI Server Handoff

이 문서는 AI 서버 담당자가 Stage1/Stage2 예측 모델을 서비스에 연결할 때 필요한 산출물과 사용 방법을 정리한 문서입니다.

## 1. 전체 예측 흐름

최종 예측은 2단계로 수행됩니다.

```text
raw player features
-> Stage1 잔류 기준 다음 시즌 성적 예측
-> Stage1 prediction을 Stage2 input으로 전달
-> Stage2 이적 보정
-> final_after_pred 반환
```

Stage1은 선수가 이적하지 않고 잔류한다고 가정했을 때의 다음 시즌 성적을 예측합니다.

Stage2는 Stage1 예측값에 이적 정보를 반영해 변화량(delta)을 보정합니다.

```text
final_after_pred = stage1_pred + stage2_delta_pred
```

단, Stage2가 Stage1-only보다 성능을 개선하지 못한 타깃은 Stage2를 적용하지 않고 Stage1 예측값을 그대로 사용합니다.

```text
if stage2_applied:
    final_after_pred = stage1_pred + stage2_delta_pred
else:
    final_after_pred = stage1_pred
```

## 2. 서버에 전달해야 하는 파일/폴더

아래 3개는 함께 배포되어야 합니다.

```text
predict_pipeline.py
stage1_deploy_artifacts/
stage2_deploy_artifacts/
```

## 3. 통합 실행 파일

### `predict_pipeline.py`

Stage1과 Stage2를 연결하는 통합 wrapper입니다.

이 파일을 서버 코드에서 import해서 사용하면 됩니다.

```python
import pandas as pd
from predict_pipeline import predict_dataframe

df = pd.read_csv("input_players.csv")
result = predict_dataframe(df)

stage1_predictions = result["stage1_predictions"]
stage2_predictions = result["stage2_predictions"]
stage2_input_rows = result["stage2_input_rows"]
```

최종 서비스에서 주로 사용해야 하는 결과는 다음입니다.

```text
result["stage2_predictions"]
```

최종 예측값 컬럼은 다음입니다.

```text
final_after_pred
```

## 4. Stage1 산출물

### `stage1_deploy_artifacts/models/`

Stage1 타깃별 학습 완료 모델 파일이 들어 있습니다.

각 모델은 `.pkl` 형식입니다.

Stage1은 다음 시즌 단일 지표를 예측합니다.

### `stage1_deploy_artifacts/stage1_model_config.json`

Stage1 모델 설정 파일입니다.

타깃별로 다음 정보가 들어 있습니다.

```text
model_path
model_type
selected_features
minutes_threshold
current_feature
alpha_model
alpha_current
cv_mean_r2
cv_mean_mae
final_test_r2
```

Stage1 최종 예측은 모델 예측값과 현재 시즌 지표를 blending해서 생성됩니다.

```text
stage1_pred = alpha_model * model_pred + alpha_current * current_stat
```

### `stage1_deploy_artifacts/stage1_feature_schema.json`

Stage1 모델 실행에 필요한 입력 feature 목록입니다.

타깃별로 필요한 컬럼, dtype, 학습 데이터 기준 결측률, 예시 값이 들어 있습니다.

서버 입력 데이터가 이 스키마의 feature들을 포함해야 Stage1 예측이 가능합니다.

### `stage1_deploy_artifacts/model_manifest.csv`

Stage1 모델 요약표입니다.

모델 종류, 타깃, 출전시간 기준, feature subset, alpha, CV 성능, final test R2 등을 사람이 보기 쉬운 표 형태로 정리한 파일입니다.

### `stage1_deploy_artifacts/predict_stage1.py`

Stage1 단독 예측 helper입니다.

통합 예측에서는 `predict_pipeline.py`가 내부적으로 사용합니다.

### `stage1_deploy_artifacts/requirements.txt`

Stage1 실행에 필요한 Python 패키지 목록입니다.

## 5. Stage2 산출물

### `stage2_deploy_artifacts/models/`

Stage2 타깃별 delta 보정 모델 파일이 들어 있습니다.

각 모델은 `.joblib` 형식입니다.

Stage2 모델은 after 성적 자체를 직접 예측하지 않고, Stage1 예측값 대비 변화량을 예측합니다.

```text
stage2_delta_pred = actual_after - stage1_pred
```

### `stage2_deploy_artifacts/stage2_model_config.json`

Stage2 모델 설정 파일입니다.

타깃별로 다음 정보가 들어 있습니다.

```text
target
target_short_name
stage1_pred_col
after_col
model_file
model_type
feature_set
input_features
apply_stage2
cv_metrics
```

`apply_stage2`가 중요합니다.

```text
apply_stage2 = true
```

이면 Stage2 보정을 적용합니다.

```text
final_after_pred = stage1_pred + stage2_delta_pred
```

```text
apply_stage2 = false
```

이면 Stage2를 적용하지 않고 Stage1 예측값을 그대로 사용합니다.

```text
final_after_pred = stage1_pred
```

현재 Stage2 미적용 타깃은 다음입니다.

```text
accurate_passes_%
blocked_shots
```

### `stage2_deploy_artifacts/stage2_feature_schema.json`

Stage2 실행에 필요한 입력 feature와 파생 feature 설명입니다.

Stage2는 Stage1 예측값과 이적 context feature를 사용합니다.

필요한 주요 입력은 다음입니다.

```text
stage1_pred_* columns
transfer_path
position_code
player_age_before
before_stat_minutes_played_total_num
```

`predict_pipeline.py`를 사용하면 `stage1_pred_*` 컬럼은 Stage1 결과에서 자동 생성됩니다.

### `stage2_deploy_artifacts/stage2_selected_models_metrics.csv`

Stage2 최종 선택 모델의 성능표입니다.

타깃별로 다음 정보가 들어 있습니다.

```text
feature_set
model_type
n_total
oof_delta_r2
baseline_after_mae
stage2_after_mae
after_mae_improvement_pct
```

여기서 `baseline_after_mae`는 Stage1-only 기준 after MAE입니다.

`stage2_after_mae`는 Stage1+Stage2 보정 후 OOF after MAE입니다.

### `stage2_deploy_artifacts/predict_stage2.py`

Stage2 단독 예측 helper입니다.

통합 예측에서는 `predict_pipeline.py`가 내부적으로 사용합니다.

### `stage2_deploy_artifacts/requirements.txt`

Stage2 실행에 필요한 Python 패키지 목록입니다.

### `stage2_deploy_artifacts/README.md`

Stage2 artifact 사용 설명서입니다.

### `stage2_deploy_artifacts/NOTES.md`

Stage2 모델 선택 방식과 feature engineering 방식을 요약한 문서입니다.

## 6. 입력 데이터 요구사항

서버에서 `predict_pipeline.py`를 사용할 경우, 입력 row/DataFrame에는 다음이 필요합니다.

### Stage1용 선수 현재 시즌 feature

Stage1 모델별 required feature는 아래 파일에 정의되어 있습니다.

```text
stage1_deploy_artifacts/stage1_feature_schema.json
```

### Stage2용 이적 context feature

아래 컬럼이 필요합니다.

```text
transfer_path
position_code
player_age_before
before_stat_minutes_played_total_num
```

`transfer_path`는 다음 형식이어야 합니다.

```text
source league -> destination league
```

예:

```text
Liga Portugal -> La Liga
Eredivisie -> Premier League
Pro League -> Ligue 1
```

`source_league`, `destination_league`는 `predict_stage2.py`에서 자동으로 생성됩니다.

## 7. 출력 데이터

`predict_pipeline.py`의 `predict_dataframe(df)`는 dict를 반환합니다.

```text
stage1_predictions
stage2_predictions
stage2_input_rows
```

### `stage1_predictions`

Stage1 예측 결과입니다.

주요 컬럼:

```text
target
position
prediction
unit
confidence
warning
current_minutes
minutes_threshold
alpha_model
alpha_current
model_prediction
```

### `stage2_predictions`

최종 서비스 예측에 사용할 결과입니다.

주요 컬럼:

```text
target
target_short_name
stage1_pred
stage2_delta_pred
final_after_pred
stage2_applied
model_type
feature_set
cv_after_mae_improvement_pct
```

최종 예측값:

```text
final_after_pred
```

Stage2 적용 여부:

```text
stage2_applied
```

## 8. 출전시간 confidence 정책

Stage1은 타깃별 출전시간 기준을 가지고 있습니다.

예측은 항상 생성하지만, 현재 시즌 출전시간이 기준보다 낮으면 confidence가 낮아집니다.

```text
high: current_minutes >= minutes_threshold
medium: 0.7 * minutes_threshold <= current_minutes < minutes_threshold
low: current_minutes < 0.7 * minutes_threshold or missing current_minutes
```

## 9. 주의사항

1. `predict_pipeline.py`, `stage1_deploy_artifacts/`, `stage2_deploy_artifacts/`는 같은 폴더 레벨에 있어야 합니다.

2. Stage2는 Stage1 output을 필요로 하므로, Stage2 단독 실행 전에 Stage1 예측 컬럼이 필요합니다. 통합 wrapper를 쓰면 자동으로 처리됩니다.

3. Stage2에서 `apply_stage2=false`인 타깃은 Stage1 예측값이 최종값입니다.

4. 입력 데이터의 컬럼명이 학습 때 사용한 feature 이름과 일치해야 합니다.

5. categorical feature의 unknown 값은 모델 내부 인코더가 처리하지만, 입력 컬럼 자체가 없으면 예측이 실패합니다.

## 10. 설치 패키지

Stage1/Stage2 requirements를 모두 설치해야 합니다.

```text
stage1_deploy_artifacts/requirements.txt
stage2_deploy_artifacts/requirements.txt
```

주요 패키지:

```text
pandas
numpy
scikit-learn
xgboost
lightgbm
joblib
```

## 11. 간단 실행 예시

```python
import pandas as pd
from predict_pipeline import predict_dataframe

df = pd.read_csv("input_players.csv")
result = predict_dataframe(df)

final_predictions = result["stage2_predictions"]
print(final_predictions[["target_short_name", "final_after_pred", "stage2_applied"]])
```
