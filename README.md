# 5sondoson-ai

축구 선수 이적 후 퍼포먼스 / 시장가치 / 유사 선수 예측 AI 추론 서버.

## 빠른 시작

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training\train_dummy_models.py
uvicorn app.main:app --reload --port 8000
```

## API

- POST /predictions/performance
- POST /predictions/market-value
- POST /predictions/similar-players
