"""3개 엔드포인트 통합 동작 검증 (백엔드 호출 contract 기준)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        print("  health 통과")


def test_models_status():
    with TestClient(app) as client:
        r = client.get("/models/status")
        assert r.status_code == 200
        data = r.json()
        assert len(data["performance"]) == 20  # 5 leagues * 4 positions
        assert len(data["market_value"]) == 4
        assert len(data["similarity"]) == 4
        print("  models/status 통과 (총 28개)")


def test_performance_basic():
    payload = {
        "playerIds": [10001, 10002, 10003],
        "destinationLeague": "EPL",
    }
    with TestClient(app) as client:
        r = client.post("/predict/performance", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 3
        ids = {p["playerId"] for p in data}
        assert ids == {10001, 10002, 10003}
        any_pred = any(
            any(v is not None for k, v in p.items() if k.startswith("pred"))
            for p in data
        )
        assert any_pred
        print(f"  performance 통과 ({len(data)}명)")


def test_market_value_basic():
    payload = {"playerIds": [10001], "destinationLeague": "EPL"}
    with TestClient(app) as client:
        r = client.post("/predict/market-value", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data) == 1
        p = data[0]
        assert p["playerId"] == 10001
        assert p["predictedMv"] is not None and p["predictedMv"] > 0
        print(f"  market_value 통과 (predictedMv={p['predictedMv']:,})")


def test_similar_players_basic():
    payload = {"playerIds": [10001], "destinationLeague": "EPL"}
    with TestClient(app) as client:
        r = client.post("/predict/similar-players", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data) == 1
        sp = data[0]["similarPlayers"]
        assert len(sp) == 5  # DEFAULT_TOP_K
        scores = [e["similarityScore"] for e in sp]
        assert scores == sorted(scores, reverse=True)
        print(f"  similar_players 통과 (top 5 내림차순: {scores})")


def test_partial_failure():
    payload = {"playerIds": [10001, -1, 10003], "destinationLeague": "EPL"}
    with TestClient(app) as client:
        r = client.post("/predict/performance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3
        failed_entry = next(p for p in data if p["playerId"] == -1)
        assert all(
            failed_entry[k] is None for k in failed_entry if k.startswith("pred")
        )
        print("  partial_failure 통과 (실패 entry pred_* 전부 None)")


def test_batch_100_players():
    """백엔드 default chunk-size=100 기준 배치 테스트."""
    payload = {
        "playerIds": list(range(10000, 10100)),
        "destinationLeague": "EPL",
    }
    with TestClient(app) as client:
        r = client.post("/predict/performance", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data) == 100
        print("  batch_100 통과 (100명)")


if __name__ == "__main__":
    print("=" * 60)
    print("통합 테스트 시작 (백엔드 contract 정렬판)")
    print("=" * 60)
    test_health()
    test_models_status()
    test_performance_basic()
    test_market_value_basic()
    test_similar_players_basic()
    test_partial_failure()
    test_batch_100_players()
    print("=" * 60)
    print("✅ 모든 테스트 통과")
    print("=" * 60)
