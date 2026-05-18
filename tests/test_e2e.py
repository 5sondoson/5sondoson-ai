"""3개 엔드포인트 전체 동작 검증."""
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
        assert "performance" in data
        assert "market_value" in data
        assert "similarity" in data
        # 28개 모델 확인 (20 + 4 + 4)
        assert len(data["performance"]) == 20
        assert len(data["market_value"]) == 4
        assert len(data["similarity"]) == 4
        print(f"  models/status 통과 (총 {20 + 4 + 4}개)")


def test_performance_basic():
    payload = {
        "players": [
            {"player_id": 10001, "season_id": 2024},
            {"player_id": 10002, "season_id": 2024},
            {"player_id": 10003, "season_id": 2024},
        ],
        "target_leagues": ["premier_league", "la_liga"],
    }
    with TestClient(app) as client:
        r = client.post("/predictions/performance", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["predictions"]) == 3
        assert data["meta"]["succeeded"] == 3
        assert data["meta"]["failed_count"] == 0

        # 포지션별 출력 키가 PR #9 스펙대로 들어가야 함
        for pred in data["predictions"]:
            position = pred["position"]
            stats = pred["by_league"]["premier_league"]["stats"]
            if position == "FW":
                assert set(stats.keys()) >= {"goals", "shots", "dribbles", "key_passes", "pass_accuracy"}
            elif position == "MF":
                assert set(stats.keys()) >= {"passes", "key_passes", "tackles", "pass_accuracy"}
            elif position == "DF":
                assert set(stats.keys()) >= {"aerials_won", "blocked_shots", "pass_accuracy"}
            elif position == "GK":
                assert set(stats.keys()) >= {"saves", "cleansheets", "pass_accuracy"}
        print(f"  performance 통과 ({len(data['predictions'])}명)")


def test_market_value_basic():
    payload = {
        "players": [{"player_id": 10001, "season_id": 2024}],
        "target_leagues": ["premier_league", "la_liga"],
    }
    with TestClient(app) as client:
        r = client.post("/predictions/market-value", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert pred["by_league"]["premier_league"] > 0
        assert pred["by_league"]["la_liga"] > 0
        print(f"  market_value 통과 (EPL={pred['by_league']['premier_league']:,.0f} EUR)")


def test_similar_players_basic():
    payload = {
        "players": [{"player_id": 10001, "season_id": 2024}],
        "target_leagues": ["premier_league"],
        "top_k": 3,
    }
    with TestClient(app) as client:
        r = client.post("/predictions/similar-players", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        pred = data["predictions"][0]
        similars = pred["by_league"]["premier_league"]
        assert len(similars) == 3
        # 유사도 내림차순 검증
        scores = [s["similarity_score"] for s in similars]
        assert scores == sorted(scores, reverse=True)
        print(f"  similar_players 통과 (top 3 유사도: {scores})")


def test_partial_failure():
    payload = {
        "players": [
            {"player_id": 10001, "season_id": 2024},
            {"player_id": -1, "season_id": 2024},
            {"player_id": 10003, "season_id": 2024},
        ],
        "target_leagues": ["premier_league"],
    }
    with TestClient(app) as client:
        r = client.post("/predictions/performance", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["meta"]["succeeded"] == 2
        assert data["meta"]["failed_count"] == 1
        assert data["failed"][0]["player_id"] == -1
        print(f"  partial_failure 통과 (성공 2, 실패 1)")


def test_batch_50_players():
    """백엔드 PR #9 스펙대로 50명 청크 테스트."""
    payload = {
        "players": [
            {"player_id": 10000 + i, "season_id": 2024}
            for i in range(50)
        ],
        "target_leagues": ["premier_league", "la_liga"],
    }
    with TestClient(app) as client:
        r = client.post("/predictions/performance", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["meta"]["succeeded"] == 50
        latency = data["meta"]["latency_ms"]
        # 50명 처리가 300초 한참 안에 끝나야 함 (실제로는 1초 이내)
        assert latency < 30000, f"latency {latency}ms 너무 길다"
        print(f"  batch_50 통과 (50명, {latency}ms)")


if __name__ == "__main__":
    print("=" * 60)
    print("통합 테스트 시작")
    print("=" * 60)
    test_health()
    test_models_status()
    test_performance_basic()
    test_market_value_basic()
    test_similar_players_basic()
    test_partial_failure()
    test_batch_50_players()
    print("=" * 60)
    print("✅ 모든 테스트 통과")
    print("=" * 60)
