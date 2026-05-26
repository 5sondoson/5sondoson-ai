from __future__ import annotations

import importlib.util
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT_DIR = ARTIFACT_DIR.parent
CONFIG_PATH = ARTIFACT_DIR / "similar_player_config.json"
DEFAULT_CANDIDATE_POOL = ARTIFACT_DIR / "big_five_candidate_pool.csv"
PLAYER_SEASON_CSV = ROOT_DIR / "sportmonks" / "data" / "sportmonks_player_season" / "sportmonks_player_season_all.csv"

BIG_FIVE_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

PER90_BASE_COLUMNS = {
    "goals": "stat_goals_total",
    "shots": "stat_shots_total_total",
    "successful_dribbles": "stat_successful_dribbles_total",
    "aerials_won": "stat_aeriels_won_total",
    "blocked_shots": "stat_blocked_shots_total",
    "key_passes": "stat_key_passes_total",
    "passes": "stat_passes_total",
    "tackles": "stat_tackles_total",
}

ACTUAL_FEATURE_COLUMNS = {
    "goals": "stat_goals_total_per90",
    "shots": "stat_shots_total_total_per90",
    "successful_dribbles": "stat_successful_dribbles_total_per90",
    "aerials_won": "stat_aeriels_won_total_per90",
    "blocked_shots": "stat_blocked_shots_total_per90",
    "accurate_passes_%": "stat_accurate_passes_percentage_total",
    "cleansheets": "stat_cleansheets_total",
    "key_passes": "stat_key_passes_total_per90",
    "passes": "stat_passes_total_per90",
    "tackles": "stat_tackles_total_per90",
}

POSITION_VECTOR_FEATURES = {
    "attacker": ["goals", "shots", "successful_dribbles", "player_age"],
    "defender": ["aerials_won", "blocked_shots", "player_age"],
    "goalkeeper": ["accurate_passes_%", "cleansheets", "player_age"],
    "midfielder": ["key_passes", "passes", "tackles", "player_age"],
}

METADATA_COLUMNS = [
    "player_id",
    "player_name",
    "player_display_name",
    "team_id",
    "team_name",
    "league_name",
    "season_id",
    "season_name",
    "position_code",
    "position_name",
    "stat_minutes_played_total",
]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predict_pipeline = _load_module("predict_pipeline", ROOT_DIR / "predict_pipeline.py")


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _season_start_year(value: object) -> int:
    if pd.isna(value):
        return -1
    try:
        return int(str(value).split("/", 1)[0])
    except ValueError:
        return -1


def latest_season_name(df: pd.DataFrame) -> str:
    seasons = df["season_name"].dropna().astype(str).unique()
    return max(seasons, key=_season_start_year)


def add_player_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "player_age" in out.columns:
        out["player_age"] = pd.to_numeric(out["player_age"], errors="coerce")
        out["player_age_sq"] = out["player_age"] ** 2
        return out

    if "player_date_of_birth" not in out.columns:
        out["player_age"] = np.nan
        out["player_age_sq"] = np.nan
        return out

    dob = pd.to_datetime(out["player_date_of_birth"], errors="coerce")
    if "season_start" in out.columns:
        ref = pd.to_datetime(out["season_start"], errors="coerce")
    else:
        ref = pd.to_datetime(
            out["season_name"].map(lambda x: f"{_season_start_year(x)}-08-01" if _season_start_year(x) > 0 else None),
            errors="coerce",
        )
    out["player_age"] = (ref - dob).dt.days / 365.25
    out["player_age_sq"] = out["player_age"] ** 2
    return out


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = pd.to_numeric(out.get("stat_minutes_played_total"), errors="coerce")
    valid_minutes = minutes.where(minutes > 0)
    for base_col in PER90_BASE_COLUMNS.values():
        if base_col in out.columns:
            out[f"{base_col}_per90"] = pd.to_numeric(out[base_col], errors="coerce") / valid_minutes * 90.0
    return out


def prepare_player_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_per90_features(add_player_age(df))


def build_candidate_pool(
    source_csv: str | Path = PLAYER_SEASON_CSV,
    output_csv: str | Path = DEFAULT_CANDIDATE_POOL,
    season_name: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(source_csv, low_memory=False)
    df = prepare_player_features(df)
    df = df[df["league_name"].isin(BIG_FIVE_LEAGUES)].copy()
    df = df[df["position_code"].isin(POSITION_VECTOR_FEATURES)].copy()
    if season_name is None:
        season_name = latest_season_name(df)
    df = df[df["season_name"].astype(str).eq(str(season_name))].copy()

    for short_name, col in ACTUAL_FEATURE_COLUMNS.items():
        df[short_name] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    keep_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    keep_cols += sorted(set(ACTUAL_FEATURE_COLUMNS) | {"player_age"})
    pool = df[keep_cols].drop_duplicates(subset=["player_id", "team_id", "season_name"], keep="first")
    pool = pool.reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return pool


def _as_frame(data: pd.DataFrame | dict[str, Any] | pd.Series) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame([dict(data)])


def build_query_vector(
    player_row: dict[str, Any] | pd.Series,
    destination_league: str,
    source_league: str | None = None,
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    row = prepare_player_features(_as_frame(player_row)).iloc[0].to_dict()
    source = source_league or row.get("league_name") or "missing"
    row["transfer_path"] = f"{source} -> {destination_league}"
    row["player_age_before"] = row.get("player_age")
    row["before_stat_minutes_played_total_num"] = row.get("stat_minutes_played_total")

    outputs = predict_pipeline.predict_one(row)
    stage2_predictions = outputs["stage2_predictions"]
    position = str(row.get("position_code"))
    if position not in POSITION_VECTOR_FEATURES:
        raise ValueError(f"Unsupported position_code: {position}")

    pred_map = {
        str(pred_row["target_short_name"]): float(pred_row["final_after_pred"])
        for _, pred_row in stage2_predictions.iterrows()
    }
    vector = {
        feature: pred_map[feature]
        for feature in POSITION_VECTOR_FEATURES[position]
        if feature != "player_age" and feature in pred_map
    }
    vector["player_age"] = float(row["player_age"]) if pd.notna(row.get("player_age")) else np.nan
    return vector, outputs


def recommend_similar_players(
    player_row: dict[str, Any] | pd.Series,
    destination_league: str,
    top_k: int = 5,
    candidate_pool_path: str | Path = DEFAULT_CANDIDATE_POOL,
    source_league: str | None = None,
) -> dict[str, pd.DataFrame]:
    config = load_config()
    if destination_league not in config["candidate_leagues"]:
        raise ValueError(f"Unsupported destination_league: {destination_league}")

    candidate_pool_path = Path(candidate_pool_path)
    if not candidate_pool_path.exists():
        build_candidate_pool(output_csv=candidate_pool_path)

    row = prepare_player_features(_as_frame(player_row)).iloc[0].to_dict()
    position = str(row.get("position_code"))
    if position not in POSITION_VECTOR_FEATURES:
        raise ValueError(f"Unsupported position_code: {position}")

    query_vector, model_outputs = build_query_vector(row, destination_league, source_league=source_league)
    vector_features = POSITION_VECTOR_FEATURES[position]

    pool = pd.read_csv(candidate_pool_path, low_memory=False)
    pool = pool[
        pool["position_code"].astype(str).eq(position)
        & pool["league_name"].astype(str).eq(destination_league)
    ].copy()
    stat_features = [feature for feature in vector_features if feature != "player_age"]
    pool = pool.dropna(subset=stat_features).copy()
    if pool.empty:
        raise ValueError(f"No candidate pool rows for position={position}, destination_league={destination_league}")

    x = pool.reindex(columns=vector_features).apply(pd.to_numeric, errors="coerce")
    q = pd.DataFrame([{feature: query_vector.get(feature, np.nan) for feature in vector_features}])
    medians = x.median(numeric_only=True)
    x = x.fillna(medians)
    q = q.fillna(medians)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    q_scaled = scaler.transform(q)
    similarities = cosine_similarity(q_scaled, x_scaled)[0]

    result = pool.copy()
    result["similarity"] = similarities
    for feature in vector_features:
        result[f"query_{feature}"] = float(q.iloc[0][feature])
        result[f"candidate_{feature}"] = x[feature].astype(float).to_numpy()
        result[f"diff_{feature}"] = result[f"candidate_{feature}"] - result[f"query_{feature}"]
    result["candidate_minutes_confidence"] = pd.cut(
        pd.to_numeric(result["stat_minutes_played_total"], errors="coerce"),
        bins=[-np.inf, 450, 900, 1200, np.inf],
        labels=["very_low", "low", "medium", "high"],
    ).astype(str)

    result = result.sort_values(["similarity", "stat_minutes_played_total"], ascending=[False, False])
    return {
        "recommendations": result.head(top_k).reset_index(drop=True),
        "query_vector": pd.DataFrame([query_vector]),
        "stage1_predictions": model_outputs["stage1_predictions"],
        "stage2_predictions": model_outputs["stage2_predictions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Similar player recommender deploy CLI.")
    parser.add_argument("--build-pool", action="store_true", help="Rebuild the Big Five candidate pool.")
    parser.add_argument("--input-csv", help="CSV containing source-league player rows.")
    parser.add_argument("--row-index", type=int, default=0, help="Row index in --input-csv to recommend for.")
    parser.add_argument("--destination-league", help="Destination Big Five league, e.g. Premier League.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-csv", help="Optional path to save recommendations.")
    args = parser.parse_args()

    if args.build_pool or not args.input_csv:
        pool = build_candidate_pool()
        print(f"Candidate pool saved to {DEFAULT_CANDIDATE_POOL}")
        print(f"Rows: {len(pool):,}")
        print(pool.groupby(["season_name", "league_name", "position_code"]).size().to_string())
        if not args.input_csv:
            return

    if not args.destination_league:
        raise SystemExit("--destination-league is required when --input-csv is provided.")

    data = pd.read_csv(args.input_csv, low_memory=False)
    if args.row_index < 0 or args.row_index >= len(data):
        raise SystemExit(f"--row-index must be between 0 and {len(data) - 1}")

    row = data.iloc[args.row_index].to_dict()
    result = recommend_similar_players(row, destination_league=args.destination_league, top_k=args.top_k)
    recommendations = result["recommendations"]

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved recommendations: {output_path}")

    cols = [
        col
        for col in [
            "player_name",
            "team_name",
            "league_name",
            "season_name",
            "position_code",
            "similarity",
            "stat_minutes_played_total",
            "candidate_minutes_confidence",
        ]
        if col in recommendations.columns
    ]
    print(recommendations[cols].to_string(index=False))


if __name__ == "__main__":
    main()
