# Similar Player Recommender Deploy Artifacts

This folder contains the deployable similar-player recommendation pipeline.

## Final Recommendation Rule

For a source-league player and a selected destination league:

1. Run Stage1 and Stage2 through `predict_pipeline.py`.
2. Build the player's predicted post-transfer performance vector from `final_after_pred`.
3. Filter candidates to:
   - same `position_code`
   - selected `destination_league`
   - latest Big Five season in `big_five_candidate_pool.csv`
4. Compare the query vector with candidate actual performance vectors using:
   - `StandardScaler`
   - `cosine_similarity`
   - equal feature weights
5. Return the top-k most similar players.

Candidate minutes are not hard-filtered in the final deploy setting. Instead, the output includes `candidate_minutes_confidence` so the API/UI can display reliability.

## Files

- `predict_similar_players.py`
  - Main deploy code.
  - Exposes `recommend_similar_players(...)`.
  - Can also rebuild `big_five_candidate_pool.csv` when executed directly.

- `similar_player_config.json`
  - Recommendation settings, position-specific vectors, candidate league list.

- `big_five_candidate_pool.csv`
  - Latest-season Big Five candidate pool.
  - Generated from `sportmonks/data/sportmonks_player_season/sportmonks_player_season_all.csv`.

- `requirements.txt`
  - Minimal Python dependencies for the recommender layer.

## Dependencies

This recommender depends on the Stage1/Stage2 deploy artifacts in the project root:

- `predict_pipeline.py`
- `stage1_deploy_artifacts/`
- `stage2_deploy_artifacts/`

## Usage

```python
import pandas as pd
from similar_player_deploy_artifacts.predict_similar_players import recommend_similar_players

source_players = pd.read_csv("stage2/sportmonks_player_season_source_leagues_player_season.csv")
player_row = source_players.iloc[0].to_dict()

result = recommend_similar_players(
    player_row,
    destination_league="Premier League",
    top_k=5,
)

recommendations = result["recommendations"]
query_vector = result["query_vector"]
stage2_predictions = result["stage2_predictions"]
```

## Main Output

`result["recommendations"]` includes:

- candidate metadata: `player_id`, `player_name`, `team_name`, `league_name`, `season_name`
- `similarity`
- `candidate_minutes_confidence`
- position-specific query/candidate/difference columns

Higher `similarity` means the candidate's current Big Five performance profile is closer to the query player's predicted post-transfer profile.
