import pandas as pd

from src.features.build_features import build_features_from_history


def test_feature_builder_uses_pregame_state_only() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "1",
                "date": "2024-01-01",
                "start_time": None,
                "home_team": "AAA",
                "away_team": "BBB",
                "home_score": 110,
                "away_score": 100,
                "actual_margin": 10,
            },
            {
                "game_id": "2",
                "date": "2024-01-03",
                "start_time": None,
                "home_team": "AAA",
                "away_team": "BBB",
                "home_score": 99,
                "away_score": 101,
                "actual_margin": -2,
            },
        ]
    )

    feats, _ = build_features_from_history(games, rolling_window=5, k_factor=20.0)
    first = feats.iloc[0]
    second = feats.iloc[1]

    assert first["elo_diff"] == 0
    assert second["elo_diff"] != 0
    assert second["home_rest_days"] == 2
