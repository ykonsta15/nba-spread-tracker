from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.features.elo import update_elo


DEFAULT_ELO = 1500.0
HOME_ADVANTAGE_ELO = 100.0


@dataclass
class RatingState:
    elo: dict[str, float]
    last_game_date: dict[str, str]
    rolling_pd: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "elo": self.elo,
            "last_game_date": self.last_game_date,
            "rolling_pd": self.rolling_pd,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RatingState":
        return cls(
            elo={str(k): float(v) for k, v in payload.get("elo", {}).items()},
            last_game_date={str(k): str(v) for k, v in payload.get("last_game_date", {}).items()},
            rolling_pd={str(k): [float(x) for x in v] for k, v in payload.get("rolling_pd", {}).items()},
        )


def _rest_days(last_date: str | None, current_date: str) -> int:
    if last_date is None:
        return 7
    last = datetime.fromisoformat(last_date).date()
    current = datetime.fromisoformat(current_date).date()
    return max((current - last).days, 0)


def _rolling_avg(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def build_features_from_history(
    games_df: pd.DataFrame,
    rolling_window: int = 5,
    k_factor: float = 20.0,
    initial_state: RatingState | None = None,
) -> tuple[pd.DataFrame, RatingState]:
    """
    Build pre-game features in chronological order.
    Expects columns: game_id,date,home_team,away_team,home_score,away_score,actual_margin (optional).
    """
    df = games_df.copy()
    if df.empty:
        return pd.DataFrame(), RatingState({}, {}, {})

    df = df.sort_values(["date", "start_time", "game_id"], na_position="last").reset_index(drop=True)

    if initial_state:
        elo = defaultdict(lambda: DEFAULT_ELO, initial_state.elo)
        last_game_date = dict(initial_state.last_game_date)
        rolling_pd = defaultdict(lambda: deque(maxlen=rolling_window))
        for team, vals in initial_state.rolling_pd.items():
            for v in vals[-rolling_window:]:
                rolling_pd[team].append(float(v))
    else:
        elo = defaultdict(lambda: DEFAULT_ELO)
        last_game_date = {}
        rolling_pd = defaultdict(lambda: deque(maxlen=rolling_window))

    feature_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        game_date = str(row["date"])

        home_elo = float(elo[home])
        away_elo = float(elo[away])

        home_rest = _rest_days(last_game_date.get(home), game_date)
        away_rest = _rest_days(last_game_date.get(away), game_date)

        home_roll = _rolling_avg(list(rolling_pd[home]))
        away_roll = _rolling_avg(list(rolling_pd[away]))

        feat = {
            "game_id": str(row["game_id"]),
            "date": game_date,
            "home_team": home,
            "away_team": away,
            "elo_diff": home_elo - away_elo,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_b2b": int(home_rest <= 1),
            "away_b2b": int(away_rest <= 1),
            "home_rolling_pd": home_roll,
            "away_rolling_pd": away_roll,
            "rolling_pd_diff": home_roll - away_roll,
            "home_indicator": 1,
        }

        if pd.notna(row.get("actual_margin")):
            feat["actual_margin"] = float(row["actual_margin"])

        feature_rows.append(feat)

        # Update state only when game result exists.
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            margin = float(home_score - away_score)
            home_win = margin > 0

            new_home, new_away = update_elo(
                home_rating=home_elo,
                away_rating=away_elo,
                home_win=home_win,
                k_factor=k_factor,
                home_advantage=HOME_ADVANTAGE_ELO,
            )
            elo[home] = new_home
            elo[away] = new_away

            rolling_pd[home].append(margin)
            rolling_pd[away].append(-margin)

            last_game_date[home] = game_date
            last_game_date[away] = game_date

    out = pd.DataFrame(feature_rows)
    state = RatingState(
        elo={team: float(v) for team, v in elo.items()},
        last_game_date=last_game_date,
        rolling_pd={team: list(vals) for team, vals in rolling_pd.items()},
    )
    return out, state


def build_features_for_upcoming_games(
    upcoming_df: pd.DataFrame,
    state: RatingState,
    rolling_window: int = 5,
) -> pd.DataFrame:
    if upcoming_df.empty:
        return pd.DataFrame()

    clean_state = RatingState(
        elo=dict(state.elo),
        last_game_date=dict(state.last_game_date),
        rolling_pd={k: list(v)[-rolling_window:] for k, v in state.rolling_pd.items()},
    )
    merged, _ = build_features_from_history(
        games_df=upcoming_df,
        rolling_window=rolling_window,
        initial_state=clean_state,
    )
    return merged


def feature_columns() -> list[str]:
    return [
        "elo_diff",
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
        "home_rolling_pd",
        "away_rolling_pd",
        "rolling_pd_diff",
        "home_indicator",
    ]
