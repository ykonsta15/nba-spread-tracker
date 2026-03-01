from __future__ import annotations

from math import pow


def expected_score(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    return 1.0 / (1.0 + pow(10.0, ((rating_b - (rating_a + home_advantage)) / 400.0)))


def update_elo(
    home_rating: float,
    away_rating: float,
    home_win: bool,
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
) -> tuple[float, float]:
    exp_home = expected_score(home_rating, away_rating, home_advantage=home_advantage)
    actual_home = 1.0 if home_win else 0.0

    new_home = home_rating + k_factor * (actual_home - exp_home)
    new_away = away_rating + k_factor * ((1.0 - actual_home) - (1.0 - exp_home))
    return new_home, new_away
