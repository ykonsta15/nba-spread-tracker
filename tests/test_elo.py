from src.features.elo import update_elo


def test_elo_updates_are_zero_sum() -> None:
    home, away = update_elo(1500.0, 1500.0, home_win=True, k_factor=20.0, home_advantage=0.0)
    assert round((home - 1500.0) + (away - 1500.0), 10) == 0.0


def test_elo_home_win_increases_home_rating() -> None:
    home, away = update_elo(1500.0, 1500.0, home_win=True, k_factor=20.0, home_advantage=0.0)
    assert home > 1500.0
    assert away < 1500.0
