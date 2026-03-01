from src.models.baseline import ats_pick, ats_result_from_margin, units_from_ats_result


def test_no_bet_threshold() -> None:
    assert ats_pick(0.5, threshold=1.0) == "NO_PICK"
    assert ats_pick(1.5, threshold=1.0) == "HOME_ATS"
    assert ats_pick(-1.5, threshold=1.0) == "AWAY_ATS"


def test_ats_and_units() -> None:
    result = ats_result_from_margin(actual_margin=6, spread_close=-3.5, pick="HOME_ATS")
    assert result == "WIN"
    assert units_from_ats_result(result) == 0.91

    result2 = ats_result_from_margin(actual_margin=2, spread_close=-3.5, pick="HOME_ATS")
    assert result2 == "LOSS"
    assert units_from_ats_result(result2) == -1.0
