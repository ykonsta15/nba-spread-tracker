from __future__ import annotations

from datetime import date, datetime
import logging

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams as teams_static

logger = logging.getLogger(__name__)


def load_teams() -> list[tuple[int, str, str]]:
    rows = teams_static.get_teams()
    return [(int(r["id"]), str(r["full_name"]), str(r["abbreviation"])) for r in rows]


def team_id_to_abbr() -> dict[int, str]:
    return {int(team["id"]): str(team["abbreviation"]) for team in teams_static.get_teams()}


def fetch_completed_games_for_season(season: str) -> pd.DataFrame:
    """
    Returns one row per game with finalized scores from a given NBA season (e.g. '2023-24').
    """
    logger.info("Fetching completed NBA games for season %s", season)
    endpoint = leaguegamefinder.LeagueGameFinder(
        league_id_nullable="00",
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    raw = endpoint.get_data_frames()[0]
    if raw.empty:
        return raw

    raw = raw[raw["WL"].notna()].copy()
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"]).dt.date
    raw["is_home"] = raw["MATCHUP"].str.contains("vs.")

    home = raw[raw["is_home"]].copy()
    away = raw[~raw["is_home"]].copy()

    merged = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_home", "_away"),
        how="inner",
    )
    if merged.empty:
        return merged

    out = pd.DataFrame(
        {
            "game_id": merged["GAME_ID"],
            "date": merged["GAME_DATE_home"].astype(str),
            "home_team": merged["TEAM_ABBREVIATION_home"],
            "away_team": merged["TEAM_ABBREVIATION_away"],
            "home_score": merged["PTS_home"].astype(int),
            "away_score": merged["PTS_away"].astype(int),
            "start_time": None,
            "season": season,
        }
    )
    out["actual_margin"] = out["home_score"] - out["away_score"]
    out = out.sort_values(["date", "game_id"]).reset_index(drop=True)
    logger.info("Fetched %d completed games", len(out))
    return out


def fetch_schedule_for_date(target_date: date) -> pd.DataFrame:
    """
    Returns games scheduled for the target date using scoreboard endpoint.
    """
    logger.info("Fetching schedule for %s", target_date)
    date_str = target_date.strftime("%m/%d/%Y")
    endpoint = scoreboardv2.ScoreboardV2(game_date=date_str, league_id="00", day_offset=0)
    game_df = endpoint.game_header.get_data_frame()
    if game_df.empty:
        return game_df

    mapping = team_id_to_abbr()
    rows = []
    for _, row in game_df.iterrows():
        rows.append(
            {
                "game_id": str(row["GAME_ID"]),
                "date": target_date.isoformat(),
                "home_team": mapping.get(int(row["HOME_TEAM_ID"]), "UNK"),
                "away_team": mapping.get(int(row["VISITOR_TEAM_ID"]), "UNK"),
                "start_time": pd.to_datetime(row["GAME_DATE_EST"]).isoformat() if row["GAME_DATE_EST"] else None,
                "season": infer_season(target_date),
                "status_text": row.get("GAME_STATUS_TEXT", ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_game_result(game_id: str) -> dict[str, int | str] | None:
    """
    Fetches result for a single game_id. Returns None if not found/final.

    NBA Stats currently ignores game_id filter in LeagueGameFinder, so we filter
    client-side to avoid returning the wrong game.
    """
    endpoint = leaguegamefinder.LeagueGameFinder(
        league_id_nullable="00",
    )
    raw = endpoint.get_data_frames()[0]
    if raw.empty:
        return None

    raw = raw[raw["GAME_ID"].astype(str) == str(game_id)].copy()
    if raw.empty:
        return None

    raw = raw[raw["WL"].notna()].copy()
    if len(raw) < 2:
        return None

    raw["is_home"] = raw["MATCHUP"].str.contains("vs.")
    home = raw[raw["is_home"]]
    away = raw[~raw["is_home"]]
    if home.empty or away.empty:
        return None

    home_row = home.iloc[0]
    away_row = away.iloc[0]
    return {
        "game_id": game_id,
        "home_team": str(home_row["TEAM_ABBREVIATION"]),
        "away_team": str(away_row["TEAM_ABBREVIATION"]),
        "home_score": int(home_row["PTS"]),
        "away_score": int(away_row["PTS"]),
    }


def infer_season(target_date: date | datetime) -> str:
    d = target_date.date() if isinstance(target_date, datetime) else target_date
    if d.month >= 10:
        return f"{d.year}-{str((d.year + 1) % 100).zfill(2)}"
    return f"{d.year - 1}-{str(d.year % 100).zfill(2)}"
