from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from nba_api.stats.static import teams as teams_static

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OddsQuote:
    game_id: str
    provider: str
    spread_close: float | None
    spread_open: float | None
    timestamp: str


class OddsProvider:
    def get_odds(self, games: pd.DataFrame) -> list[OddsQuote]:
        raise NotImplementedError


class NoOddsProvider(OddsProvider):
    def get_odds(self, games: pd.DataFrame) -> list[OddsQuote]:
        logger.info("No odds provider configured; skipping odds for %d game(s)", len(games))
        return []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _team_alias_maps() -> tuple[dict[str, str], dict[str, set[str]]]:
    alias_to_abbr: dict[str, str] = {}
    abbr_to_aliases: dict[str, set[str]] = {}

    for team in teams_static.get_teams():
        abbr = str(team["abbreviation"]).upper()
        aliases = {
            str(team.get("abbreviation", "")),
            str(team.get("full_name", "")),
            str(team.get("nickname", "")),
            str(team.get("city", "")),
            f"{team.get('city', '')} {team.get('nickname', '')}".strip(),
        }

        norm_aliases = {_norm(a) for a in aliases if a}
        abbr_to_aliases[abbr] = norm_aliases
        for a in norm_aliases:
            alias_to_abbr[a] = abbr

    # Common alt spellings
    alias_to_abbr[_norm("LA Clippers")] = "LAC"
    alias_to_abbr[_norm("LA Lakers")] = "LAL"

    return alias_to_abbr, abbr_to_aliases


_ALIAS_TO_ABBR, _ABBR_TO_ALIASES = _team_alias_maps()


def _to_abbr(name_or_abbr: str) -> str | None:
    if not name_or_abbr:
        return None
    raw = str(name_or_abbr).strip()
    upper = raw.upper()
    if len(upper) <= 4 and upper in _ABBR_TO_ALIASES:
        return upper
    return _ALIAS_TO_ABBR.get(_norm(raw))


def _find_event_for_game(payload: list[dict], home_abbr: str, away_abbr: str) -> dict | None:
    for event in payload:
        ev_home = _to_abbr(str(event.get("home_team", "")))
        ev_away = _to_abbr(str(event.get("away_team", "")))
        if ev_home == home_abbr and ev_away == away_abbr:
            return event
    return None


def _extract_home_spread(event: dict, home_abbr: str) -> float | None:
    points: list[float] = []
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if str(market.get("key", "")) != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                out_abbr = _to_abbr(str(outcome.get("name", "")))
                if out_abbr == home_abbr:
                    point = outcome.get("point")
                    if point is not None:
                        points.append(float(point))
    if not points:
        return None
    points.sort()
    mid = len(points) // 2
    if len(points) % 2 == 1:
        return points[mid]
    return (points[mid - 1] + points[mid]) / 2.0


class MockCsvOddsProvider(OddsProvider):
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def get_odds(self, games: pd.DataFrame) -> list[OddsQuote]:
        if not self.csv_path.exists():
            logger.warning("Mock odds CSV not found at %s", self.csv_path)
            return []

        odds_df = pd.read_csv(self.csv_path, dtype={"game_id": str})
        if odds_df.empty:
            return []

        expected_cols = {"spread_close"}
        missing = expected_cols - set(odds_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in mock odds CSV: {missing}")

        quotes: list[OddsQuote] = []
        for _, game in games.iterrows():
            game_id = str(game["game_id"])
            match = odds_df[odds_df.get("game_id", pd.Series(dtype=str)).astype(str) == game_id]

            if match.empty and {"date", "home_team", "away_team"}.issubset(odds_df.columns):
                match = odds_df[
                    (odds_df["date"].astype(str) == str(game["date"]))
                    & (odds_df["home_team"].astype(str) == str(game["home_team"]))
                    & (odds_df["away_team"].astype(str) == str(game["away_team"]))
                ]
            if match.empty:
                continue

            row = match.iloc[0]
            ts = str(row.get("timestamp") or datetime.now(timezone.utc).isoformat())
            quotes.append(
                OddsQuote(
                    game_id=game_id,
                    provider=str(row.get("provider", "mock_csv")),
                    spread_close=float(row["spread_close"]) if pd.notna(row["spread_close"]) else None,
                    spread_open=float(row["spread_open"]) if pd.notna(row.get("spread_open")) else None,
                    timestamp=ts,
                )
            )

        logger.info("Loaded %d odds quotes from mock CSV", len(quotes))
        return quotes


class TheOddsApiProvider(OddsProvider):
    BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")

    def get_odds(self, games: pd.DataFrame) -> list[OddsQuote]:
        if not self.api_key:
            logger.warning("THE_ODDS_API_KEY missing; skipping TheOddsAPI provider")
            return []

        regions = os.getenv("THE_ODDS_REGIONS", "us")
        bookmakers = os.getenv("THE_ODDS_BOOKMAKERS", "fanduel,draftkings,betmgm,caesars")

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "bookmakers": bookmakers,
            "markets": "spreads",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        response = requests.get(self.BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        quotes: list[OddsQuote] = []
        for _, game in games.iterrows():
            game_id = str(game["game_id"])
            home_abbr = _to_abbr(str(game["home_team"])) or str(game["home_team"]).upper()
            away_abbr = _to_abbr(str(game["away_team"])) or str(game["away_team"]).upper()

            match_event = _find_event_for_game(payload, home_abbr=home_abbr, away_abbr=away_abbr)
            if not match_event:
                continue

            spread_close = _extract_home_spread(match_event, home_abbr=home_abbr)
            quotes.append(
                OddsQuote(
                    game_id=game_id,
                    provider="theoddsapi_consensus",
                    spread_close=spread_close,
                    spread_open=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        logger.info("Loaded %d odds quotes from TheOddsAPI", len(quotes))
        return quotes


def build_odds_provider(kind: str, mock_csv_path: Path) -> OddsProvider:
    load_dotenv()
    key_exists = bool(os.getenv("THE_ODDS_API_KEY"))
    if kind == "theoddsapi":
        return TheOddsApiProvider()
    if kind == "mock":
        return MockCsvOddsProvider(mock_csv_path)
    if kind == "none":
        return NoOddsProvider()
    if kind == "auto":
        return TheOddsApiProvider() if key_exists else NoOddsProvider()
    return NoOddsProvider()
