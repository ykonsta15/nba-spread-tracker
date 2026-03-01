from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import date
import logging

from src.data.nba_client import fetch_completed_games_for_season, infer_season
from src.data.odds_provider import build_odds_provider
from src.db.repository import GameRow, OddsRow, Repository, ResultRow, utc_now_iso
from src.utils.config import settings
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def normalize_season_label(season_arg: str | None) -> str:
    if not season_arg:
        return infer_season(date.today())

    raw = season_arg.strip()
    if "-" in raw:
        return raw
    year = int(raw)
    return f"{year - 1}-{str(year % 100).zfill(2)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill season games/results into SQLite")
    parser.add_argument("--season", required=False, help="Season as 2026 or 2025-26. Defaults to current season.")
    parser.add_argument("--odds-provider", choices=["auto", "mock", "theoddsapi", "none"], default="auto")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    season_label = normalize_season_label(args.season)

    repo = Repository(settings.db_path)
    season_df = fetch_completed_games_for_season(season_label)

    game_rows = [
        GameRow(
            game_id=str(r.game_id),
            date=str(r.date),
            home_team=str(r.home_team),
            away_team=str(r.away_team),
            start_time=r.start_time if r.start_time else None,
            season=str(r.season),
        )
        for r in season_df.itertuples(index=False)
    ]
    repo.upsert_games(game_rows)

    result_rows = [
        ResultRow(
            game_id=str(r.game_id),
            home_score=int(r.home_score),
            away_score=int(r.away_score),
            actual_margin=float(r.actual_margin),
            ats_result=None,
            units=None,
            updated_at=utc_now_iso(),
        )
        for r in season_df.itertuples(index=False)
    ]
    repo.upsert_results(result_rows)

    odds_provider = build_odds_provider(args.odds_provider, settings.mock_odds_csv)
    quotes = odds_provider.get_odds(season_df[["game_id", "date", "home_team", "away_team"]])
    repo.insert_odds(
        OddsRow(
            game_id=q.game_id,
            provider=q.provider,
            spread_close=q.spread_close,
            spread_open=q.spread_open,
            timestamp=q.timestamp,
        )
        for q in quotes
    )

    logger.info("Backfill complete: %d games, %d results, %d odds rows", len(game_rows), len(result_rows), len(quotes))
    print(
        f"Backfilled {len(game_rows)} games for {season_label}. Results: {len(result_rows)}. Odds quotes inserted: {len(quotes)}"
    )


if __name__ == "__main__":
    main()
