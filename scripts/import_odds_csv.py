from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

import pandas as pd

from src.db.repository import OddsRow, Repository, utc_now_iso
from src.utils.config import settings
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import historical odds from CSV into odds table")
    parser.add_argument("--csv", required=True, help="Path to odds CSV")
    parser.add_argument("--provider", default="historical_csv", help="Provider label stored in DB")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    source = pd.read_csv(csv_path, dtype={"game_id": str})
    if source.empty:
        print("CSV is empty. No odds imported.")
        return

    required = {"spread_close"}
    if not required.issubset(source.columns):
        raise ValueError("CSV must include at least: spread_close")

    repo = Repository(settings.db_path)
    games = repo.games_with_latest()[["game_id", "date", "home_team", "away_team"]].copy()
    games["date"] = games["date"].astype(str)

    if "game_id" in source.columns and source["game_id"].notna().any():
        merged = source.merge(games, on="game_id", how="inner", suffixes=("", "_db"))
    else:
        needed = {"date", "home_team", "away_team"}
        if not needed.issubset(source.columns):
            raise ValueError("CSV needs game_id OR (date, home_team, away_team) columns for matching.")
        source["date"] = source["date"].astype(str)
        merged = source.merge(games, on=["date", "home_team", "away_team"], how="inner", suffixes=("", "_db"))

    if merged.empty:
        print("No matching games found between CSV and DB.")
        return

    imported = 0
    rows: list[OddsRow] = []
    for r in merged.itertuples(index=False):
        spread_close = getattr(r, "spread_close", None)
        if spread_close is None or pd.isna(spread_close):
            continue
        spread_open = getattr(r, "spread_open", None)
        timestamp = getattr(r, "timestamp", None)
        rows.append(
            OddsRow(
                game_id=str(getattr(r, "game_id")),
                provider=args.provider,
                spread_close=float(spread_close),
                spread_open=float(spread_open) if spread_open is not None and not pd.isna(spread_open) else None,
                timestamp=str(timestamp) if timestamp and not pd.isna(timestamp) else utc_now_iso(),
            )
        )
        imported += 1

    repo.insert_odds(rows)
    print(f"Imported {imported} odds rows from {csv_path}.")


if __name__ == "__main__":
    main()
