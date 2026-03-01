from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import date, datetime, timedelta
import logging

import pandas as pd

from src.data.nba_client import fetch_schedule_for_date
from src.data.odds_provider import build_odds_provider
from src.db.repository import GameRow, OddsRow, PredictionRow, Repository, utc_now_iso
from src.features.build_features import build_features_for_upcoming_games
from src.models.baseline import ats_pick, load_artifacts, predict_margin
from src.utils.config import settings
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NBA spread edges for a date range")
    parser.add_argument("--date", default=date.today().isoformat(), help="Range start date, YYYY-MM-DD")
    parser.add_argument("--days-ahead", type=int, default=7, help="How many days ahead to include (max 30)")
    parser.add_argument("--odds-provider", choices=["auto", "mock", "theoddsapi", "none"], default="auto")
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def main() -> None:
    setup_logging()
    args = parse_args()
    start_date = datetime.fromisoformat(args.date).date()
    days_ahead = max(0, min(int(args.days_ahead), 30))
    end_date = start_date + timedelta(days=days_ahead)

    repo = Repository(settings.db_path)
    bundle, state, _ = load_artifacts(settings.artifacts_dir)

    schedule_frames: list[pd.DataFrame] = []
    for i in range(days_ahead + 1):
        target_date = start_date + timedelta(days=i)
        daily = fetch_schedule_for_date(target_date)
        if not daily.empty:
            schedule_frames.append(daily)

    if not schedule_frames:
        print(f"No scheduled games found from {start_date.isoformat()} through {end_date.isoformat()}")
        return

    schedule_df = pd.concat(schedule_frames, ignore_index=True)

    repo.upsert_games(
        GameRow(
            game_id=str(r.game_id),
            date=str(r.date),
            home_team=str(r.home_team),
            away_team=str(r.away_team),
            start_time=str(r.start_time) if r.start_time else None,
            season=str(r.season),
        )
        for r in schedule_df.itertuples(index=False)
    )

    odds_provider = build_odds_provider(args.odds_provider, settings.mock_odds_csv)
    quotes = odds_provider.get_odds(schedule_df[["game_id", "date", "home_team", "away_team"]])
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

    latest = repo.games_with_latest()
    latest["game_date"] = pd.to_datetime(latest["date"]).dt.date

    in_window = latest[(latest["game_date"] >= start_date) & (latest["game_date"] <= end_date)].copy()
    pending = in_window[in_window["home_score"].isna()].copy()

    if pending.empty:
        print(f"No pending games to predict from {start_date.isoformat()} through {end_date.isoformat()}.")
        return

    feature_input = pending[["game_id", "date", "start_time", "home_team", "away_team"]].copy()
    feature_input["home_score"] = None
    feature_input["away_score"] = None
    feature_input["actual_margin"] = None

    feat_df = build_features_for_upcoming_games(feature_input, state, rolling_window=settings.rolling_window)
    feat_df["predicted_margin"] = predict_margin(bundle, feat_df)

    merged = feat_df.merge(pending[["game_id", "spread_close"]], on="game_id", how="left")
    merged["edge"] = merged["predicted_margin"] - merged["spread_close"]
    merged.loc[merged["spread_close"].isna(), "edge"] = None
    merged["pick"] = merged["edge"].apply(
        lambda e: ats_pick(parsed, settings.no_bet_threshold) if (parsed := _safe_float(e)) is not None else "NO_PICK"
    )

    now = utc_now_iso()
    repo.insert_predictions(
        PredictionRow(
            game_id=str(r.game_id),
            model_version=bundle.model_version,
            predicted_margin=float(r.predicted_margin),
            edge=_safe_float(r.edge),
            pick=str(r.pick),
            created_at=now,
        )
        for r in merged.itertuples(index=False)
    )

    print(
        f"Predictions written for {len(merged)} game(s) from {start_date.isoformat()} through {end_date.isoformat()} "
        f"using model {bundle.model_version}."
    )


if __name__ == "__main__":
    main()
