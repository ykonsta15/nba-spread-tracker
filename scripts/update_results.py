from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date
import logging

from src.data.nba_client import fetch_game_result
from src.db.repository import Repository, ResultRow, utc_now_iso
from src.models.baseline import ats_result_from_margin, units_from_ats_result
from src.utils.config import settings
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    repo = Repository(settings.db_path)
    pending = repo.pending_game_ids(through_date=date.today().isoformat())
    if not pending:
        print("No pending games to update.")
        return

    latest = repo.games_with_latest().set_index("game_id")
    updates: list[ResultRow] = []

    for game_id in pending:
        result = fetch_game_result(game_id)
        if not result:
            continue

        home_score = int(result["home_score"])
        away_score = int(result["away_score"])
        actual_margin = float(home_score - away_score)

        ats_result = None
        units = None
        if game_id in latest.index:
            row = latest.loc[game_id]
            spread = row.get("spread_close")
            pick = row.get("pick")
            if spread == spread and isinstance(pick, str) and pick in {"HOME_ATS", "AWAY_ATS", "NO_PICK"}:
                ats_result = ats_result_from_margin(actual_margin, float(spread), pick)
                units = units_from_ats_result(ats_result)

        updates.append(
            ResultRow(
                game_id=game_id,
                home_score=home_score,
                away_score=away_score,
                actual_margin=actual_margin,
                ats_result=ats_result,
                units=units,
                updated_at=utc_now_iso(),
            )
        )

    repo.upsert_results(updates)
    print(f"Updated {len(updates)} game results.")


if __name__ == "__main__":
    main()
