from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nba_client import load_teams
from src.db.repository import Repository
from src.db.schema import init_db
from src.utils.config import settings
from src.utils.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    init_db(settings.db_path)
    repo = Repository(settings.db_path)
    repo.upsert_teams(load_teams())
    print(f"Database initialized at {settings.db_path.resolve()}")


if __name__ == "__main__":
    main()
