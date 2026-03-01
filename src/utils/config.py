from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("nba_spread.db")
    artifacts_dir: Path = Path("artifacts")
    mock_odds_csv: Path = Path("data/sample_odds.csv")
    no_bet_threshold: float = 1.0
    rolling_window: int = 5
    k_factor: float = 20.0


settings = Settings(
    db_path=Path(os.getenv("NBA_DB_PATH", "nba_spread.db")),
    artifacts_dir=Path(os.getenv("NBA_ARTIFACTS_DIR", "artifacts")),
    mock_odds_csv=Path(os.getenv("NBA_MOCK_ODDS_PATH", "data/sample_odds.csv")),
    no_bet_threshold=float(os.getenv("NBA_NO_BET_THRESHOLD", "1.0")),
    rolling_window=int(os.getenv("NBA_ROLLING_WINDOW", "5")),
    k_factor=float(os.getenv("NBA_ELO_K", "20")),
)
