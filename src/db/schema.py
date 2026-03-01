from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    abbr TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    start_time TEXT,
    season TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    spread_close REAL,
    spread_open REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_game_id_ts ON odds(game_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_margin REAL NOT NULL,
    edge REAL,
    pick TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_game_id_ts ON predictions(game_id, created_at DESC);

CREATE TABLE IF NOT EXISTS results (
    game_id TEXT PRIMARY KEY,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    actual_margin REAL NOT NULL,
    ats_result TEXT,
    units REAL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_version TEXT PRIMARY KEY,
    train_start TEXT,
    train_end TEXT,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
