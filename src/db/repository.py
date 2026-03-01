from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any

import pandas as pd


@dataclass(frozen=True)
class GameRow:
    game_id: str
    date: str
    home_team: str
    away_team: str
    start_time: str | None
    season: str


@dataclass(frozen=True)
class OddsRow:
    game_id: str
    provider: str
    spread_close: float | None
    spread_open: float | None
    timestamp: str


@dataclass(frozen=True)
class PredictionRow:
    game_id: str
    model_version: str
    predicted_margin: float
    edge: float | None
    pick: str | None
    created_at: str


@dataclass(frozen=True)
class ResultRow:
    game_id: str
    home_score: int
    away_score: int
    actual_margin: float
    ats_result: str | None
    units: float | None
    updated_at: str


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_teams(self, teams: Iterable[tuple[int, str, str]]) -> None:
        conn = self._conn()
        try:
            conn.executemany(
                """
                INSERT INTO teams(team_id, name, abbr)
                VALUES (?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    name=excluded.name,
                    abbr=excluded.abbr
                """,
                list(teams),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_games(self, games: Iterable[GameRow]) -> None:
        payload = [(g.game_id, g.date, g.home_team, g.away_team, g.start_time, g.season) for g in games]
        if not payload:
            return
        conn = self._conn()
        try:
            conn.executemany(
                """
                INSERT INTO games(game_id, date, home_team, away_team, start_time, season)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    date=excluded.date,
                    home_team=excluded.home_team,
                    away_team=excluded.away_team,
                    start_time=excluded.start_time,
                    season=excluded.season
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    def insert_odds(self, rows: Iterable[OddsRow]) -> None:
        payload = [(r.game_id, r.provider, r.spread_close, r.spread_open, r.timestamp) for r in rows]
        if not payload:
            return
        conn = self._conn()
        try:
            conn.executemany(
                """
                INSERT INTO odds(game_id, provider, spread_close, spread_open, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    def insert_predictions(self, rows: Iterable[PredictionRow]) -> None:
        payload = [(r.game_id, r.model_version, r.predicted_margin, r.edge, r.pick, r.created_at) for r in rows]
        if not payload:
            return
        conn = self._conn()
        try:
            conn.executemany(
                """
                INSERT INTO predictions(game_id, model_version, predicted_margin, edge, pick, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_results(self, rows: Iterable[ResultRow]) -> None:
        payload = [
            (r.game_id, r.home_score, r.away_score, r.actual_margin, r.ats_result, r.units, r.updated_at)
            for r in rows
        ]
        if not payload:
            return
        conn = self._conn()
        try:
            conn.executemany(
                """
                INSERT INTO results(game_id, home_score, away_score, actual_margin, ats_result, units, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    home_score=excluded.home_score,
                    away_score=excluded.away_score,
                    actual_margin=excluded.actual_margin,
                    ats_result=excluded.ats_result,
                    units=excluded.units,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    def insert_model_run(
        self,
        model_version: str,
        train_start: str | None,
        train_end: str | None,
        metrics: dict[str, Any],
    ) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_runs(model_version, train_start, train_end, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_version,
                    train_start,
                    train_end,
                    json.dumps(metrics),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def latest_odds(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            query = """
            SELECT o.*
            FROM odds o
            INNER JOIN (
                SELECT game_id, MAX(timestamp) AS latest_ts
                FROM odds
                GROUP BY game_id
            ) x ON x.game_id = o.game_id AND x.latest_ts = o.timestamp
            """
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def latest_predictions(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            query = """
            SELECT p.*
            FROM predictions p
            INNER JOIN (
                SELECT game_id, MAX(created_at) AS latest_ts
                FROM predictions
                GROUP BY game_id
            ) x ON x.game_id = p.game_id AND x.latest_ts = p.created_at
            """
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def games_with_latest(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            query = """
            SELECT
                g.game_id,
                g.date,
                g.start_time,
                g.season,
                g.home_team,
                g.away_team,
                o.provider,
                o.spread_open,
                o.spread_close,
                p.model_version,
                p.predicted_margin,
                p.edge,
                p.pick,
                p.created_at,
                r.home_score,
                r.away_score,
                r.actual_margin,
                r.ats_result,
                r.units
            FROM games g
            LEFT JOIN (
                SELECT o1.*
                FROM odds o1
                INNER JOIN (
                    SELECT game_id, MAX(timestamp) AS latest_ts
                    FROM odds
                    GROUP BY game_id
                ) ox ON ox.game_id = o1.game_id AND ox.latest_ts = o1.timestamp
            ) o ON o.game_id = g.game_id
            LEFT JOIN (
                SELECT p1.*
                FROM predictions p1
                INNER JOIN (
                    SELECT game_id, MAX(created_at) AS latest_ts
                    FROM predictions
                    GROUP BY game_id
                ) px ON px.game_id = p1.game_id AND px.latest_ts = p1.created_at
            ) p ON p.game_id = g.game_id
            LEFT JOIN results r ON r.game_id = g.game_id
            ORDER BY g.date DESC, g.game_id DESC
            """
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def historical_games_for_features(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            query = """
            SELECT g.game_id, g.date, g.start_time, g.home_team, g.away_team, g.season,
                   r.home_score, r.away_score,
                   (r.home_score - r.away_score) AS actual_margin
            FROM games g
            LEFT JOIN results r ON r.game_id = g.game_id
            ORDER BY g.date ASC, COALESCE(g.start_time, g.date) ASC, g.game_id ASC
            """
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def get_games_by_date(self, game_date: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                """
                SELECT * FROM games WHERE date = ? ORDER BY start_time, game_id
                """,
                conn,
                params=[game_date],
            )
        finally:
            conn.close()

    def pending_game_ids(self, through_date: str | None = None) -> list[str]:
        conn = self._conn()
        try:
            if through_date is None:
                rows = conn.execute(
                    """
                    SELECT g.game_id
                    FROM games g
                    LEFT JOIN results r ON r.game_id = g.game_id
                    WHERE r.game_id IS NULL
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT g.game_id
                    FROM games g
                    LEFT JOIN results r ON r.game_id = g.game_id
                    WHERE r.game_id IS NULL AND g.date <= ?
                    """,
                    (through_date,),
                ).fetchall()
            return [row["game_id"] for row in rows]
        finally:
            conn.close()

    def get_model_versions(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query("SELECT * FROM model_runs ORDER BY created_at DESC", conn)
        finally:
            conn.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
