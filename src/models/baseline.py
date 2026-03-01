from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import feature_columns, RatingState


@dataclass(frozen=True)
class TrainedBundle:
    model: Pipeline
    model_version: str
    feature_cols: list[str]


def make_model(alpha: float = 3.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    cut_idx = max(int(len(df) * (1 - test_frac)), 1)
    train = df.iloc[:cut_idx].copy()
    test = df.iloc[cut_idx:].copy()
    return train, test


def ats_pick(edge: float | None, threshold: float = 1.0) -> str:
    if edge is None or np.isnan(edge):
        return "NO_PICK"
    if abs(edge) < threshold:
        return "NO_PICK"
    return "HOME_ATS" if edge > 0 else "AWAY_ATS"


def ats_result_from_margin(actual_margin: float, spread_close: float, pick: str) -> str:
    if pick == "NO_PICK":
        return "NO_ACTION"

    ats_margin_home = actual_margin + spread_close
    if abs(ats_margin_home) < 1e-9:
        return "PUSH"

    home_covers = ats_margin_home > 0
    if pick == "HOME_ATS":
        return "WIN" if home_covers else "LOSS"
    if pick == "AWAY_ATS":
        return "WIN" if not home_covers else "LOSS"
    return "NO_ACTION"


def units_from_ats_result(result: str) -> float:
    if result == "WIN":
        return 0.91
    if result == "LOSS":
        return -1.0
    return 0.0


def evaluate_regression_and_ats(
    df: pd.DataFrame,
    threshold: float = 1.0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if df.empty:
        return {
            "sample_size": 0,
            "mae": None,
            "ats_bets": 0,
            "ats_wins": 0,
            "ats_losses": 0,
            "ats_pushes": 0,
            "ats_win_rate": None,
            "units": 0.0,
        }

    metrics["sample_size"] = int(len(df))
    metrics["mae"] = float(mean_absolute_error(df["actual_margin"], df["predicted_margin"]))

    ats_df = df.dropna(subset=["spread_close"]).copy()
    ats_df["edge"] = ats_df["predicted_margin"] - ats_df["spread_close"]
    ats_df["pick"] = ats_df["edge"].apply(lambda x: ats_pick(float(x), threshold=threshold))
    ats_df = ats_df[ats_df["pick"] != "NO_PICK"].copy()

    if ats_df.empty:
        metrics.update(
            {
                "ats_bets": 0,
                "ats_wins": 0,
                "ats_losses": 0,
                "ats_pushes": 0,
                "ats_win_rate": None,
                "units": 0.0,
            }
        )
        return metrics

    ats_df["ats_result"] = ats_df.apply(
        lambda r: ats_result_from_margin(float(r["actual_margin"]), float(r["spread_close"]), str(r["pick"])),
        axis=1,
    )
    ats_df["units"] = ats_df["ats_result"].map(units_from_ats_result)

    wins = int((ats_df["ats_result"] == "WIN").sum())
    losses = int((ats_df["ats_result"] == "LOSS").sum())
    pushes = int((ats_df["ats_result"] == "PUSH").sum())

    denom = wins + losses
    metrics.update(
        {
            "ats_bets": int(len(ats_df)),
            "ats_wins": wins,
            "ats_losses": losses,
            "ats_pushes": pushes,
            "ats_win_rate": float(wins / denom) if denom else None,
            "units": float(ats_df["units"].sum()),
            "avg_edge": float(ats_df["edge"].abs().mean()) if not ats_df.empty else None,
        }
    )
    return metrics


def train_model(train_df: pd.DataFrame) -> TrainedBundle:
    cols = feature_columns()
    model = make_model()
    model.fit(train_df[cols], train_df["actual_margin"])

    version = f"ridge_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return TrainedBundle(model=model, model_version=version, feature_cols=cols)


def predict_margin(bundle: TrainedBundle, features_df: pd.DataFrame) -> np.ndarray:
    if features_df.empty:
        return np.array([])
    return bundle.model.predict(features_df[bundle.feature_cols])


def save_artifacts(
    artifacts_dir: Path,
    bundle: TrainedBundle,
    state: RatingState,
    train_summary: dict[str, Any],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.model, artifacts_dir / "model_latest.joblib")

    metadata = {
        "model_version": bundle.model_version,
        "feature_cols": bundle.feature_cols,
        "train_summary": train_summary,
    }
    (artifacts_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts_dir / "rating_state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def load_artifacts(artifacts_dir: Path) -> tuple[TrainedBundle, RatingState, dict[str, Any]]:
    model = joblib.load(artifacts_dir / "model_latest.joblib")
    metadata = json.loads((artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
    state_dict = json.loads((artifacts_dir / "rating_state.json").read_text(encoding="utf-8"))

    bundle = TrainedBundle(
        model=model,
        model_version=str(metadata["model_version"]),
        feature_cols=list(metadata["feature_cols"]),
    )
    state = RatingState.from_dict(state_dict)
    return bundle, state, metadata
