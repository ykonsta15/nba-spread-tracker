from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging

from src.db.repository import Repository
from src.features.build_features import build_features_from_history
from src.models.baseline import evaluate_regression_and_ats, predict_margin, save_artifacts, time_split, train_model
from src.utils.config import settings
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    repo = Repository(settings.db_path)

    history = repo.historical_games_for_features()
    features_df, state = build_features_from_history(
        history,
        rolling_window=settings.rolling_window,
        k_factor=settings.k_factor,
    )
    trainable = features_df.dropna(subset=["actual_margin"]).copy()
    if trainable.empty:
        raise RuntimeError("No completed games available for training. Run backfill first.")

    odds = repo.latest_odds()
    trainable = trainable.merge(odds[["game_id", "spread_close", "spread_open"]], on="game_id", how="left")
    trainable = trainable.sort_values(["date", "game_id"]).reset_index(drop=True)

    train_df, test_df = time_split(trainable, test_frac=0.2)
    bundle = train_model(train_df)

    train_df["predicted_margin"] = predict_margin(bundle, train_df)
    test_df["predicted_margin"] = predict_margin(bundle, test_df)

    train_metrics = evaluate_regression_and_ats(train_df, threshold=settings.no_bet_threshold)
    test_metrics = evaluate_regression_and_ats(test_df, threshold=settings.no_bet_threshold)

    summary = {
        "threshold": settings.no_bet_threshold,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_start": str(train_df["date"].min()) if not train_df.empty else None,
        "train_end": str(train_df["date"].max()) if not train_df.empty else None,
    }

    save_artifacts(settings.artifacts_dir, bundle, state, summary)
    repo.insert_model_run(
        model_version=bundle.model_version,
        train_start=summary["train_start"],
        train_end=summary["train_end"],
        metrics=summary,
    )

    logger.info("Model version %s trained", bundle.model_version)
    train_mae = train_metrics.get("mae")
    test_mae = test_metrics.get("mae")
    train_mae_text = f"{train_mae:.3f}" if train_mae is not None else "NA"
    test_mae_text = f"{test_mae:.3f}" if test_mae is not None else "NA"
    print(
        f"Trained {bundle.model_version} | train MAE={train_mae_text} | "
        f"test MAE={test_mae_text}"
    )


if __name__ == "__main__":
    main()
