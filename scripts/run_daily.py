from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import date
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily update workflow")
    parser.add_argument("--date", default=date.today().isoformat(), help="Start date YYYY-MM-DD")
    parser.add_argument("--days-ahead", type=int, default=7, help="Prediction horizon (max 30)")
    parser.add_argument("--odds-provider", choices=["auto", "mock", "theoddsapi", "none"], default="auto")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    run_cmd([
        sys.executable,
        "scripts/predict_today.py",
        "--date",
        args.date,
        "--days-ahead",
        str(args.days_ahead),
        "--odds-provider",
        args.odds_provider,
    ])
    run_cmd([sys.executable, "scripts/update_results.py"])

    print("Daily workflow complete.")


if __name__ == "__main__":
    main()
