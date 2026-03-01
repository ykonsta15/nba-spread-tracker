# NBA Spread Baseline (Portfolio Project)

A portfolio-ready, end-to-end NBA spread prediction baseline with:
- reproducible SQLite pipeline
- timestamped pre-game predictions
- ATS pick tracking against closing spread
- Streamlit dashboard for slate + performance + game detail + model card

This project focuses on **defensible process and monitoring**, not guaranteed profitability.

## What this project does

1. Backfills NBA games and final scores by season.
2. Loads spread data from a pluggable odds provider:
   - default: real odds API if key exists, otherwise no odds stored
   - optional: TheOddsAPI if `THE_ODDS_API_KEY` is set.
3. Builds leakage-safe pre-game features (Elo/rest/back-to-back/rolling point differential).
4. Trains a ridge baseline to predict home margin (`home_score - away_score`).
5. Converts predicted margin into ATS picks with a no-bet threshold.
6. Updates results and ATS/unit outcomes as games complete.
7. Serves a Streamlit dashboard with tracking visuals and model card.

## Tech stack

- Python 3.10+
- SQLite
- `nba_api` for schedule/results
- Streamlit + Plotly dashboard
- scikit-learn baseline model

## Repository layout

- `src/data`: NBA and odds providers
- `src/features`: Elo + feature engineering
- `src/models`: baseline model + ATS scoring
- `src/db`: SQLite schema + repository
- `src/app`: Streamlit app
- `scripts`: reproducible pipeline commands
- `data/sample_odds.csv`: mock odds input
- `tests`: unit tests for core logic

## Setup (one command)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Run pipeline (commands)

```bash
python3 scripts/init_db.py
python3 scripts/backfill_season.py
# backfill_season defaults to current season (e.g., 2025-26 on March 1, 2026)
python3 scripts/train_model.py
python3 scripts/predict_today.py --date 2026-03-01 --days-ahead 7
python3 scripts/update_results.py
python3 scripts/run_daily.py --date 2026-03-01 --days-ahead 7 --odds-provider auto
# Optional: import historical closing spreads for past ATS metrics
python3 scripts/import_odds_csv.py --csv data/historical_odds.csv
python3 -m streamlit run src/app/app.py
```

## Expected output (examples)

`python3 scripts/init_db.py`
- `Database initialized at .../nba_spread.db`

`python3 scripts/backfill_season.py`
- `Backfilled <N> games for <CURRENT_SEASON>. Results: <N>. Odds quotes inserted: <M>`

`python3 scripts/train_model.py`
- `Trained ridge_<UTC_TIMESTAMP> | train MAE=<x.xxx> | test MAE=<y.yyy>`
- artifacts written:
  - `artifacts/model_latest.joblib`
  - `artifacts/model_metadata.json`
  - `artifacts/rating_state.json`

`python3 scripts/predict_today.py --date YYYY-MM-DD --days-ahead 7`
- `Predictions written for <K> game(s) from YYYY-MM-DD through YYYY-MM-DD using model ridge_<...>.`
- If no scheduled games:
  - `No scheduled games found for YYYY-MM-DD`

`python3 scripts/update_results.py`
- `Updated <R> game results.`
- If nothing pending:
  - `No pending games to update.`

## Dashboard pages

- **Home**:
  - Past Range and Upcoming Range filters (1 day, 7 days, 30 days, or custom up to 30 days)
  - upcoming/pending slate with game date, spread, predicted margin, edge, ATS pick
  - finished games with game date, result, ATS outcome, units
- **Performance**:
  - KPI cards (ATS record, units, win rate, MAE, sample)
  - cumulative units chart
  - rolling win rate chart
  - edge distribution
  - win rate by edge bucket
- **Game Detail**:
  - game-level prediction/result snapshot
- **Model Card**:
  - baseline method, features, evaluation, limits

## Odds provider modes

- Default (`auto`) behavior:
  - uses TheOddsAPI when `THE_ODDS_API_KEY` exists
  - otherwise stores no odds (real games still load); use `--odds-provider mock` only for demo data

Optional `.env`:

```bash
cp .env.example .env
# fill in THE_ODDS_API_KEY if desired
# optional: tune odds source
# THE_ODDS_REGIONS=us
# THE_ODDS_BOOKMAKERS=fanduel,draftkings,betmgm,caesars
```

## Backfill/train design choices

- Target: `actual_margin = home_score - away_score`
- Features:
  - Elo diff
  - rest days
  - back-to-back flags
  - rolling point differential (last 5)
  - home indicator
- Split: time-based (last 20% holdout)
- Leakage prevention: features are built chronologically from prior-state only

## ATS evaluation details

- Edge: `predicted_margin - spread_close`
- Pick:
  - `HOME_ATS` if edge > threshold
  - `AWAY_ATS` if edge < -threshold
  - `NO_PICK` otherwise
- Default no-bet threshold: `1.0` point (configurable via `NBA_NO_BET_THRESHOLD`)
- Units at -110:
  - Win: `+0.91`
  - Loss: `-1.00`
  - Push/No Action: `0.00`

## Screenshot instructions (for portfolio README updates)

1. Run app:
   ```bash
   python3 -m streamlit run src/app/app.py
   ```
2. Capture:
   - Home page (selected date slate)
   - Performance page (KPI + cumulative units)
   - Model card page
3. Save images to a folder like `docs/screenshots/`.
4. Add Markdown image links in README.

## Tests

```bash
python3 -m pytest -q
```

Covers:
- Elo update behavior
- feature builder chronological state usage
- ATS pick/result/unit scoring

## Limitations and next steps

- No injury/lineup/market microstructure features.
- Mock odds CSV can be sparse; games without spreads are skipped for ATS metrics.
- TheOddsAPI mapping is lightweight and may require richer team-name normalization.
- Future upgrades:
  - walk-forward retraining and calibration
  - richer features (travel, fatigue, lineup proxies)
  - better odds history capture (open/mid/close snapshots)
  - CI job for nightly update + dashboard snapshot export


### Historical spreads for past ATS
If `Spread (Close)` is empty for past games, import a real historical odds CSV:

```bash
python3 scripts/import_odds_csv.py --csv data/historical_odds.csv
```

CSV must include `spread_close` and either:
- `game_id`, or
- `date`, `home_team`, `away_team`

Optional columns: `spread_open`, `timestamp`.


## Daily automation
Use one command to refresh upcoming predictions and settle completed games:

```bash
python3 scripts/run_daily.py --date YYYY-MM-DD --days-ahead 7 --odds-provider auto
```

This does not run automatically by itself. To run daily without manual work, schedule it with cron/launchd or your hosting platform scheduler.


### macOS launchd (always-on local automation)
A launchd agent is included at `automation/com.codex.nba.daily.plist` and was installed to `~/Library/LaunchAgents/com.codex.nba.daily.plist`. It is configured for 7:00 AM local time daily.

Useful commands:
```bash
launchctl list | rg com.codex.nba.daily
launchctl print gui/$(id -u)/com.codex.nba.daily
launchctl kickstart -k gui/$(id -u)/com.codex.nba.daily
launchctl unload ~/Library/LaunchAgents/com.codex.nba.daily.plist
launchctl load ~/Library/LaunchAgents/com.codex.nba.daily.plist
```

Logs:
- `logs/launchd_daily.out.log`
- `logs/launchd_daily.err.log`
