from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date, timedelta
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.db.repository import Repository
from src.utils.config import settings

st.set_page_config(page_title="NBA Spread Baseline", page_icon="🏀", layout="wide")

repo = Repository(settings.db_path)


def _model_pick_from_margin(predicted_margin: float | None) -> str | None:
    if predicted_margin is None or pd.isna(predicted_margin):
        return None
    return "HOME_ML_LEAN" if float(predicted_margin) > 0 else "AWAY_ML_LEAN"


def _ats_pick_display(spread_close: float | None, pick: str | None) -> str:
    if spread_close is None or pd.isna(spread_close):
        return "NO_SPREAD"
    if not pick or pd.isna(pick):
        return "NO_PICK"
    return str(pick)


def _load_all() -> pd.DataFrame:
    df = repo.games_with_latest()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["game_date"] = df["date"].dt.date
    df["status"] = df["home_score"].apply(lambda x: "Final" if pd.notna(x) else "Pending")
    df["matchup"] = df["away_team"] + " @ " + df["home_team"]
    df["model_pick"] = df["predicted_margin"].apply(_model_pick_from_margin)
    df["ats_pick_display"] = df.apply(lambda r: _ats_pick_display(r.get("spread_close"), r.get("pick")), axis=1)
    return df


def kpi_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    done = df[(df["status"] == "Final") & (df["ats_result"].notna())].copy()
    if done.empty:
        return {
            "record": "0-0-0",
            "units": 0.0,
            "win_rate": 0.0,
            "mae": 0.0,
            "sample": 0,
            "avg_edge": 0.0,
        }

    wins = int((done["ats_result"] == "WIN").sum())
    losses = int((done["ats_result"] == "LOSS").sum())
    pushes = int((done["ats_result"] == "PUSH").sum())
    denom = wins + losses

    mae_df = done.dropna(subset=["predicted_margin", "actual_margin"])
    mae = float((mae_df["predicted_margin"] - mae_df["actual_margin"]).abs().mean()) if not mae_df.empty else 0.0

    return {
        "record": f"{wins}-{losses}-{pushes}",
        "units": float(done["units"].fillna(0).sum()),
        "win_rate": float(wins / denom) if denom else 0.0,
        "mae": mae,
        "sample": int(len(done)),
        "avg_edge": float(done["edge"].abs().mean()) if done["edge"].notna().any() else 0.0,
    }


def add_edge_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    abs_edge = out["edge"].abs()
    out["edge_bucket"] = pd.cut(
        abs_edge,
        bins=[-0.01, 1, 2, 3, 100],
        labels=["0-1", "1-2", "2-3", "3+"],
    )
    return out


def _past_range_control(key: str) -> tuple[date, date]:
    preset = st.selectbox(
        "Past Range",
        ["1 day", "7 days", "30 days", "Custom (max 30 days)"],
        index=1,
        key=f"past_preset_{key}",
    )
    today = date.today()

    if preset == "1 day":
        return today, today
    if preset == "7 days":
        return today - timedelta(days=6), today
    if preset == "30 days":
        return today - timedelta(days=29), today

    start = st.date_input("Past Start", value=today - timedelta(days=6), key=f"past_start_{key}")
    end = st.date_input("Past End", value=today, key=f"past_end_{key}")
    if start > end:
        start, end = end, start
    if (end - start).days > 29:
        st.warning("Past custom range capped to 30 days.")
        start = end - timedelta(days=29)
    return start, end


def _upcoming_range_control(key: str) -> tuple[date, date]:
    preset = st.selectbox(
        "Upcoming Range",
        ["1 day", "7 days", "30 days", "Custom (max 30 days)"],
        index=1,
        key=f"upcoming_preset_{key}",
    )
    today = date.today()

    if preset == "1 day":
        return today, today
    if preset == "7 days":
        return today, today + timedelta(days=6)
    if preset == "30 days":
        return today, today + timedelta(days=29)

    start = st.date_input("Upcoming Start", value=today, key=f"up_start_{key}")
    end = st.date_input("Upcoming End", value=today + timedelta(days=6), key=f"up_end_{key}")
    if start > end:
        start, end = end, start
    if (end - start).days > 29:
        st.warning("Upcoming custom range capped to 30 days.")
        end = start + timedelta(days=29)
    return start, end


def render_home(df: pd.DataFrame) -> None:
    st.title("NBA Spread Tracker")

    if df.empty:
        st.warning("No data in DB. Run `python3 scripts/init_db.py`, `python3 scripts/backfill_season.py`, and `python3 scripts/predict_today.py`.")
        return

    min_date = df["game_date"].min()
    max_date = df["game_date"].max()

    with st.expander("Definitions", expanded=False):
        st.write("`Model Pick`: straight-up lean based only on predicted margin sign.")
        st.write("`ATS`: Against The Spread, i.e. pick versus bookmaker spread.")
        st.write("`Edge`: predicted margin minus spread (in points). Positive favors home ATS.")

    st.caption(f"Data in DB: {min_date.isoformat()} to {max_date.isoformat()}")

    c1, c2 = st.columns(2)
    with c1:
        past_start, past_end = _past_range_control("home")
    with c2:
        up_start, up_end = _upcoming_range_control("home")

    past_df = df[(df["game_date"] >= past_start) & (df["game_date"] <= past_end)].copy()
    finals = past_df[past_df["status"] == "Final"].copy().sort_values(["date", "game_id"], ascending=[False, False])

    upcoming_df = df[(df["game_date"] >= up_start) & (df["game_date"] <= up_end)].copy()
    pending = upcoming_df[upcoming_df["status"] == "Pending"].copy().sort_values(["date", "game_id"], ascending=[True, True])

    if pending["spread_close"].isna().all() and not pending.empty:
        st.info("Upcoming games have no spreads yet. Set `THE_ODDS_API_KEY` and rerun `predict_today.py` for ATS picks.")

    st.subheader("Upcoming / Pending")
    if pending.empty:
        st.info("No pending games in this upcoming range.")
    st.dataframe(
        pending[
            [
                "game_date",
                "matchup",
                "predicted_margin",
                "model_pick",
                "spread_close",
                "edge",
                "ats_pick_display",
            ]
        ].rename(
            columns={
                "game_date": "Date",
                "predicted_margin": "Pred Margin",
                "model_pick": "Model Pick",
                "spread_close": "Spread (Close)",
                "edge": "Edge",
                "ats_pick_display": "Pick vs Spread (ATS)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Finished")
    if finals.empty:
        st.info("No finished games in this past range.")
    st.dataframe(
        finals[
            [
                "game_date",
                "matchup",
                "home_score",
                "away_score",
                "actual_margin",
                "predicted_margin",
                "model_pick",
                "spread_close",
                "edge",
                "ats_pick_display",
                "ats_result",
                "units",
            ]
        ].rename(
            columns={
                "game_date": "Date",
                "actual_margin": "Actual Margin",
                "predicted_margin": "Pred Margin",
                "model_pick": "Model Pick",
                "spread_close": "Spread (Close)",
                "ats_pick_display": "Pick vs Spread (ATS)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    detail_source = pd.concat([pending, finals], ignore_index=True)
    if not detail_source.empty:
        st.subheader("Game Detail")
        detail_source = detail_source.sort_values(["date", "matchup"], ascending=[False, True])
        options = detail_source.apply(lambda r: f"{r['game_date']} | {r['matchup']}", axis=1).tolist()
        idx = st.selectbox("Select Game", range(len(options)), format_func=lambda i: options[i])
        row = detail_source.iloc[int(idx)]

        d1, d2 = st.columns(2)
        d1.write(
            {
                "date": row["game_date"].isoformat(),
                "matchup": row["matchup"],
                "predicted_margin": row["predicted_margin"],
                "model_pick": row["model_pick"],
                "spread_close": row["spread_close"],
                "edge": row["edge"],
                "pick_vs_spread_ats": row["ats_pick_display"],
                "status": row["status"],
            }
        )
        d2.write(
            {
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "actual_margin": row["actual_margin"],
                "ats_result": row["ats_result"],
                "units": row["units"],
                "model_version": row["model_version"],
            }
        )


def render_performance(df: pd.DataFrame) -> None:
    st.title("Performance")
    if df.empty:
        st.warning("No data available.")
        return

    all_metrics = kpi_metrics(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("All-Time ATS Record", all_metrics["record"])
    c2.metric("All-Time Units", f"{all_metrics['units']:.2f}")
    c3.metric("All-Time Win Rate", f"{all_metrics['win_rate'] * 100:.1f}%")
    c4.metric("All-Time MAE", f"{all_metrics['mae']:.2f}")
    c5.metric("All-Time Sample", str(all_metrics["sample"]))

    start, end = _past_range_control("perf")
    filtered = df[(df["game_date"] >= start) & (df["game_date"] <= end)].copy()
    st.caption(f"Charts below are filtered to: {start.isoformat()} through {end.isoformat()}")

    results = filtered[(filtered["status"] == "Final") & (filtered["units"].notna())].copy()
    if results.empty:
        st.info("No finalized ATS rows yet for this selected range.")
        return

    results = results.sort_values("date")
    results["cum_units"] = results["units"].cumsum()
    results["rolling_win"] = (
        (results["ats_result"] == "WIN").astype(int).rolling(window=20, min_periods=5).mean()
    )
    results = add_edge_bucket(results)

    chart1 = px.line(results, x="date", y="cum_units", title="Cumulative Units")
    st.plotly_chart(chart1, use_container_width=True)

    chart2 = px.line(results, x="date", y="rolling_win", title="Rolling Win Rate (20 bets)")
    st.plotly_chart(chart2, use_container_width=True)

    chart3 = px.histogram(results.dropna(subset=["edge"]), x="edge", nbins=30, title="Edge Distribution")
    st.plotly_chart(chart3, use_container_width=True)

    bucket = (
        results.dropna(subset=["edge_bucket"])
        .groupby("edge_bucket", observed=True)
        .agg(win_rate=("ats_result", lambda s: (s == "WIN").mean()), bets=("matchup", "count"))
        .reset_index()
    )
    chart4 = px.bar(bucket, x="edge_bucket", y="win_rate", hover_data=["bets"], title="Win Rate by |Edge| Bucket")
    st.plotly_chart(chart4, use_container_width=True)


def render_model_card() -> None:
    st.title("Model Card")
    st.markdown(
        """
### Baseline Approach
- **Target**: `actual_margin = home_score - away_score`
- **Model**: Ridge regression on pre-game team features
- **Features**: Elo diff, rest days, back-to-back flags, rolling point differential, home indicator

### Definitions
- **Model Pick**: straight-up lean from predicted margin sign
- **ATS**: Against The Spread (pick vs bookmaker spread)
- **Edge**: `predicted_margin - spread`

### Evaluation
- Time-based split (last 20% holdout)
- MAE for margin prediction
- ATS record and units using -110 vig assumptions (`+0.91/-1/0`)

### Scope
- Focus is forward tracking with real schedule/results + ongoing odds snapshots.
        """
    )

    runs = repo.get_model_versions()
    if not runs.empty:
        latest = runs.iloc[0]
        st.subheader("Latest Training Run")
        st.code(json.dumps(json.loads(latest["metrics_json"]), indent=2), language="json")


def main() -> None:
    df = _load_all()
    page = st.sidebar.radio("Page", ["Home", "Performance", "Model Card"])

    if page == "Home":
        render_home(df)
    elif page == "Performance":
        render_performance(df)
    else:
        render_model_card()


if __name__ == "__main__":
    main()
