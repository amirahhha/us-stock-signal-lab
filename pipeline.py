"""Daily data, feature, model, and ranking pipeline for US Signal Desk."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from universe import BENCHMARK, UNIVERSE


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "us_signal_lab.duckdb"

MODEL_BASE_FEATURES = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "dist_sma20",
    "dist_sma50",
    "dist_sma200",
    "rsi14",
    "macd_hist_pct",
    "atr_pct",
    "realized_vol20",
    "volume_ratio",
    "rel_spy_63d",
]
MODEL_FEATURES = [f"rank_{feature}" for feature in MODEL_BASE_FEATURES]


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))
        level_one = set(raw.columns.get_level_values(1))
        if ticker in level_zero:
            frame = raw[ticker].copy()
        elif ticker in level_one:
            frame = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = raw.copy()

    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    frame = frame[required].reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "trade_date"})
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    frame["trade_date"] = dates.dt.normalize()
    frame["ticker"] = ticker
    frame = frame.dropna(subset=["trade_date", "close"])
    frame = frame[frame["close"] > 0]
    return frame[["trade_date", "ticker", *required]]


def download_prices(period: str = "5y") -> tuple[pd.DataFrame, list[str]]:
    tickers = [*UNIVERSE.keys(), BENCHMARK]
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        actions=False,
        threads=True,
        progress=False,
        repair=True,
        timeout=30,
    )

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for ticker in tickers:
        frame = _extract_ticker_frame(raw, ticker)
        if frame.empty:
            missing.append(ticker)
        else:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No price history was returned. The existing database was not changed.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "trade_date"]).drop_duplicates(
        ["ticker", "trade_date"], keep="last"
    )
    prices["company"] = prices["ticker"].map(
        {ticker: company for ticker, (company, _) in UNIVERSE.items()}
    ).fillna("SPDR S&P 500 ETF Trust")
    prices["sector"] = prices["ticker"].map(
        {ticker: sector for ticker, (_, sector) in UNIVERSE.items()}
    ).fillna("Benchmark")
    return prices.reset_index(drop=True), missing


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    result = result.mask((average_loss == 0) & (average_gain > 0), 100)
    return result


def _ticker_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("trade_date").copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    frame["ret_1d"] = close.pct_change()
    for window in (5, 21, 63, 126, 252):
        frame[f"ret_{window}d"] = close.pct_change(window)

    for window in (20, 50, 200):
        frame[f"sma{window}"] = close.rolling(window, min_periods=window).mean()
        frame[f"dist_sma{window}"] = close / frame[f"sma{window}"] - 1

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_hist_pct"] = (frame["macd"] - frame["macd_signal"]) / close
    frame["rsi14"] = _rsi(close)

    prior_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    frame["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["atr_pct"] = frame["atr14"] / close
    frame["realized_vol20"] = frame["ret_1d"].rolling(20, min_periods=20).std() * math.sqrt(252)
    frame["volume_ratio"] = volume / volume.rolling(20, min_periods=20).mean()
    frame["avg_dollar_volume20"] = (close * volume).rolling(20, min_periods=20).mean()
    frame["high_252"] = close.rolling(252, min_periods=126).max()
    frame["drawdown_52w"] = close / frame["high_252"] - 1
    frame["forward_5d"] = close.shift(-5) / close - 1
    frame["forward_21d"] = close.shift(-21) / close - 1
    frame["history_count"] = np.arange(1, len(frame) + 1)
    return frame


def calculate_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_frames = [
        _ticker_features(frame)
        for _, frame in prices.groupby("ticker", sort=False, group_keys=False)
    ]
    all_features = pd.concat(feature_frames, ignore_index=True)

    spy = all_features[all_features["ticker"] == BENCHMARK][
        ["trade_date", "close", "sma50", "sma200", "ret_63d", "ret_126d"]
    ].rename(
        columns={
            "close": "spy_close",
            "sma50": "spy_sma50",
            "sma200": "spy_sma200",
            "ret_63d": "spy_ret_63d",
            "ret_126d": "spy_ret_126d",
        }
    )

    features = all_features[all_features["ticker"] != BENCHMARK].merge(
        spy, on="trade_date", how="left"
    )
    features["rel_spy_63d"] = features["ret_63d"] - features["spy_ret_63d"]
    features["rel_spy_126d"] = features["ret_126d"] - features["spy_ret_126d"]
    features["target_up_5d"] = np.where(
        features["forward_5d"].notna(),
        (features["forward_5d"] > 0).astype(int),
        np.nan,
    )
    features["target_up_21d"] = np.where(
        features["forward_21d"].notna(),
        (features["forward_21d"] > 0).astype(int),
        np.nan,
    )

    latest_spy = spy.dropna(subset=["spy_close"]).sort_values("trade_date").iloc[-1]
    regime_positive = bool(
        latest_spy["spy_close"] > latest_spy["spy_sma200"]
        and latest_spy["spy_sma50"] > latest_spy["spy_sma200"]
    )
    regime = {
        "as_of_date": pd.Timestamp(latest_spy["trade_date"]).date().isoformat(),
        "spy_close": float(latest_spy["spy_close"]),
        "spy_sma50": float(latest_spy["spy_sma50"]),
        "spy_sma200": float(latest_spy["spy_sma200"]),
        "regime": "risk-on" if regime_positive else "risk-cautious",
        "regime_positive": regime_positive,
    }
    return features.sort_values(["trade_date", "ticker"]).reset_index(drop=True), regime


def add_technical_scores(features: pd.DataFrame) -> pd.DataFrame:
    scored = features.copy()
    grouped = scored.groupby("trade_date", sort=False)
    for feature in MODEL_BASE_FEATURES:
        scored[f"rank_{feature}"] = grouped[feature].rank(pct=True)

    momentum_21_rank = grouped["ret_21d"].rank(pct=True).fillna(0.5)
    momentum_63_rank = grouped["ret_63d"].rank(pct=True).fillna(0.5)
    relative_rank = grouped["rel_spy_63d"].rank(pct=True).fillna(0.5)
    volatility_rank = grouped["realized_vol20"].rank(pct=True).fillna(0.5)
    atr_rank = grouped["atr_pct"].rank(pct=True).fillna(0.5)

    trend_score = (
        (scored["close"] > scored["sma20"]).astype(float) * 6
        + (scored["close"] > scored["sma50"]).astype(float) * 8
        + (scored["close"] > scored["sma200"]).astype(float) * 10
        + (scored["sma20"] > scored["sma50"]).astype(float) * 4
        + (scored["sma50"] > scored["sma200"]).astype(float) * 4
        + (scored["macd_hist_pct"] > 0).astype(float) * 4
    )
    momentum_score = momentum_21_rank * 10 + momentum_63_rank * 10 + relative_rank * 10
    timing_score = np.select(
        [
            scored["rsi14"].between(50, 70, inclusive="both"),
            scored["rsi14"].between(45, 50, inclusive="left"),
            scored["rsi14"].between(70, 75, inclusive="right"),
        ],
        [10.0, 6.0, 3.0],
        default=0.0,
    )
    volume_score = np.select(
        [
            (scored["volume_ratio"] >= 1.1) & (scored["ret_1d"] > 0),
            scored["volume_ratio"] >= 0.9,
        ],
        [5.0, 2.0],
        default=0.0,
    )
    risk_score = (1 - volatility_rank) * 8 + (1 - atr_rank) * 6
    high_score = np.select(
        [
            scored["drawdown_52w"].between(-0.15, 0, inclusive="both"),
            scored["drawdown_52w"].between(-0.25, -0.15, inclusive="left"),
        ],
        [5.0, 2.0],
        default=0.0,
    )

    penalty = (
        (scored["close"] < 5).astype(float) * 25
        + (scored["avg_dollar_volume20"] < 20_000_000).astype(float) * 25
        + (scored["rsi14"] > 80).astype(float) * 10
        + (scored["dist_sma20"] > 0.12).astype(float) * 8
        + (scored["close"] < scored["sma200"]).astype(float) * 12
    )

    scored["trend_score"] = trend_score
    scored["momentum_score"] = momentum_score
    scored["timing_score"] = timing_score + volume_score
    scored["risk_score"] = risk_score + high_score
    scored["technical_score"] = (
        trend_score + momentum_score + timing_score + volume_score + risk_score + high_score - penalty
    ).clip(0, 100)
    return scored


def fit_predictive_model(
    scored: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, dict[str, object]]:
    model_data = scored.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*MODEL_FEATURES, "target_up_21d", "forward_21d"]
    ).copy()
    unique_dates = np.array(sorted(model_data["trade_date"].unique()))
    if len(unique_dates) < 300:
        raise RuntimeError("At least 300 trading dates are required to train the model.")

    validation_sessions = min(126, max(63, len(unique_dates) // 5))
    validation_start = unique_dates[-validation_sessions]
    validation_index = int(np.where(unique_dates == validation_start)[0][0])
    training_end_index = max(0, validation_index - 21)
    training_end = unique_dates[training_end_index]

    train = model_data[model_data["trade_date"] < training_end].copy()
    validation = model_data[model_data["trade_date"] >= validation_start].copy()
    if train.empty or validation.empty:
        raise RuntimeError("The chronological model split produced an empty sample.")

    model = Pipeline(
        [
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    train_features = train[MODEL_FEATURES].astype(float)
    validation_features = validation[MODEL_FEATURES].astype(float)
    if not np.isfinite(train_features.to_numpy()).all():
        raise RuntimeError("Non-finite values remained in the model training matrix.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
        model.fit(train_features, train["target_up_21d"].astype(int))
        validation_probability = model.predict_proba(validation_features)[:, 1]
    if not np.isfinite(validation_probability).all():
        raise RuntimeError("The validation model produced a non-finite probability.")
    validation_prediction = (validation_probability >= 0.5).astype(int)
    validation_target = validation["target_up_21d"].astype(int)

    validation_results = validation[
        [
            "trade_date",
            "ticker",
            "forward_21d",
            "target_up_21d",
            "technical_score",
            "close",
            "sma200",
            "avg_dollar_volume20",
            "history_count",
            "rsi14",
            "dist_sma20",
        ]
    ].copy()
    validation_results["probability_up_21d"] = validation_probability
    validation_results["predicted_up_21d"] = validation_prediction
    validation_results["selection_score"] = (
        0.10 * validation_results["technical_score"]
        + 0.90 * validation_results["probability_up_21d"] * 100
    )
    validation_results["eligible"] = (
        (validation_results["close"] >= 5)
        & (validation_results["avg_dollar_volume20"] >= 20_000_000)
        & (validation_results["history_count"] >= 220)
        & (validation_results["close"] > validation_results["sma200"])
        & (validation_results["rsi14"] < 80)
        & (validation_results["dist_sma20"] < 0.15)
    )
    eligible_validation = validation_results[validation_results["eligible"]].copy()
    eligible_validation["selection_rank"] = eligible_validation.groupby("trade_date")[
        "selection_score"
    ].rank(pct=True)
    top_bucket = eligible_validation[eligible_validation["selection_rank"] >= 0.9]
    top_pick_by_date = (
        eligible_validation.sort_values(["trade_date", "selection_score"])
        .groupby("trade_date", as_index=False)
        .tail(1)
    )

    metrics: dict[str, object] = {
        "training_start": pd.Timestamp(train["trade_date"].min()).date().isoformat(),
        "training_end": pd.Timestamp(train["trade_date"].max()).date().isoformat(),
        "validation_start": pd.Timestamp(validation["trade_date"].min()).date().isoformat(),
        "validation_end": pd.Timestamp(validation["trade_date"].max()).date().isoformat(),
        "training_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "baseline_up_rate": float(validation_target.mean()),
        "eligible_up_rate": float(eligible_validation["target_up_21d"].mean()),
        "accuracy": float(accuracy_score(validation_target, validation_prediction)),
        "precision": float(precision_score(validation_target, validation_prediction, zero_division=0)),
        "recall": float(recall_score(validation_target, validation_prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(validation_target, validation_probability)),
        "top_decile_hit_rate": float(top_bucket["target_up_21d"].mean()),
        "top_decile_avg_return": float(top_bucket["forward_21d"].mean()),
        "top_pick_hit_rate": float(top_pick_by_date["target_up_21d"].mean()),
        "top_pick_avg_return": float(top_pick_by_date["forward_21d"].mean()),
        "model_horizon_sessions": 21,
        "method": "chronological holdout with a 21-session gap",
    }

    final_model = Pipeline(
        [
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    final_features = model_data[MODEL_FEATURES].astype(float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
        final_model.fit(final_features, model_data["target_up_21d"].astype(int))
    return final_model, validation_results, metrics


def _reason_text(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    risks: list[str] = []

    if row["close"] > row["sma200"]:
        reasons.append(f"price is {row['dist_sma200']:+.1%} versus its 200-day average")
    else:
        risks.append(f"price is {row['dist_sma200']:+.1%} versus its 200-day average")

    if row["sma20"] > row["sma50"] > row["sma200"]:
        reasons.append("20-, 50-, and 200-day averages are positively stacked")
    elif row["sma20"] < row["sma50"]:
        risks.append("short-term trend is below the 50-day trend")

    if row["ret_21d"] > 0:
        reasons.append(f"21-day momentum is {row['ret_21d']:+.1%}")
    else:
        risks.append(f"21-day momentum is {row['ret_21d']:+.1%}")

    if row["rel_spy_63d"] > 0:
        reasons.append(f"outperformed SPY by {row['rel_spy_63d']:+.1%} over 63 sessions")
    else:
        risks.append(f"lagged SPY by {abs(row['rel_spy_63d']):.1%} over 63 sessions")

    if 50 <= row["rsi14"] <= 70:
        reasons.append(f"RSI is constructive without being extreme ({row['rsi14']:.0f})")
    elif row["rsi14"] > 75:
        risks.append(f"RSI is extended ({row['rsi14']:.0f})")
    elif row["rsi14"] < 40:
        risks.append(f"RSI shows weak momentum ({row['rsi14']:.0f})")

    if row["volume_ratio"] >= 1.1 and row["ret_1d"] > 0:
        reasons.append(f"positive session volume was {row['volume_ratio']:.1f}× its 20-day average")

    if row["atr_pct"] > 0.04:
        risks.append(f"daily ATR is elevated at {row['atr_pct']:.1%} of price")
    if row["dist_sma20"] > 0.10:
        risks.append(f"price is extended {row['dist_sma20']:+.1%} above its 20-day average")
    if row["probability_up_21d"] < 0.5:
        risks.append("the model probability is below 50%")

    if not reasons:
        reasons.append("it has the strongest relative composite score in the current universe")
    if not risks:
        risks.append(
            "earnings and breaking news are not modeled · "
            "the 21-session forecast can reverse quickly"
        )

    return " · ".join(reasons[:4]), " · ".join(risks[:3])


def build_latest_rankings(
    scored: pd.DataFrame,
    model: Pipeline,
    regime: dict[str, object],
) -> pd.DataFrame:
    latest_date = scored["trade_date"].max()
    latest = scored[scored["trade_date"] == latest_date].copy()
    latest = latest.dropna(subset=MODEL_FEATURES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
        latest["probability_up_21d"] = model.predict_proba(
            latest[MODEL_FEATURES].astype(float)
        )[:, 1]
    if not np.isfinite(latest["probability_up_21d"]).all():
        raise RuntimeError("The latest model produced a non-finite probability.")
    latest["model_score"] = latest["probability_up_21d"] * 100
    latest["final_score"] = 0.10 * latest["technical_score"] + 0.90 * latest["model_score"]
    if not bool(regime["regime_positive"]):
        latest["final_score"] = latest["final_score"] - 5

    latest["liquidity_gate"] = (
        (latest["close"] >= 5)
        & (latest["avg_dollar_volume20"] >= 20_000_000)
        & (latest["history_count"] >= 220)
    )
    latest["trend_gate"] = latest["close"] > latest["sma200"]
    latest["overextension_gate"] = (latest["rsi14"] < 80) & (latest["dist_sma20"] < 0.15)
    latest["eligible"] = (
        latest["liquidity_gate"] & latest["trend_gate"] & latest["overextension_gate"]
    )
    latest["final_score"] = latest["final_score"].clip(0, 100)
    latest = latest.sort_values(
        ["eligible", "final_score", "avg_dollar_volume20"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    latest["rank"] = np.arange(1, len(latest) + 1)
    latest["eligible_percentile"] = np.nan
    eligible_index = latest.index[latest["eligible"]]
    latest.loc[eligible_index, "eligible_percentile"] = latest.loc[
        eligible_index, "final_score"
    ].rank(pct=True)
    latest["signal"] = np.select(
        [
            latest["eligible"] & (latest["eligible_percentile"] >= 0.9),
            latest["eligible"] & (latest["eligible_percentile"] >= 0.75),
            latest["eligible"],
        ],
        ["BUY CANDIDATE", "RESEARCH", "WATCH"],
        default="AVOID",
    )
    if not latest.empty and latest.loc[0, "eligible"]:
        latest.loc[0, "signal"] = "TOP PICK"

    reason_pairs = latest.apply(_reason_text, axis=1)
    latest["reasons"] = [pair[0] for pair in reason_pairs]
    latest["risks"] = [pair[1] for pair in reason_pairs]
    latest["risk_reference"] = latest["close"] - (2 * latest["atr14"])
    latest["regime"] = str(regime["regime"])
    return latest


def save_database(
    database_path: Path,
    prices: pd.DataFrame,
    scored: pd.DataFrame,
    rankings: pd.DataFrame,
    validation_results: pd.DataFrame,
    metrics: dict[str, object],
    regime: dict[str, object],
    missing: list[str],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    as_of_date = pd.Timestamp(rankings["trade_date"].max()).date()

    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": run_timestamp,
                "as_of_date": as_of_date,
                "universe_size": len(UNIVERSE),
                "ranked_tickers": len(rankings),
                "price_rows": len(prices),
                "feature_rows": len(scored),
                "missing_tickers": ", ".join(missing),
                "regime": regime["regime"],
                "regime_json": json.dumps(regime),
                "data_source": "Yahoo Finance via yfinance",
            }
        ]
    )
    metrics_frame = pd.DataFrame([{**metrics, "as_of_date": as_of_date, "run_timestamp_utc": run_timestamp}])
    ranking_snapshot = rankings.copy()
    ranking_snapshot["run_timestamp_utc"] = run_timestamp
    ranking_snapshot["as_of_date"] = as_of_date

    with duckdb.connect(str(database_path)) as connection:
        for name, frame in {
            "incoming_prices": prices,
            "incoming_features": scored,
            "incoming_rankings": rankings,
            "incoming_validation": validation_results,
            "incoming_metrics": metrics_frame,
            "incoming_metadata": metadata,
            "incoming_snapshot": ranking_snapshot,
        }.items():
            connection.register(name, frame)

        connection.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM incoming_prices")
        connection.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM incoming_features")
        connection.execute("CREATE OR REPLACE TABLE rankings AS SELECT * FROM incoming_rankings")
        connection.execute(
            "CREATE OR REPLACE TABLE validation_predictions AS SELECT * FROM incoming_validation"
        )
        connection.execute("CREATE OR REPLACE TABLE model_metrics AS SELECT * FROM incoming_metrics")
        connection.execute("CREATE OR REPLACE TABLE metadata AS SELECT * FROM incoming_metadata")

        existing_history_columns: list[str] = []
        try:
            existing_history_columns = [
                row[1] for row in connection.execute("PRAGMA table_info('ranking_history')").fetchall()
            ]
        except duckdb.CatalogException:
            pass
        incoming_history_columns = ranking_snapshot.columns.tolist()
        if existing_history_columns != incoming_history_columns:
            connection.execute(
                "CREATE OR REPLACE TABLE ranking_history AS "
                "SELECT * FROM incoming_snapshot WHERE FALSE"
            )
        connection.execute("DELETE FROM ranking_history WHERE as_of_date = ?", [as_of_date])
        connection.execute("INSERT INTO ranking_history SELECT * FROM incoming_snapshot")


def run_pipeline(period: str, database_path: Path) -> dict[str, object]:
    print(f"Downloading {period} of adjusted daily data for {len(UNIVERSE)} stocks plus SPY...")
    prices, missing = download_prices(period=period)
    print(
        f"Downloaded {len(prices):,} rows from {prices['trade_date'].min().date()} "
        f"through {prices['trade_date'].max().date()}."
    )
    if missing:
        print(f"Missing tickers: {', '.join(missing)}")

    features, regime = calculate_features(prices)
    scored = add_technical_scores(features)
    model, validation_results, metrics = fit_predictive_model(scored)
    rankings = build_latest_rankings(scored, model, regime)
    save_database(
        database_path,
        prices,
        scored,
        rankings,
        validation_results,
        metrics,
        regime,
        missing,
    )

    top = rankings.iloc[0]
    result = {
        "database": str(database_path),
        "as_of_date": pd.Timestamp(top["trade_date"]).date().isoformat(),
        "top_ticker": top["ticker"],
        "top_score": round(float(top["final_score"]), 2),
        "probability_up_21d": round(float(top["probability_up_21d"]), 4),
        "signal": top["signal"],
        "regime": regime["regime"],
        "validation_auc": round(float(metrics["roc_auc"]), 4),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period",
        default="5y",
        choices=["2y", "5y", "10y", "max"],
        help="Price history requested from Yahoo Finance.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DB_PATH,
        help="DuckDB output path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(arguments.period, arguments.database.expanduser().resolve())
