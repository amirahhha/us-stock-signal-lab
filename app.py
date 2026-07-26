"""Local Streamlit dashboard for the US Signal Desk research model."""

from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import DB_PATH, PROJECT_DIR


st.set_page_config(
    page_title="US Signal Desk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --bg: #090e14;
        --panel: #111922;
        --panel-2: #151f2b;
        --border: #243244;
        --text: #edf3f8;
        --muted: #8fa1b3;
        --blue: #5bb7ff;
        --green: #45d391;
        --red: #ff737d;
        --amber: #f2bd62;
      }
      .stApp { background: var(--bg); color: var(--text); }
      .block-container { max-width: 1460px; padding-top: 1.4rem; padding-bottom: 3rem; }
      header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton { display: none; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      h1 { font-size: 2.35rem !important; font-weight: 720 !important; margin-bottom: .15rem !important; }
      h2 { font-size: 1.25rem !important; margin-top: 1.55rem !important; }
      p, li { color: #c4d0db; line-height: 1.55; }
      .eyebrow {
        color: var(--blue); font-size: .72rem; font-weight: 750; letter-spacing: .14em;
        text-transform: uppercase; margin: 1.4rem 0 .55rem 0;
      }
      .lede { color: var(--muted); font-size: .95rem; margin-bottom: .7rem; }
      .source-line { color: #718398; font-size: .78rem; margin-bottom: 1rem; }
      .metric-card {
        background: linear-gradient(145deg, #121b26 0%, #0f1720 100%);
        border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.05rem;
        min-height: 132px;
      }
      .metric-label {
        color: var(--muted); font-size: .68rem; font-weight: 750; letter-spacing: .11em;
        text-transform: uppercase; margin-bottom: .55rem;
      }
      .metric-value { color: var(--text); font-size: 1.55rem; font-weight: 730; line-height: 1.18; }
      .metric-value.positive { color: var(--green); }
      .metric-value.negative { color: var(--red); }
      .metric-value.accent { color: var(--blue); }
      .metric-detail { color: #8395a8; font-size: .76rem; margin-top: .55rem; line-height: 1.35; }
      .pick-panel {
        background: linear-gradient(120deg, rgba(42, 84, 116, .27), rgba(18, 28, 39, .75));
        border: 1px solid #31536f; border-radius: 12px; padding: 1.15rem 1.25rem;
      }
      .pick-ticker { color: var(--green); font-size: 2rem; font-weight: 760; }
      .pick-company { color: var(--text); font-size: 1rem; font-weight: 650; }
      .reason-box, .risk-box {
        border-radius: 8px; padding: .9rem 1rem; min-height: 116px;
      }
      .reason-box { background: rgba(34, 105, 76, .18); border: 1px solid rgba(69, 211, 145, .28); }
      .risk-box { background: rgba(123, 76, 29, .17); border: 1px solid rgba(242, 189, 98, .26); }
      .box-label {
        font-size: .68rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase;
        margin-bottom: .45rem;
      }
      .reason-box .box-label { color: var(--green); }
      .risk-box .box-label { color: var(--amber); }
      .box-copy { color: #c8d3dc; font-size: .86rem; line-height: 1.55; }
      .note {
        background: #111a24; border-left: 3px solid #3d7ba8; color: #9fb1c2;
        padding: .75rem .9rem; font-size: .8rem; margin: .6rem 0 1rem 0;
      }
      .status {
        display: inline-block; border-radius: 999px; padding: .28rem .58rem; font-size: .67rem;
        font-weight: 760; letter-spacing: .08em; text-transform: uppercase;
        background: rgba(91, 183, 255, .13); color: var(--blue); border: 1px solid rgba(91,183,255,.25);
      }
      [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
      [data-testid="stExpander"] { border: 1px solid var(--border); background: #0e151e; }
      .footer { color: #627487; font-size: .72rem; text-align: center; margin-top: 2.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _query(sql: str, parameters: list[object] | None = None) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as connection:
        return connection.execute(sql, parameters or []).fetchdf()


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return {
        "rankings": _query("SELECT * FROM rankings ORDER BY rank"),
        "metrics": _query("SELECT * FROM model_metrics LIMIT 1"),
        "metadata": _query("SELECT * FROM metadata LIMIT 1"),
        "validation": _query("SELECT * FROM validation_predictions"),
        "history": _query("SELECT * FROM ranking_history ORDER BY as_of_date, rank"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker_prices(ticker: str) -> pd.DataFrame:
    return _query(
        """
        SELECT trade_date, open, high, low, close, volume
        FROM prices
        WHERE ticker = ?
        ORDER BY trade_date
        """,
        [ticker],
    )


def metric_card(label: str, value: str, detail: str, tone: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value {tone}">{html.escape(value)}</div>
          <div class="metric-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_money(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    return f"${value:,.0f}"


def run_refresh() -> tuple[bool, str]:
    process = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "pipeline.py"), "--period", "5y"],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    output = (process.stdout + "\n" + process.stderr).strip()
    return process.returncode == 0, output


st.title("US Signal Desk")
st.markdown(
    '<div class="lede">Daily technical screening and 21-session probability ranking for a curated universe of liquid U.S. large-cap stocks.</div>',
    unsafe_allow_html=True,
)

if not DB_PATH.exists():
    st.error("The research database has not been created yet.")
    st.code(f'cd "{PROJECT_DIR}"\npython pipeline.py --period 5y', language="bash")
    st.stop()

data = load_dashboard_data()
rankings = data["rankings"]
metrics = data["metrics"].iloc[0]
metadata = data["metadata"].iloc[0]
validation = data["validation"]
ranking_history = data["history"]

as_of = pd.Timestamp(metadata["as_of_date"]).strftime("%-d %B %Y")
run_utc = pd.Timestamp(metadata["run_timestamp_utc"]).strftime("%-d %b %Y · %H:%M UTC")
st.markdown(
    f"""
    <div class="source-line">
      Data source: Yahoo Finance via yfinance · adjusted daily OHLCV · benchmark: SPY ·
      latest market session: {as_of} · database refreshed: {run_utc}
    </div>
    """,
    unsafe_allow_html=True,
)

refresh_col, status_col = st.columns([1, 5], vertical_alignment="center")
with refresh_col:
    if st.button("Refresh database", type="secondary", use_container_width=True):
        with st.spinner("Downloading prices and rebuilding the model..."):
            success, refresh_output = run_refresh()
        if success:
            st.cache_data.clear()
            st.success("Database refreshed.")
            st.rerun()
        else:
            st.error("Refresh failed; the previous database remains available.")
            with st.expander("Refresh log"):
                st.code(refresh_output)
with status_col:
    st.markdown(
        '<span class="status">local research system</span>',
        unsafe_allow_html=True,
    )

top = rankings.iloc[0]
eligible_count = int(rankings["eligible"].sum())
above_200_count = int((rankings["close"] > rankings["sma200"]).sum())
regime_positive = str(top["regime"]) == "risk-on"

st.markdown('<div class="eyebrow">Daily model selection</div>', unsafe_allow_html=True)
pick_left, pick_right = st.columns([1.05, 1.95], gap="medium")
with pick_left:
    st.markdown(
        f"""
        <div class="pick-panel">
          <div class="metric-label">daily research candidate · {html.escape(str(top['signal']))}</div>
          <div class="pick-ticker">{html.escape(str(top['ticker']))}</div>
          <div class="pick-company">{html.escape(str(top['company']))}</div>
          <div class="metric-detail">{html.escape(str(top['sector']))} · close ${top['close']:,.2f} · rank 1 of {len(rankings)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with pick_right:
    st.markdown(
        f"""
        <div class="note">
          The system first applies technical, liquidity, trend, and extension gates, then ranks the
          eligible stocks. <b>{html.escape(str(top['ticker']))}</b> is first for the next 21 trading
          sessions. The selection score is 90% logistic-model probability and 10% technical
          strength. It is a research candidate, not an instruction to trade.
        </div>
        """,
        unsafe_allow_html=True,
    )
    reason_col, risk_col = st.columns(2, gap="small")
    with reason_col:
        st.markdown(
            f"""
            <div class="reason-box">
              <div class="box-label">Why it ranks</div>
              <div class="box-copy">{html.escape(str(top['reasons']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with risk_col:
        st.markdown(
            f"""
            <div class="risk-box">
              <div class="box-label">What can invalidate it</div>
              <div class="box-copy">{html.escape(str(top['risks']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

metric_columns = st.columns(5, gap="small")
with metric_columns[0]:
    metric_card("selection score", f"{top['final_score']:.1f}/100", "90% model + 10% technical", "positive")
with metric_columns[1]:
    metric_card(
        "21-session up probability",
        f"{top['probability_up_21d']:.1%}",
        "model estimate, not certainty",
        "accent",
    )
with metric_columns[2]:
    metric_card(
        "market regime",
        str(top["regime"]).upper(),
        f"SPY trend filter · {above_200_count}/{len(rankings)} above 200d",
        "positive" if regime_positive else "negative",
    )
with metric_columns[3]:
    metric_card(
        "eligible candidates",
        str(eligible_count),
        "liquidity + trend + extension gates",
    )
with metric_columns[4]:
    metric_card(
        "validation top decile",
        f"{metrics['top_decile_hit_rate']:.1%}",
        f"baseline up rate {metrics['baseline_up_rate']:.1%}",
        "accent",
    )

st.markdown('<div class="eyebrow">Candidate leaderboard</div>', unsafe_allow_html=True)
filter_col, count_col = st.columns([3, 1])
with filter_col:
    sectors = sorted(rankings["sector"].dropna().unique().tolist())
    selected_sectors = st.multiselect(
        "Sector filter",
        sectors,
        default=sectors,
        label_visibility="collapsed",
    )
with count_col:
    top_n = st.selectbox("Rows", [10, 15, 25, len(rankings)], index=1, label_visibility="collapsed")

filtered = rankings[rankings["sector"].isin(selected_sectors)].head(int(top_n)).copy()
display_table = filtered[
    [
        "rank",
        "ticker",
        "company",
        "sector",
        "signal",
        "final_score",
        "technical_score",
        "probability_up_21d",
        "ret_21d",
        "ret_63d",
        "rsi14",
        "atr_pct",
        "avg_dollar_volume20",
    ]
].rename(
    columns={
        "rank": "Rank",
        "ticker": "Ticker",
        "company": "Company",
        "sector": "Sector",
        "signal": "Signal",
        "final_score": "Composite",
        "technical_score": "Technical",
        "probability_up_21d": "P(up 21d)",
        "ret_21d": "21d return",
        "ret_63d": "63d return",
        "rsi14": "RSI(14)",
        "atr_pct": "ATR %",
        "avg_dollar_volume20": "Avg $ volume",
    }
)
for percentage_column in ["P(up 21d)", "21d return", "63d return", "ATR %"]:
    display_table[percentage_column] = display_table[percentage_column] * 100
st.dataframe(
    display_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Composite": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "Technical": st.column_config.NumberColumn(format="%.1f"),
        "P(up 21d)": st.column_config.NumberColumn(format="%.1f%%"),
        "21d return": st.column_config.NumberColumn(format="%.1f%%"),
        "63d return": st.column_config.NumberColumn(format="%.1f%%"),
        "RSI(14)": st.column_config.NumberColumn(format="%.1f"),
        "ATR %": st.column_config.NumberColumn(format="%.1f%%"),
        "Avg $ volume": st.column_config.NumberColumn(format="$%.0f"),
    },
)

scatter = px.scatter(
    rankings,
    x="technical_score",
    y="probability_up_21d",
    color="sector",
    hover_name="ticker",
    hover_data={
        "company": True,
        "final_score": ":.1f",
        "ret_21d": ":.1%",
        "rsi14": ":.0f",
        "technical_score": False,
        "probability_up_21d": False,
    },
    labels={
        "technical_score": "Technical score",
        "probability_up_21d": "21-session probability",
        "sector": "Sector",
    },
    title="Technical strength versus model probability",
)
scatter.update_traces(marker={"size": 10, "opacity": 0.78, "line": {"width": 0}})
scatter.update_yaxes(tickformat=".0%")
scatter.update_layout(
    height=430,
    paper_bgcolor="#090e14",
    plot_bgcolor="#111922",
    font={"color": "#aebdca"},
    title_font={"color": "#e7eef5", "size": 15},
    legend={"orientation": "h", "y": -0.22},
    margin={"l": 20, "r": 20, "t": 55, "b": 70},
)
st.plotly_chart(scatter, use_container_width=True, config={"displaylogo": False})

st.markdown('<div class="eyebrow">Stock drilldown</div>', unsafe_allow_html=True)
selected_ticker = st.selectbox(
    "Select a stock",
    rankings["ticker"].tolist(),
    index=0,
    format_func=lambda ticker: f"{ticker} · {rankings.set_index('ticker').loc[ticker, 'company']}",
)
selected = rankings.set_index("ticker").loc[selected_ticker]
price_data = load_ticker_prices(selected_ticker).tail(320).copy()
for window in (20, 50, 200):
    price_data[f"sma{window}"] = price_data["close"].rolling(window).mean()

chart_col, profile_col = st.columns([2.25, 1], gap="medium")
with chart_col:
    price_figure = go.Figure()
    price_figure.add_trace(
        go.Candlestick(
            x=price_data["trade_date"],
            open=price_data["open"],
            high=price_data["high"],
            low=price_data["low"],
            close=price_data["close"],
            name=selected_ticker,
            increasing_line_color="#45d391",
            decreasing_line_color="#ff737d",
        )
    )
    for name, color in [("sma20", "#5bb7ff"), ("sma50", "#f2bd62"), ("sma200", "#a78bfa")]:
        price_figure.add_trace(
            go.Scatter(
                x=price_data["trade_date"],
                y=price_data[name],
                mode="lines",
                name=name.upper(),
                line={"color": color, "width": 1.4},
            )
        )
    price_figure.update_layout(
        title=f"{selected_ticker} · adjusted daily price",
        height=520,
        paper_bgcolor="#090e14",
        plot_bgcolor="#111922",
        font={"color": "#aebdca"},
        title_font={"color": "#e7eef5", "size": 15},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 20, "r": 20, "t": 70, "b": 25},
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(price_figure, use_container_width=True, config={"displaylogo": False})
with profile_col:
    metric_card("signal", str(selected["signal"]), f"rank {int(selected['rank'])} of {len(rankings)}")
    metric_card(
        "price / 200-day",
        f"{selected['dist_sma200']:+.1%}",
        f"close ${selected['close']:,.2f} · SMA200 ${selected['sma200']:,.2f}",
        "positive" if selected["dist_sma200"] > 0 else "negative",
    )
    metric_card(
        "momentum",
        f"{selected['ret_21d']:+.1%}",
        f"63-day {selected['ret_63d']:+.1%} · RSI {selected['rsi14']:.0f}",
    )
    metric_card(
        "risk reference",
        f"${selected['risk_reference']:,.2f}",
        f"two ATR below close · ATR {selected['atr_pct']:.1%}",
    )

detail_reason, detail_risk = st.columns(2)
with detail_reason:
    st.markdown(
        f"""
        <div class="reason-box">
          <div class="box-label">positive evidence</div>
          <div class="box-copy">{html.escape(str(selected['reasons']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with detail_risk:
    st.markdown(
        f"""
        <div class="risk-box">
          <div class="box-label">risk evidence</div>
          <div class="box-copy">{html.escape(str(selected['risks']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

breakdown = pd.DataFrame(
    {
        "Component": ["Trend", "Momentum", "Timing", "Risk", "Model"],
        "Score": [
            selected["trend_score"],
            selected["momentum_score"],
            selected["timing_score"],
            selected["risk_score"],
            selected["model_score"],
        ],
        "Maximum": [36, 30, 15, 19, 100],
    }
)
breakdown["Share of maximum"] = breakdown["Score"] / breakdown["Maximum"]
breakdown_chart = px.bar(
    breakdown,
    x="Component",
    y="Share of maximum",
    text=breakdown["Score"].map(lambda value: f"{value:.1f}"),
    title="Raw signal component strength",
)
breakdown_chart.update_traces(marker_color=["#5bb7ff", "#45d391", "#f2bd62", "#a78bfa", "#70d6ff"])
breakdown_chart.update_yaxes(tickformat=".0%", range=[0, 1.05])
breakdown_chart.update_layout(
    height=330,
    paper_bgcolor="#090e14",
    plot_bgcolor="#111922",
    font={"color": "#aebdca"},
    title_font={"color": "#e7eef5", "size": 15},
    showlegend=False,
    margin={"l": 20, "r": 20, "t": 55, "b": 25},
)
st.plotly_chart(breakdown_chart, use_container_width=True, config={"displaylogo": False})

st.markdown('<div class="eyebrow">Model validation</div>', unsafe_allow_html=True)
validation_cols = st.columns(5, gap="small")
with validation_cols[0]:
    metric_card("ROC AUC", f"{metrics['roc_auc']:.3f}", "0.50 is random ranking")
with validation_cols[1]:
    metric_card("accuracy", f"{metrics['accuracy']:.1%}", "all validation observations")
with validation_cols[2]:
    metric_card("positive precision", f"{metrics['precision']:.1%}", "when probability ≥ 50%")
with validation_cols[3]:
    metric_card("top-decile hit rate", f"{metrics['top_decile_hit_rate']:.1%}", "highest daily probabilities")
with validation_cols[4]:
    metric_card(
        "top-decile avg 21d return",
        f"{metrics['top_decile_avg_return']:+.2%}",
        "before costs; overlapping labels",
    )

validation_ranked = validation.copy()
validation_ranked["selection_rank"] = np.nan
eligible_validation_index = validation_ranked.index[validation_ranked["eligible"]]
validation_ranked.loc[eligible_validation_index, "selection_rank"] = validation_ranked.loc[
    eligible_validation_index
].groupby("trade_date")["selection_score"].rank(pct=True)
validation_daily = (
    validation_ranked
    .groupby("trade_date")
    .apply(
        lambda frame: pd.Series(
            {
                "Top eligible decile": frame.loc[
                    frame["selection_rank"] >= 0.9, "target_up_21d"
                ].mean(),
                "All stocks": frame["target_up_21d"].mean(),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)
validation_daily[["Top eligible decile", "All stocks"]] = validation_daily[
    ["Top eligible decile", "All stocks"]
].rolling(20, min_periods=10).mean()
validation_chart = px.line(
    validation_daily,
    x="trade_date",
    y=["Top eligible decile", "All stocks"],
    labels={"value": "20-session rolling hit rate", "trade_date": "", "variable": ""},
    title=(
        f"Chronological holdout · {pd.Timestamp(metrics['validation_start']).date()} "
        f"to {pd.Timestamp(metrics['validation_end']).date()}"
    ),
)
validation_chart.update_traces(line={"width": 2.2})
validation_chart.update_yaxes(tickformat=".0%")
validation_chart.update_layout(
    height=390,
    paper_bgcolor="#090e14",
    plot_bgcolor="#111922",
    font={"color": "#aebdca"},
    title_font={"color": "#e7eef5", "size": 15},
    legend={"orientation": "h", "y": 1.08},
    margin={"l": 20, "r": 20, "t": 65, "b": 25},
)
st.plotly_chart(validation_chart, use_container_width=True, config={"displaylogo": False})
st.markdown(
    """
    <div class="note">
      Validation is chronological and leaves a 21-session gap between training and holdout data
      to reduce look-ahead leakage. The 21-session outcomes overlap, and the figures exclude
      commissions, slippage, taxes, and position sizing; this is model diagnostics, not a tradable
      performance claim.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Database and signal history</div>', unsafe_allow_html=True)
history_dates = ranking_history["as_of_date"].nunique()
database_columns = st.columns(4, gap="small")
with database_columns[0]:
    metric_card("price rows", f"{int(metadata['price_rows']):,}", "adjusted daily observations")
with database_columns[1]:
    metric_card("feature rows", f"{int(metadata['feature_rows']):,}", "indicator observations")
with database_columns[2]:
    metric_card("ranked universe", str(int(metadata["ranked_tickers"])), "version-controlled list")
with database_columns[3]:
    metric_card("history sessions", str(history_dates), "one snapshot per market date")

if history_dates > 1:
    top_history = ranking_history[ranking_history["rank"] == 1][
        ["as_of_date", "ticker", "final_score", "probability_up_21d", "signal"]
    ].sort_values("as_of_date", ascending=False)
    st.dataframe(top_history, hide_index=True, use_container_width=True)
else:
    st.caption("Signal history begins with this run and will accumulate one snapshot per refreshed market session.")

with st.expander("Methodology and definitions"):
    st.markdown(
        """
        **Ranking design**

        - The technical score measures trend, momentum, timing, volume confirmation, volatility,
          ATR, proximity to the 52-week high, liquidity, and overextension.
        - The statistical layer is a logistic regression fitted to daily cross-sectional feature
          ranks and estimates whether the adjusted close will be higher 21 trading sessions later.
        - Technical rules establish eligibility first. Eligible stocks are then ranked using
          90% model probability and 10% technical strength. A risk-cautious SPY regime applies
          a five-point penalty.
        - Eligibility requires price of at least $5, average 20-day dollar volume of at least
          $20 million, sufficient history, price above the 200-day average, RSI below 80, and
          limited distance above the 20-day average.

        **Abbreviations**

        - **SMA:** simple moving average; the mean closing price over a fixed number of sessions.
        - **RSI:** relative strength index; a 0–100 momentum oscillator.
        - **MACD:** moving average convergence divergence; the difference between 12- and
          26-session exponential averages, compared with its signal line.
        - **ATR:** average true range; recent daily price movement used here as a volatility scale.
        - **ROC AUC:** how well model probabilities rank positive outcomes above negative outcomes;
          0.50 is equivalent to random ordering.
        """
    )

st.markdown(
    """
    <div class="footer">
      US Signal Desk · Python · DuckDB · scikit-learn · Streamlit · Plotly · Yahoo Finance snapshot<br>
      Research and portfolio demonstration only. Not personalized investment advice or a guarantee of future performance.
    </div>
    """,
    unsafe_allow_html=True,
)
