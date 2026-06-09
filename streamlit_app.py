"""
streamlit_app.py
================
Browser dashboard over corr_engine. Run:

    streamlit run streamlit_app.py

Reads the price cache (built by `corr_tool.py build`) and lets you build
correlation matrices over any window, look up most/least correlated names,
run the pairs-divergence screen, and inspect a pair's spread z-score.
Small ad-hoc baskets can be pulled live instead of from the cache.

Point it at the shared cache with the CORR_DATA_DIR env var or the sidebar box.
"""

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import corr_engine as ce

st.set_page_config(page_title="Correlation / Pairs", layout="wide")

# Light theming — disciplined, not decorative. Swap ACCENT for a Shaker hex.
ACCENT = "#1f6f54"
st.markdown(f"""
<style>
  .block-container {{padding-top: 2.2rem;}}
  h1, h2, h3 {{letter-spacing: -0.01em;}}
  div[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums;}}
  .stTabs [data-baseweb="tab-list"] {{gap: 1.5rem;}}
  a {{color: {ACCENT};}}
</style>
""", unsafe_allow_html=True)


# --------------------------- cached data access ---------------------------
@st.cache_data(show_spinner=False)
def load_prices(data_dir: str, mtime: float):
    """Cached loader; mtime in the key invalidates when the cache is rebuilt."""
    return ce.load_cache(data_dir)


@st.cache_data(show_spinner="Fetching live...")
def fetch_live(tickers: tuple, years: float):
    return ce.fetch_live(list(tickers), years=years)


def cache_prices(data_dir: str):
    p = Path(data_dir) / "prices.parquet"
    if not p.exists():
        return None
    return load_prices(data_dir, p.stat().st_mtime)


def to_excel_bytes(export_fn, obj) -> bytes:
    buf = io.BytesIO()
    export_fn(obj, buf)
    buf.seek(0)
    return buf.getvalue()


def style_matrix(corr: pd.DataFrame):
    return (corr.style
            .background_gradient(cmap="RdYlGn", vmin=-1, vmax=1, axis=None)
            .format("{:.2f}"))


# ------------------------------- sidebar -------------------------------
st.sidebar.header("Data")
data_dir = st.sidebar.text_input(
    "Cache folder", os.environ.get("CORR_DATA_DIR", "corr_data"),
    help="Folder holding prices.parquet (built by corr_tool.py build).")

prices = cache_prices(data_dir)
if prices is not None:
    lo, hi = prices.index.min().date(), prices.index.max().date()
    st.sidebar.success(f"{prices.shape[1]} tickers · {lo} → {hi}")
    universe = list(prices.columns)
else:
    st.sidebar.warning("No cache here. Build one, or use Live where offered.")
    universe = []

st.sidebar.header("Timeframe")
mode = st.sidebar.radio("Window", ["Years back", "Date range"], horizontal=True)
sel_years = sel_start = sel_end = None
if mode == "Years back":
    sel_years = st.sidebar.slider("Years", 0.5, 10.0, 3.0, 0.5)
else:
    default_lo = prices.index.min().date() if prices is not None else None
    default_hi = prices.index.max().date() if prices is not None else None
    sel_start = st.sidebar.date_input("Start", default_lo)
    sel_end = st.sidebar.date_input("End", default_hi)


# -------------------------------- header --------------------------------
st.title("Correlation & Pairs")
st.caption("Returns-based correlation over any window, plus a high-correlation "
           "divergence screen for pair ideas. All off one cached price history.")

tab_matrix, tab_single, tab_screen, tab_pair = st.tabs(
    ["Correlation matrix", "Single name", "Divergence screen", "Pair spread"])


# ------------------------------ tab: matrix ------------------------------
with tab_matrix:
    src = st.radio("Source", ["Cached universe", "Live"], horizontal=True,
                   key="matrix_src",
                   help="Live pulls the tickers you type straight from Yahoo "
                        "(fine for a handful; no cache required).")
    if src == "Cached universe":
        default = [t for t in ["MSFT", "NVDA", "MU", "AVGO", "TSM"] if t in universe]
        picks = st.multiselect("Tickers", universe, default=default or universe[:5])
        live_px = None
    else:
        typed = st.text_input("Tickers (comma-separated)", "MSFT, NVDA, MU, AVGO, TSM")
        picks = [ce.normalize_to_yf(t) for t in typed.split(",") if t.strip()]
        live_px = picks

    if st.button("Build matrix", type="primary") and picks:
        try:
            if live_px is not None:
                px = fetch_live(tuple(sorted(live_px)),
                                sel_years if sel_years else ce.DEFAULT_YEARS)
                C = ce.corr_matrix(px, tickers=picks, start=sel_start,
                                   end=sel_end, years=sel_years)
            else:
                C = ce.corr_matrix(prices, tickers=picks, start=sel_start,
                                   end=sel_end, years=sel_years)
            if C.shape[0] < 2:
                st.error("Need at least two tickers with enough overlapping data.")
            else:
                st.dataframe(style_matrix(C), use_container_width=True)
                st.download_button(
                    "Download Excel (heatmap)",
                    to_excel_bytes(ce.export_matrix_to_excel, C),
                    file_name="correlation_matrix.xlsx")
        except Exception as e:
            st.error(str(e))


# --------------------------- tab: single name ---------------------------
with tab_single:
    if not universe:
        st.info("Build a cache to use this tab (it ranks against the full universe).")
    else:
        c1, c2 = st.columns([2, 1])
        tkr = c1.selectbox("Ticker", universe,
                           index=universe.index("NVDA") if "NVDA" in universe else 0)
        k = c2.slider("How many", 5, 30, 15)
        try:
            top, bot = ce.most_least_correlated(
                prices, tkr, k=k, start=sel_start, end=sel_end, years=sel_years)
            left, right = st.columns(2)
            left.subheader("Most correlated")
            left.dataframe(top.style.format("{:.3f}"), use_container_width=True)
            right.subheader("Least correlated")
            right.dataframe(bot.style.format("{:.3f}"), use_container_width=True)
        except Exception as e:
            st.error(str(e))


# -------------------------- tab: divergence screen --------------------------
with tab_screen:
    if not universe:
        st.info("Build a cache to run the screen (it scans every pair).")
    else:
        st.caption("Historically correlated pairs that have diverged recently.")
        a, b, c, d, e = st.columns(5)
        corr_th = a.slider("Min corr", 0.50, 0.95, 0.78, 0.01)
        z_th = b.slider("Min |z|", 1.0, 5.0, 3.0, 0.5)
        gap_th = c.slider("Min 20d gap", 0.0, 0.30, 0.10, 0.01)
        lookback = d.slider("Beta lookback", 30, 252, 120, 10)
        ret_n = e.slider("Return window", 5, 60, 20, 5)
        if st.button("Run screen", type="primary"):
            with st.spinner("Scanning pairs..."):
                try:
                    scr = ce.screen_divergences(
                        prices, corr_threshold=corr_th, z_threshold=z_th,
                        ret_gap_min=gap_th, beta_lookback=lookback, ret_n=ret_n,
                        start=sel_start, end=sel_end, years=sel_years, verbose=False)
                    st.write(f"**{len(scr)} candidate pairs**")
                    st.dataframe(scr.style.format(precision=3),
                                 use_container_width=True, height=460)
                    st.download_button(
                        "Download Excel", to_excel_bytes(ce.export_screen_to_excel, scr),
                        file_name="divergence_screen.xlsx")
                except Exception as ex:
                    st.error(str(ex))


# ------------------------------ tab: pair spread ------------------------------
with tab_pair:
    use_live = st.checkbox("Fetch this pair live", value=not universe)
    if universe and not use_live:
        c1, c2, c3 = st.columns(3)
        i = c1.selectbox("Leg 1", universe,
                         index=universe.index("MU") if "MU" in universe else 0)
        j = c2.selectbox("Leg 2", universe,
                         index=universe.index("NVDA") if "NVDA" in universe else 1)
        lb = c3.slider("Beta lookback", 30, 252, 120, 10, key="pair_lb")
        src_px = prices
    else:
        c1, c2, c3 = st.columns(3)
        i = ce.normalize_to_yf(c1.text_input("Leg 1", "MU"))
        j = ce.normalize_to_yf(c2.text_input("Leg 2", "NVDA"))
        lb = c3.slider("Beta lookback", 30, 252, 120, 10, key="pair_lb_live")
        src_px = None

    if st.button("Plot spread", type="primary"):
        try:
            if src_px is None:
                src_px = fetch_live((i, j) if i < j else (j, i),
                                    sel_years if sel_years else ce.DEFAULT_YEARS)
            m = ce.pair_spread(src_px, i, j, beta_lookback=lb)
            if m is None:
                st.error("Not enough overlapping data for that pair.")
            else:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Hedge ratio (beta)", f"{m['beta']:.2f}")
                mc2.metric("Spread z (latest)", f"{m['z_last']:.2f}")
                mc3.metric("Spread sigma", f"{m['spread_sigma']:.3f}")
                fig, ax = plt.subplots(figsize=(9, 3.2))
                m["z_series"].plot(ax=ax, color=ACCENT, lw=1.4)
                for y in (2, -2):
                    ax.axhline(y, ls="--", lw=0.8, color="#888")
                ax.axhline(0, lw=0.8, color="#444")
                ax.set_ylabel("z-score")
                ax.set_title(f"{i} vs {j} spread (beta lookback {lb}d)")
                ax.margins(x=0)
                st.pyplot(fig)
        except Exception as e:
            st.error(str(e))
