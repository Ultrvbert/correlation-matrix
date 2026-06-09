"""
corr_engine.py
==============
Correlation / pairs-divergence engine for equity research.

Front-end agnostic: import these functions from a CLI, a Streamlit app, or an
Excel macro. No Jupyter, no ipywidgets, no hardcoded paths.

Two workloads, deliberately separated:
  1. BUILD (slow, network)  -> download a universe once, cache daily prices.
  2. QUERY (instant, offline) -> read the cache, recompute correlation over any
     window, run the divergence screen, export to Excel.

yfinance is imported lazily so the query/export paths work with no network
and no yfinance installed (useful when bundling a query-only front-end).
"""

from __future__ import annotations

import datetime as dt
import re
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Defaults (override via function args / CLI; do NOT hardcode user paths here)
# ----------------------------------------------------------------------------
DEFAULT_YEARS = 5            # how much history to cache on BUILD
MIN_OBS = 252                # ~1y of trading days required to keep a name
BATCH_SIZE = 75              # small batches survive Yahoo throttling
MAX_RETRIES = 4
SLEEP_SEC = 1.0
DELISTED_CUTOFF_DAYS = 120   # last quote must be within this many days of end
RANDOM_SEED = 42


# ----------------------------------------------------------------------------
# Ticker normalization
# ----------------------------------------------------------------------------
def normalize_to_yf(sym: str) -> str:
    """BRK.B -> BRK-B, 'bf b' -> BF-B, strip junk. yfinance-friendly."""
    s = str(sym).strip().upper()
    s = s.replace("/", "-").replace(".", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


# ----------------------------------------------------------------------------
# Data layer (network)
# ----------------------------------------------------------------------------
def _extract_close(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the adjusted close out of a yf.download frame.

    With auto_adjust=True yfinance returns 'Close' already adjusted (no
    'Adj Close' column), so we only look for 'Close'.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        field = "Close" if "Close" in lvl0 else lvl0[0]
        return df[field].copy()
    return df.copy()


def download_prices(tickers, start=None, end=None, years=DEFAULT_YEARS,
                    batch_size=BATCH_SIZE, verbose=True):
    """Robust batched download of daily adjusted closes.

    Randomizes order (avoids alphabet bias under throttling), retries per
    batch, then falls back to single-ticker pulls for failed batches.
    Returns (prices_df, failed_list).
    """
    import yfinance as yf  # lazy: only needed when actually downloading

    tickers = list(dict.fromkeys(normalize_to_yf(t) for t in tickers))
    random.seed(RANDOM_SEED)
    random.shuffle(tickers)

    end = end or dt.date.today()
    if start is None:
        start = end - dt.timedelta(days=int(365.25 * years) + 10)

    def _pull(syms, threads):
        return _extract_close(yf.download(
            syms, start=str(start), end=str(end), interval="1d",
            auto_adjust=True, progress=False, group_by="column", threads=threads,
        ))

    frames, failed = [], []
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(tickers), batch_size), 1):
        batch = tickers[i:i + batch_size]
        data = pd.DataFrame()
        for attempt in range(MAX_RETRIES):
            try:
                data = _pull(batch, threads=False)
                if not data.empty:
                    break
            except Exception:
                pass
            time.sleep(SLEEP_SEC * (attempt + 1))

        # single-ticker fallback for whatever the batch missed
        if data.empty:
            cols = []
            for t in batch:
                try:
                    px = _extract_close(yf.download(
                        t, start=str(start), end=str(end), interval="1d",
                        auto_adjust=True, progress=False, threads=False))
                    if px.empty:
                        failed.append(t)
                        continue
                    if px.ndim == 1:
                        px = px.to_frame(name=t)
                    elif t not in px.columns and px.shape[1] == 1:
                        px.columns = [t]
                    cols.append(px[[t]] if t in px.columns else px)
                except Exception:
                    failed.append(t)
            if cols:
                data = pd.concat(cols, axis=1)

        if not data.empty:
            frames.append(data)
        if verbose:
            print(f"  batch {bi}/{n_batches} -> {0 if data.empty else data.shape[1]} cols")

    if not frames:
        return pd.DataFrame(), sorted(pd.unique(failed))

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all").sort_index().sort_index(axis=1)
    return prices, sorted(set(failed))


def filter_active(prices: pd.DataFrame, min_obs=MIN_OBS,
                  cutoff_days=DELISTED_CUTOFF_DAYS):
    """Drop names that are inactive (stale last quote) or too short."""
    if prices.empty:
        return prices
    latest = prices.index.max()
    cutoff_last = latest - pd.Timedelta(days=cutoff_days)
    last_valid = prices.apply(lambda s: s.last_valid_index(), axis=0)
    n_obs = prices.count(axis=0)
    keep = [c for c in prices.columns
            if last_valid[c] is not pd.NaT
            and last_valid[c] >= cutoff_last
            and n_obs[c] >= min_obs]
    return prices[keep]


# ----------------------------------------------------------------------------
# Cache layer
# ----------------------------------------------------------------------------
def build_cache(tickers, data_dir, years=DEFAULT_YEARS, active_only=True,
                verbose=True):
    """Download a universe once and persist prices to <data_dir>/prices.parquet.

    Returns the prices DataFrame. Run this on a schedule or on demand; the
    query/export functions read the cache and never touch the network.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    prices, failed = download_prices(tickers, years=years, verbose=verbose)
    if active_only:
        prices = filter_active(prices)
    prices.to_parquet(data_dir / "prices.parquet")
    if failed:
        pd.Series(failed, name="ticker").to_csv(
            data_dir / "download_failed.csv", index=False)
    if verbose:
        print(f"cached {prices.shape[1]} tickers x {prices.shape[0]} days "
              f"-> {data_dir/'prices.parquet'}  ({len(failed)} failed)")
    return prices


def load_cache(data_dir):
    """Load cached daily prices. Raises if the cache hasn't been built."""
    p = Path(data_dir) / "prices.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"No price cache at {p}. Run build_cache(...) first.")
    return pd.read_parquet(p)


def fetch_live(tickers, years=DEFAULT_YEARS):
    """Live pull for a small ad-hoc basket (fast & reliable for a handful)."""
    prices, _ = download_prices(tickers, years=years, batch_size=len(tickers),
                                verbose=False)
    return prices


# ----------------------------------------------------------------------------
# Compute layer (instant, offline)
# ----------------------------------------------------------------------------
def _slice_window(prices, start=None, end=None, years=None):
    """Subset price history to a window. years overrides start if given."""
    if years is not None:
        end_d = pd.to_datetime(end) if end else prices.index.max()
        start = end_d - pd.Timedelta(days=int(365.25 * years))
    out = prices
    if start is not None:
        out = out.loc[pd.to_datetime(start):]
    if end is not None:
        out = out.loc[:pd.to_datetime(end)]
    return out


def returns_from_prices(prices):
    # fill_method=None is the important bit: forward-filling before diffing
    # silently inflates correlations. Keep it.
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def corr_matrix(prices, tickers=None, start=None, end=None, years=None,
                min_obs=20):
    """Correlation matrix over a chosen window for a chosen set of names.

    tickers=None uses every cached name; pass a list for an N-name matrix.
    The window is what makes timeframe selectable with zero re-download.
    """
    px = _slice_window(prices, start, end, years)
    if tickers is not None:
        want = [normalize_to_yf(t) for t in tickers]
        missing = [t for t in want if t not in px.columns]
        px = px[[t for t in want if t in px.columns]]
        if missing:
            print(f"  not in cache (ignored): {missing}")
    rets = returns_from_prices(px)
    rets = rets.loc[:, rets.count() >= min_obs]
    return rets.corr(method="pearson")


def most_least_correlated(prices, ticker, k=15, start=None, end=None,
                          years=None):
    """Most- and least-correlated names to `ticker` over a window."""
    t = normalize_to_yf(ticker)
    corr = corr_matrix(prices, start=start, end=end, years=years)
    if t not in corr.columns:
        from difflib import get_close_matches
        sugg = get_close_matches(t, list(corr.columns), n=5, cutoff=0.5)
        raise ValueError(f"{t} not in universe. Did you mean: {sugg or 'n/a'}")
    s = corr.loc[t].drop(index=t, errors="ignore").dropna().sort_values(ascending=False)
    return (s.head(k).rename("corr").to_frame(),
            s.tail(k).iloc[::-1].rename("corr").to_frame())


def top_pairs(prices, threshold=0.75, start=None, end=None, years=None):
    """All unique pairs with correlation >= threshold, ranked descending."""
    corr = corr_matrix(prices, start=start, end=end, years=years)
    C = corr.values
    iu = np.triu_indices_from(C, k=1)
    pairs = pd.DataFrame({
        "i": corr.index[iu[0]], "j": corr.columns[iu[1]], "corr": C[iu],
    }).dropna()
    pairs = pairs[pairs["corr"] >= threshold]
    return pairs.sort_values("corr", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pairs-divergence screen (the short/long idea engine)
# ----------------------------------------------------------------------------
def _ols_beta(y: np.ndarray, x: np.ndarray):
    """alpha, beta for y ~ alpha + beta*x via numpy lstsq (no statsmodels)."""
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def pair_spread(prices, i, j, beta_lookback=120):
    """Spread z-score for a pair on log prices. RETURNS a metrics dict
    (the notebook's plot_pair returned None — this fixes that)."""
    i, j = normalize_to_yf(i), normalize_to_yf(j)
    px = prices[[i, j]].dropna()
    if px.shape[0] < 40:
        return None
    pxb = px.iloc[-beta_lookback:] if px.shape[0] > beta_lookback else px
    li, lj = np.log(pxb[i].to_numpy()), np.log(pxb[j].to_numpy())
    if len(np.unique(li)) < 5 or len(np.unique(lj)) < 5:
        return None
    alpha, beta = _ols_beta(li, lj)
    spread = li - (alpha + beta * lj)
    mu, sigma = spread.mean(), spread.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return None
    z_series = pd.Series((spread - mu) / sigma, index=pxb.index)
    return {"i": i, "j": j, "alpha": alpha, "beta": beta,
            "z_last": float(z_series.iloc[-1]), "spread_sigma": float(sigma),
            "z_series": z_series}


def screen_divergences(prices, corr_threshold=0.78, z_threshold=3.0,
                       ret_gap_min=0.10, beta_lookback=120, ret_n=20,
                       max_pairs=6000, start=None, end=None, years=None,
                       verbose=True):
    """High historical correlation + recent divergence => candidate pairs.

    For each pair above corr_threshold, fit a hedge ratio, compute the spread
    z-score and a recent N-day return gap; keep pairs where |z| >= z_threshold
    OR the return gap >= ret_gap_min. Ranked by |z|.
    """
    px = _slice_window(prices, start, end, years)
    pairs = top_pairs(px, threshold=corr_threshold)
    if pairs.empty:
        raise ValueError("No pairs meet corr_threshold; lower it.")
    if max_pairs:
        pairs = pairs.head(max_pairs)

    rows = []
    for n, (_, r) in enumerate(pairs.iterrows(), 1):
        m = pair_spread(px, r["i"], r["j"], beta_lookback=beta_lookback)
        if m is None:
            continue
        sub = px[[m["i"], m["j"]]].dropna().iloc[-(ret_n + 1):]
        if sub.shape[0] < ret_n // 2:
            continue
        r_i = float(np.log(sub[m["i"]].iloc[-1] / sub[m["i"]].iloc[0]))
        r_j = float(np.log(sub[m["j"]].iloc[-1] / sub[m["j"]].iloc[0]))
        gap = abs(r_i - r_j)
        if abs(m["z_last"]) >= z_threshold or gap >= ret_gap_min:
            rows.append({"i": m["i"], "j": m["j"], "corr": r["corr"],
                         "beta": m["beta"], "z_last": m["z_last"],
                         f"ret_{ret_n}d_i": r_i, f"ret_{ret_n}d_j": r_j,
                         "ret_gap": gap})
        if verbose and n % 500 == 0:
            print(f"  evaluated {n}/{len(pairs)} pairs...")

    cols = ["i", "j", "corr", "beta", "z_last",
            f"ret_{ret_n}d_i", f"ret_{ret_n}d_j", "ret_gap"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["abs_z"] = out["z_last"].abs()
    out = out.sort_values(["abs_z", "corr", "ret_gap"],
                          ascending=[False, False, False])
    return out[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Export layer (Excel for colleagues)
# ----------------------------------------------------------------------------
def export_matrix_to_excel(corr: pd.DataFrame, path, sheet="Correlation"):
    """Write a correlation matrix with a red-white-green heatmap."""
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        corr.round(3).to_excel(xl, sheet_name=sheet)
        ws = xl.sheets[sheet]
        n = corr.shape[0]
        first, last = "B2", f"{get_column_letter(n + 1)}{n + 1}"
        ws.conditional_formatting.add(f"{first}:{last}", ColorScaleRule(
            start_type="num", start_value=-1, start_color="F8696B",
            mid_type="num", mid_value=0, mid_color="FFFFFF",
            end_type="num", end_value=1, end_color="63BE7B"))
        ws.freeze_panes = "B2"
    return path


def export_screen_to_excel(df: pd.DataFrame, path, sheet="Divergences"):
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.round(4).to_excel(xl, sheet_name=sheet, index=False)
        ws = xl.sheets[sheet]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col), default=8) + 2
            ws.column_dimensions[col[0].column_letter].width = min(width, 24)
    return path
