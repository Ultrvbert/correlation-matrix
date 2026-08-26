"""
corr_tool.py
============
Front-end for corr_engine. Two ways to run:

  Scripted (analysts / scheduled refresh):
    python corr_tool.py build  --tickers-csv "Russell 3000 Ticker List.csv" --years 5
    python corr_tool.py matrix --tickers MSFT,NVDA,MU,AVGO,TSM --years 1
    python corr_tool.py matrix --tickers MSFT,NVDA,MU --years 1 --live   # no cache needed
    python corr_tool.py corr   --ticker NVDA --years 3
    python corr_tool.py screen --corr 0.80 --z 3 --years 3

  Double-clicked exe (non-Python colleagues):
    Run with no arguments -> an interactive menu prompts for everything.

Set CORR_DATA_DIR (env var) or pass --data-dir to point at the shared cache,
"""

import argparse
import os
import sys
from pathlib import Path

import corr_engine as ce

DATA_DIR = os.environ.get("CORR_DATA_DIR", "corr_data")


def _read_ticker_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    for c in ["yf_ticker", "Ticker", "Symbol", "symbol", "Ticker Symbol"]:
        if c in df.columns:
            return df[c].dropna().astype(str).tolist()
    # fall back to first column
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_prices(data_dir, tickers=None, live=False, years=ce.DEFAULT_YEARS):
    if live:
        if not tickers:
            sys.exit("--live needs --tickers")
        print(f"fetching {len(tickers)} tickers live...")
        return ce.fetch_live(tickers, years=years)
    return ce.load_cache(data_dir)


# ----------------------------- commands -----------------------------
def cmd_build(a):
    tickers = _read_ticker_csv(a.tickers_csv)
    print(f"building cache for {len(tickers)} tickers ({a.years}y) -> {a.data_dir}")
    ce.build_cache(tickers, a.data_dir, years=a.years)


def cmd_matrix(a):
    tickers = a.tickers.split(",") if a.tickers else None
    prices = _load_prices(a.data_dir, tickers, a.live, a.years or ce.DEFAULT_YEARS)
    C = ce.corr_matrix(prices, tickers=tickers, start=a.start, end=a.end,
                       years=a.years)
    print(C.round(3).to_string())
    out = a.out or "correlation_matrix.xlsx"
    ce.export_matrix_to_excel(C, out)
    print(f"\nsaved -> {out}")


def cmd_corr(a):
    prices = _load_prices(a.data_dir, [a.ticker] if a.live else None,
                          a.live, a.years or ce.DEFAULT_YEARS)
    top, bot = ce.most_least_correlated(prices, a.ticker, k=a.k,
                                        start=a.start, end=a.end, years=a.years)
    print(f"\nMost correlated to {a.ticker}:\n{top.round(3).to_string()}")
    print(f"\nLeast correlated to {a.ticker}:\n{bot.round(3).to_string()}")


def cmd_screen(a):
    prices = ce.load_cache(a.data_dir)
    scr = ce.screen_divergences(prices, corr_threshold=a.corr, z_threshold=a.z,
                                ret_gap_min=a.gap, start=a.start, end=a.end,
                                years=a.years)
    print(f"\n{len(scr)} divergent high-corr pairs:\n{scr.head(25).round(3).to_string()}")
    out = a.out or "divergence_screen.xlsx"
    ce.export_screen_to_excel(scr, out)
    print(f"\nsaved -> {out}")


# ----------------------------- interactive menu -----------------------------
def interactive():
    print("=" * 56)
    print("  Correlation / Pairs Tool")
    print("=" * 56)
    data_dir = input(f"Cache folder [{DATA_DIR}]: ").strip() or DATA_DIR
    print("\nWhat do you want?")
    print("  1) Correlation matrix for a few tickers")
    print("  2) Most/least correlated to one ticker")
    print("  3) Divergence screen (pair ideas)")
    choice = input("Choose 1-3: ").strip()

    yrs = input("Timeframe in years [3]: ").strip()
    years = float(yrs) if yrs else 3.0

    try:
        if choice == "1":
            tk = input("Tickers (comma-separated): ").upper().replace(" ", "")
            tickers = [t for t in tk.split(",") if t]
            live = input("No cache built yet? Fetch live? [y/N]: ").lower().startswith("y")
            prices = _load_prices(data_dir, tickers, live, years)
            C = ce.corr_matrix(prices, tickers=tickers, years=years)
            print("\n" + C.round(3).to_string())
            out = input("Save to [correlation_matrix.xlsx]: ").strip() or "correlation_matrix.xlsx"
            ce.export_matrix_to_excel(C, out)
            print(f"saved -> {out}")
        elif choice == "2":
            t = input("Ticker: ").strip().upper()
            prices = ce.load_cache(data_dir)
            top, bot = ce.most_least_correlated(prices, t, years=years)
            print(f"\nMost correlated to {t}:\n{top.round(3).to_string()}")
            print(f"\nLeast correlated to {t}:\n{bot.round(3).to_string()}")
        elif choice == "3":
            prices = ce.load_cache(data_dir)
            scr = ce.screen_divergences(prices, years=years)
            print(f"\n{len(scr)} pairs:\n{scr.head(25).round(3).to_string()}")
            out = input("Save to [divergence_screen.xlsx]: ").strip() or "divergence_screen.xlsx"
            ce.export_screen_to_excel(scr, out)
            print(f"saved -> {out}")
        else:
            print("Nothing selected.")
    except Exception as e:
        print(f"\nError: {e}")
    input("\nPress Enter to close...")


def build_parser():
    p = argparse.ArgumentParser(description="Correlation / pairs tool")
    p.add_argument("--data-dir", default=DATA_DIR)
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="download a universe and cache prices")
    b.add_argument("--tickers-csv", required=True)
    b.add_argument("--years", type=float, default=ce.DEFAULT_YEARS)
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("matrix", help="N-ticker correlation matrix -> Excel")
    m.add_argument("--tickers")
    m.add_argument("--years", type=float)
    m.add_argument("--start"); m.add_argument("--end")
    m.add_argument("--live", action="store_true")
    m.add_argument("--out")
    m.set_defaults(func=cmd_matrix)

    c = sub.add_parser("corr", help="most/least correlated to one ticker")
    c.add_argument("--ticker", required=True)
    c.add_argument("--k", type=int, default=15)
    c.add_argument("--years", type=float)
    c.add_argument("--start"); c.add_argument("--end")
    c.add_argument("--live", action="store_true")
    c.set_defaults(func=cmd_corr)

    s = sub.add_parser("screen", help="divergence screen -> Excel")
    s.add_argument("--corr", type=float, default=0.78)
    s.add_argument("--z", type=float, default=3.0)
    s.add_argument("--gap", type=float, default=0.10)
    s.add_argument("--years", type=float)
    s.add_argument("--start"); s.add_argument("--end")
    s.add_argument("--out")
    s.set_defaults(func=cmd_screen)
    return p


def main():
    if len(sys.argv) == 1:
        interactive()           # double-clicked exe path
        return
    args = build_parser().parse_args()
    if not getattr(args, "cmd", None):
        build_parser().print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
