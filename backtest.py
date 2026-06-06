"""
backtest.py — Full 2-year P&L backtest using the bot's actual models + logic.

Replicates exactly:
  - ATR-based SL (1.5×ATR) and TP (2.5×ATR)
  - ML-exit: close long when prob < SELL_THRESHOLD (0.38)
              close short when short_prob < COVER_THRESHOLD (0.38)
  - 1% risk-per-trade position sizing
  - Both LONG and SHORT models

⚠  Caveat: models were trained on this same 730-day window.
   First 85% of data has look-ahead bias. The last ~110 days
   (15% test set) is the only truly clean out-of-sample period.
   Results on the full window are optimistic — treat as an upper bound.

Usage:
    python backtest.py              # all 10 pairs
    python backtest.py XRPUSDT     # single pair
"""

import os
import sys
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- point at the bot's directory so imports resolve --------------------------
BOT_DIR = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

import config

# patch missing config attrs before model_trainer imports them
if not hasattr(config, "SHORT_THRESHOLD"):  config.SHORT_THRESHOLD  = 0.50
if not hasattr(config, "COVER_THRESHOLD"):  config.COVER_THRESHOLD  = 0.38
if not hasattr(config, "ENABLE_SHORTING"):  config.ENABLE_SHORTING  = True

from model_trainer import (
    fetch_binance_data,
    fetch_htf_data,
    add_features,
    _IsotonicCalibrated,   # must be in scope for pickle.load to work
)

# -- constants (mirrors config.py) --------------------------------------------
PAIRS = config.SYMBOLS          # all 10 pairs
INTERVAL   = config.INTERVAL    # 1h
LOOKBACK   = config.LOOKBACK    # 730 day ago UTC

BUY_THRESHOLD   = config.BUY_THRESHOLD                          # 0.50
SELL_THRESHOLD  = config.SELL_THRESHOLD                         # 0.38
SHORT_THRESHOLD = getattr(config, "SHORT_THRESHOLD",  0.50)    # 0.50
COVER_THRESHOLD = getattr(config, "COVER_THRESHOLD",  0.38)    # 0.38

TP_ATR_MULT    = config.TAKE_PROFIT_ATR_MULT   # 2.5
SL_ATR_MULT    = config.TRAIL_STOP_ATR_MULT    # 1.5
RISK_PER_TRADE = config.RISK_PER_TRADE          # 0.01

STARTING_EQUITY = config.PAPER_STARTING_BALANCE  # 10 000


# -- load saved model pkl files -----------------------------------------------

def _load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(pair: str) -> tuple:
    """Returns (long_payload, short_payload). Trains on-the-fly if missing."""
    from model_trainer import train_and_save, train_and_save_short, model_file_for, short_model_file_for

    long_path  = Path(model_file_for(pair))
    short_path = Path(short_model_file_for(pair))

    if not long_path.exists():
        print(f"  [{pair}] long model missing — training now...")
        try:
            train_and_save(pair)
        except Exception as e:
            print(f"  [{pair}] long training failed: {e}")

    if not short_path.exists():
        print(f"  [{pair}] short model missing — training now...")
        try:
            train_and_save_short(pair)
        except Exception as e:
            print(f"  [{pair}] short training failed: {e}")

    return _load_pkl(long_path), _load_pkl(short_path)


# -- core simulation -----------------------------------------------------------

def simulate_pair(pair: str, starting_equity: float) -> pd.DataFrame:
    long_payload, short_payload = load_models(pair)
    if long_payload is None and short_payload is None:
        print(f"  [{pair}] no model files found — skipping")
        return pd.DataFrame()

    print(f"  [{pair}] fetching data...", end=" ", flush=True)
    df     = fetch_binance_data(pair, INTERVAL, LOOKBACK)
    htf_df = fetch_htf_data(pair, INTERVAL, LOOKBACK)
    df     = add_features(df, htf_df)
    print(f"{len(df)} candles")

    # -- generate probabilities ------------------------------------------------
    if long_payload:
        feats_l = long_payload["feature_cols"]
        # keep only cols present in df (model may have been trained on a subset)
        feats_l = [c for c in feats_l if c in df.columns]
        df["prob"] = long_payload["model"].predict_proba(df[feats_l])[:, 1]
    else:
        df["prob"] = 0.0

    if short_payload:
        feats_s = short_payload["feature_cols"]
        feats_s = [c for c in feats_s if c in df.columns]
        df["short_prob"] = short_payload["model"].predict_proba(df[feats_s])[:, 1]
    else:
        df["short_prob"] = 0.0

    # -- event loop: iterate candle-by-candle ----------------------------------
    trades  = []
    equity  = starting_equity
    pos     = None  # None = flat; dict = open position

    for ts, row in df.iterrows():
        close = row["close"]
        atr   = row.get("atr_pct", 0.01) * close   # atr_pct × price = ATR in $

        # -- manage open position ----------------------------------------------
        if pos is not None:
            exit_px = exit_type = None

            if pos["direction"] == "LONG":
                if row["low"] <= pos["sl"]:          # SL hit (use candle low)
                    exit_px, exit_type = pos["sl"], "SL"
                elif row["high"] >= pos["tp"]:        # TP hit (use candle high)
                    exit_px, exit_type = pos["tp"], "TP"
                elif row["prob"] < SELL_THRESHOLD:    # ML says exit
                    exit_px, exit_type = close, "ML-SELL"

            else:  # SHORT
                if row["high"] >= pos["sl"]:          # SL hit
                    exit_px, exit_type = pos["sl"], "SL"
                elif row["low"] <= pos["tp"]:         # TP hit
                    exit_px, exit_type = pos["tp"], "TP"
                elif row["short_prob"] < COVER_THRESHOLD:
                    exit_px, exit_type = close, "ML-COVER"

            if exit_px is not None:
                if pos["direction"] == "LONG":
                    pnl = (exit_px - pos["entry_px"]) * pos["qty"]
                else:
                    pnl = (pos["entry_px"] - exit_px) * pos["qty"]

                equity += pnl
                dur_h = round((ts - pos["entry_ts"]).total_seconds() / 3600, 1)
                trades.append({
                    "pair":       pair,
                    "direction":  pos["direction"],
                    "entry_time": pos["entry_ts"],
                    "entry_px":   pos["entry_px"],
                    "sl":         pos["sl"],
                    "tp":         pos["tp"],
                    "exit_time":  ts,
                    "exit_px":    round(exit_px, 6),
                    "exit_type":  exit_type,
                    "pnl_usd":    round(pnl, 2),
                    "pnl_pct":    round(pnl / (pos["entry_px"] * pos["qty"]) * 100, 2),
                    "equity":     round(equity, 2),
                    "duration_h": dur_h,
                })
                pos = None

        # -- check for new entry (only when flat) ------------------------------
        if pos is None and atr > 0:
            risk_usd = equity * RISK_PER_TRADE

            if long_payload and row["prob"] >= BUY_THRESHOLD:
                sl  = close - atr * SL_ATR_MULT
                tp  = close + atr * TP_ATR_MULT
                qty = risk_usd / (close - sl)
                pos = {
                    "direction": "LONG",
                    "entry_ts":  ts,
                    "entry_px":  close,
                    "sl": sl, "tp": tp, "qty": qty,
                }

            elif short_payload and row["short_prob"] >= SHORT_THRESHOLD:
                sl  = close + atr * SL_ATR_MULT
                tp  = close - atr * TP_ATR_MULT
                qty = risk_usd / (sl - close)
                pos = {
                    "direction": "SHORT",
                    "entry_ts":  ts,
                    "entry_px":  close,
                    "sl": sl, "tp": tp, "qty": qty,
                }

    return pd.DataFrame(trades)


# -- reporting -----------------------------------------------------------------

def print_report(all_trades: pd.DataFrame):
    if all_trades.empty:
        print("No trades generated.")
        return

    total  = len(all_trades)
    wins   = all_trades[all_trades["pnl_usd"] > 0]
    losses = all_trades[all_trades["pnl_usd"] <= 0]
    net    = all_trades["pnl_usd"].sum()
    wr     = len(wins) / total * 100
    avg_w  = wins["pnl_usd"].mean()   if len(wins)   else 0
    avg_l  = losses["pnl_usd"].mean() if len(losses) else 0
    rr     = abs(avg_w / avg_l)       if avg_l       else float("inf")

    # max drawdown from equity curve
    eq    = all_trades["equity"]
    peak  = eq.cummax()
    dd    = (eq - peak) / peak * 100
    mdd   = dd.min()

    # profit factor
    gross_win  = wins["pnl_usd"].sum()   if len(wins)   else 0
    gross_loss = losses["pnl_usd"].sum() if len(losses) else 1
    pf = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")

    # consecutive wins/losses
    pnl_signs = (all_trades["pnl_usd"] > 0).astype(int).tolist()
    max_consec_w = max_consec_l = cur = 0
    for i, s in enumerate(pnl_signs):
        cur = cur + 1 if (i == 0 or pnl_signs[i-1] == s) else 1
        if s == 1: max_consec_w = max(max_consec_w, cur)
        else:      max_consec_l = max(max_consec_l, cur)

    first_ts = all_trades["entry_time"].min().strftime("%Y-%m-%d")
    last_ts  = all_trades["exit_time"].max().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT — {len(all_trades['pair'].unique())} PAIRS")
    print(f"  Period : {first_ts} to {last_ts}")
    print(f"  Start  : ${STARTING_EQUITY:,.0f}   Final: ${eq.iloc[-1]:,.2f}")
    print("=" * 60)
    print(f"  Total trades      : {total}")
    print(f"  Win rate          : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net P&L           : ${net:+,.2f}  ({net/STARTING_EQUITY*100:+.1f}%)")
    print(f"  Avg win / loss    : ${avg_w:+.2f} / ${avg_l:+.2f}")
    print(f"  Reward:Risk       : {rr:.2f}")
    print(f"  Profit factor     : {pf:.2f}")
    print(f"  Max drawdown      : {mdd:.1f}%")
    print(f"  Max consec wins   : {max_consec_w}")
    print(f"  Max consec losses : {max_consec_l}")

    print("\n  -- By pair ------------------------------------------")
    for pair, g in all_trades.groupby("pair"):
        pw  = len(g[g["pnl_usd"] > 0]) / len(g) * 100
        pnl = g["pnl_usd"].sum()
        print(f"  {pair:<12} {len(g):>4} trades  "
              f"P&L: ${pnl:>+9.2f}  WR: {pw:>5.1f}%")

    print("\n  -- By direction -------------------------------------")
    for d, g in all_trades.groupby("direction"):
        pw  = len(g[g["pnl_usd"] > 0]) / len(g) * 100
        pnl = g["pnl_usd"].sum()
        print(f"  {d:<8}  {len(g):>4} trades  "
              f"P&L: ${pnl:>+9.2f}  WR: {pw:>5.1f}%")

    print("\n  -- By exit type -------------------------------------")
    for et, g in all_trades.groupby("exit_type"):
        pw  = len(g[g["pnl_usd"] > 0]) / len(g) * 100
        pnl = g["pnl_usd"].sum()
        print(f"  {et:<15}  {len(g):>4} trades  "
              f"P&L: ${pnl:>+9.2f}  WR: {pw:>5.1f}%")

    print("=" * 60)
    print(f"\n  NOTE: Models trained on this same 730-day window.")
    print(f"  Only the last ~15% (~110 days) is truly out-of-sample.")
    print(f"  Full-window results are OPTIMISTIC (look-ahead present).")

    # save CSVs
    out_trades = BOT_DIR / "backtest_trades.csv"
    all_trades.to_csv(out_trades, index=False)

    # monthly P&L breakdown
    all_trades["month"] = all_trades["exit_time"].dt.to_period("M")
    monthly = all_trades.groupby("month")["pnl_usd"].sum()
    out_monthly = BOT_DIR / "backtest_monthly.csv"
    monthly.to_csv(out_monthly)

    print(f"\n  backtest_trades.csv  -> {out_trades}")
    print(f"  backtest_monthly.csv -> {out_monthly}")

    print("\n  -- Monthly P&L --------------------------------------")
    for month, pnl in monthly.items():
        bar = "+" * int(abs(pnl) / 20) if pnl > 0 else "-" * int(abs(pnl) / 20)
        sign = "+" if pnl >= 0 else ""
        print(f"  {month}  ${sign}{pnl:>8.2f}  {bar}")


# -- entry point ---------------------------------------------------------------

def main():
    pairs = sys.argv[1:] if len(sys.argv) > 1 else PAIRS
    print(f"\nBacktest: {len(pairs)} pairs | {LOOKBACK} | "
          f"equity ${STARTING_EQUITY:,.0f} | risk {RISK_PER_TRADE:.0%}/trade")
    print(f"SL: {SL_ATR_MULT}x ATR | TP: {TP_ATR_MULT}x ATR | "
          f"BUY>={BUY_THRESHOLD} | SHORT>={SHORT_THRESHOLD}\n")

    all_trades = []
    equity = STARTING_EQUITY
    for pair in pairs:
        for attempt in range(3):
            try:
                df = simulate_pair(pair, equity)
                if not df.empty:
                    equity = df["equity"].iloc[-1]
                    all_trades.append(df)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  [{pair}] attempt {attempt+1} failed ({e.__class__.__name__}), retrying...")
                    time.sleep(5)
                else:
                    print(f"  [{pair}] skipped after 3 failures: {e.__class__.__name__}")

    if all_trades:
        print_report(pd.concat(all_trades, ignore_index=True).sort_values("entry_time"))
    else:
        print("No trades found. Check that model .pkl files exist in:", BOT_DIR)


if __name__ == "__main__":
    main()
