"""
backtest_fng_compare.py
Compare backtest results WITH vs WITHOUT Fear & Greed filter.

Fetches historical F&G (daily) from alternative.me, aligns to candle dates,
then runs simulate_pair twice and diffs the metrics.
"""

import sys
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

BOT_DIR = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

import config
if not hasattr(config, "SHORT_THRESHOLD"):  config.SHORT_THRESHOLD = 0.50
if not hasattr(config, "COVER_THRESHOLD"):  config.COVER_THRESHOLD = 0.38
if not hasattr(config, "ENABLE_SHORTING"):  config.ENABLE_SHORTING = True

from model_trainer import (
    fetch_binance_data,
    fetch_htf_data,
    add_features,
    _IsotonicCalibrated,
    model_file_for,
    short_model_file_for,
)

PAIRS           = config.SYMBOLS
INTERVAL        = config.INTERVAL
LOOKBACK        = config.LOOKBACK
BUY_THRESHOLD   = config.BUY_THRESHOLD
SELL_THRESHOLD  = config.SELL_THRESHOLD
SHORT_THRESHOLD = getattr(config, "SHORT_THRESHOLD", 0.50)
COVER_THRESHOLD = getattr(config, "COVER_THRESHOLD", 0.38)
TP_ATR_MULT     = config.TAKE_PROFIT_ATR_MULT
SL_ATR_MULT     = config.TRAIL_STOP_ATR_MULT
RISK_PER_TRADE  = config.RISK_PER_TRADE
STARTING_EQUITY = config.PAPER_STARTING_BALANCE

FNG_LONG_MIN  = config.FNG_LONG_MIN
FNG_LONG_MAX  = config.FNG_LONG_MAX
FNG_SHORT_MIN = config.FNG_SHORT_MIN
FNG_SHORT_MAX = config.FNG_SHORT_MAX


# ── fetch historical F&G ──────────────────────────────────────────────────────

def fetch_historical_fng(days: int = 730) -> pd.Series:
    """Returns a Series indexed by date (UTC) → F&G value (0–100)."""
    print(f"Fetching {days}d historical Fear & Greed...", end=" ", flush=True)
    r = requests.get(f"https://api.alternative.me/fng/?limit={days}&format=json", timeout=15)
    data = r.json()["data"]
    records = [(pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC").normalize(), int(d["value"]))
               for d in data]
    s = pd.Series(dict(records)).sort_index()
    print(f"{len(s)} days ({s.index[0].date()} to {s.index[-1].date()})")
    return s


# ── load models ───────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(pair: str):
    from model_trainer import train_and_save, train_and_save_short
    lp = Path(model_file_for(pair))
    sp = Path(short_model_file_for(pair))
    if not lp.exists():
        try: train_and_save(pair)
        except Exception as e: print(f"  [{pair}] long train failed: {e}")
    if not sp.exists():
        try: train_and_save_short(pair)
        except Exception as e: print(f"  [{pair}] short train failed: {e}")
    return _load_pkl(lp), _load_pkl(sp)


# ── simulate ──────────────────────────────────────────────────────────────────

def simulate_pair(pair: str, starting_equity: float,
                  fng_series: pd.Series | None, use_fng: bool) -> pd.DataFrame:
    long_payload, short_payload = load_models(pair)
    if long_payload is None and short_payload is None:
        return pd.DataFrame()

    df     = fetch_binance_data(pair, INTERVAL, LOOKBACK)
    htf_df = fetch_htf_data(pair, INTERVAL, LOOKBACK)
    df     = add_features(df, htf_df)

    if long_payload:
        feats_l = [c for c in long_payload["feature_cols"] if c in df.columns]
        df["prob"] = long_payload["model"].predict_proba(df[feats_l])[:, 1]
    else:
        df["prob"] = 0.0

    if short_payload:
        feats_s = [c for c in short_payload["feature_cols"] if c in df.columns]
        df["short_prob"] = short_payload["model"].predict_proba(df[feats_s])[:, 1]
    else:
        df["short_prob"] = 0.0

    trades = []
    equity = starting_equity
    pos    = None

    for ts, row in df.iterrows():
        close = row["close"]
        atr   = row.get("atr_pct", 0.01) * close

        # manage open position
        if pos is not None:
            exit_px = exit_type = None
            if pos["direction"] == "LONG":
                if row["low"]  <= pos["sl"]:             exit_px, exit_type = pos["sl"],  "SL"
                elif row["high"] >= pos["tp"]:           exit_px, exit_type = pos["tp"],  "TP"
                elif row["prob"] < SELL_THRESHOLD:       exit_px, exit_type = close, "ML-SELL"
            else:
                if row["high"] >= pos["sl"]:             exit_px, exit_type = pos["sl"],  "SL"
                elif row["low"]  <= pos["tp"]:           exit_px, exit_type = pos["tp"],  "TP"
                elif row["short_prob"] < COVER_THRESHOLD: exit_px, exit_type = close, "ML-COVER"

            if exit_px is not None:
                pnl = ((exit_px - pos["entry_px"]) if pos["direction"] == "LONG"
                       else (pos["entry_px"] - exit_px)) * pos["qty"]
                equity += pnl
                dur_h = round((ts - pos["entry_ts"]).total_seconds() / 3600, 1)
                trades.append({
                    "pair":       pair,
                    "direction":  pos["direction"],
                    "entry_time": pos["entry_ts"],
                    "exit_time":  ts,
                    "exit_type":  exit_type,
                    "pnl_usd":    round(pnl, 2),
                    "equity":     round(equity, 2),
                    "duration_h": dur_h,
                })
                pos = None

        # new entry
        if pos is None and atr > 0:
            risk_usd = equity * RISK_PER_TRADE

            # resolve F&G for this candle's date
            fng_ok_long  = True
            fng_ok_short = True
            if use_fng and fng_series is not None:
                day_key = pd.Timestamp(ts).normalize().tz_localize("UTC") if ts.tzinfo is None else pd.Timestamp(ts).normalize()
                # find closest available date (F&G is daily; candles are hourly)
                idx = fng_series.index.searchsorted(day_key, side="right") - 1
                if 0 <= idx < len(fng_series):
                    fng = int(fng_series.iloc[idx])
                    fng_ok_long  = FNG_LONG_MIN  <= fng <= FNG_LONG_MAX
                    fng_ok_short = FNG_SHORT_MIN <= fng <= FNG_SHORT_MAX
                else:
                    fng_ok_long = fng_ok_short = True  # no data — don't block

            if long_payload and fng_ok_long and row["prob"] >= BUY_THRESHOLD:
                sl  = close - atr * SL_ATR_MULT
                tp  = close + atr * TP_ATR_MULT
                qty = risk_usd / max(close - sl, 1e-9)
                pos = {"direction": "LONG",  "entry_ts": ts, "entry_px": close,
                       "sl": sl, "tp": tp, "qty": qty}

            elif short_payload and fng_ok_short and row["short_prob"] >= SHORT_THRESHOLD:
                sl  = close + atr * SL_ATR_MULT
                tp  = close - atr * TP_ATR_MULT
                qty = risk_usd / max(sl - close, 1e-9)
                pos = {"direction": "SHORT", "entry_ts": ts, "entry_px": close,
                       "sl": sl, "tp": tp, "qty": qty}

    return pd.DataFrame(trades)


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "trades": 0}
    total = len(df)
    wins  = (df["pnl_usd"] > 0).sum()
    net   = df["pnl_usd"].sum()
    wr    = wins / total * 100
    eq    = df["equity"]
    peak  = eq.cummax()
    mdd   = ((eq - peak) / peak * 100).min()
    avg_w = df.loc[df["pnl_usd"] > 0, "pnl_usd"].mean() if wins else 0
    avg_l = df.loc[df["pnl_usd"] <= 0, "pnl_usd"].mean() if (total - wins) else 0
    rr    = abs(avg_w / avg_l) if avg_l else float("inf")
    gw    = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gl    = df.loc[df["pnl_usd"] <= 0, "pnl_usd"].sum()
    pf    = abs(gw / gl) if gl else float("inf")
    final = eq.iloc[-1]
    ret   = (final / STARTING_EQUITY - 1) * 100
    long_wr  = (df.loc[df["direction"] == "LONG",  "pnl_usd"] > 0).mean() * 100 if (df["direction"] == "LONG").any()  else float("nan")
    short_wr = (df.loc[df["direction"] == "SHORT", "pnl_usd"] > 0).mean() * 100 if (df["direction"] == "SHORT").any() else float("nan")
    return {
        "label":    label,
        "trades":   total,
        "win_rate": wr,
        "net_pnl":  net,
        "return":   ret,
        "final_eq": final,
        "max_dd":   mdd,
        "rr":       rr,
        "pf":       pf,
        "long_wr":  long_wr,
        "short_wr": short_wr,
    }


def print_comparison(m_off: dict, m_on: dict):
    def fmt(key, fmt_str, suffix=""):
        v_off = m_off.get(key, float("nan"))
        v_on  = m_on.get(key, float("nan"))
        diff  = v_on - v_off if isinstance(v_on, float) and isinstance(v_off, float) else "—"
        d_str = f"{diff:+.2f}{suffix}" if isinstance(diff, float) else str(diff)
        return (f"  {key:<18} {fmt_str.format(v_off):<18} "
                f"{fmt_str.format(v_on):<18} {d_str}")

    print("\n" + "=" * 65)
    print(f"  F&G FILTER COMPARISON — {len(PAIRS)} pairs | {LOOKBACK}")
    print(f"  Settings: LONG {FNG_LONG_MIN}–{FNG_LONG_MAX} | SHORT {FNG_SHORT_MIN}–{FNG_SHORT_MAX}")
    print("=" * 65)
    print(f"  {'Metric':<18} {'F&G OFF':<18} {'F&G ON':<18} {'Diff'}")
    print("-" * 65)
    print(fmt("trades",   "{:.0f}"))
    print(fmt("win_rate", "{:.1f}%",  "%"))
    print(fmt("long_wr",  "{:.1f}%",  "%"))
    print(fmt("short_wr", "{:.1f}%",  "%"))
    print(fmt("net_pnl",  "${:,.0f}", ""))
    print(fmt("return",   "{:.1f}%",  "%"))
    print(fmt("max_dd",   "{:.1f}%",  "%"))
    print(fmt("rr",       "{:.2f}"))
    print(fmt("pf",       "{:.2f}"))
    print("=" * 65)
    delta_trades = m_on["trades"] - m_off["trades"]
    print(f"\n  F&G filter skipped {-delta_trades} trades ({-delta_trades/max(m_off['trades'],1)*100:.1f}% reduction)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pairs = sys.argv[1:] if len(sys.argv) > 1 else PAIRS

    fng_series = None
    try:
        fng_series = fetch_historical_fng(730)
    except Exception as e:
        print(f"WARNING: Could not fetch historical F&G: {e}")
        print("Running without F&G data — F&G ON will equal F&G OFF\n")

    print(f"\nRunning backtest for {len(pairs)} pairs...")

    all_off, all_on = [], []
    equity_off = equity_on = STARTING_EQUITY

    for pair in pairs:
        print(f"\n  [{pair}]", end=" ")
        for attempt in range(3):
            try:
                print("off...", end=" ", flush=True)
                df_off = simulate_pair(pair, equity_off, fng_series, use_fng=False)
                print("on...", end=" ", flush=True)
                df_on  = simulate_pair(pair, equity_on,  fng_series, use_fng=True)
                if not df_off.empty: equity_off = df_off["equity"].iloc[-1]
                if not df_on.empty:  equity_on  = df_on["equity"].iloc[-1]
                off_tr = len(df_off)
                on_tr  = len(df_on)
                print(f"done ({off_tr} vs {on_tr} trades)")
                all_off.append(df_off)
                all_on.append(df_on)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"retry ({e.__class__.__name__})...", end=" ", flush=True)
                    time.sleep(5)
                else:
                    print(f"FAILED: {e}")

    df_off_all = pd.concat([d for d in all_off if not d.empty], ignore_index=True) if all_off else pd.DataFrame()
    df_on_all  = pd.concat([d for d in all_on  if not d.empty], ignore_index=True) if all_on  else pd.DataFrame()

    m_off = metrics(df_off_all, "F&G OFF")
    m_on  = metrics(df_on_all,  "F&G ON")

    print_comparison(m_off, m_on)

    # direction breakdown
    if not df_off_all.empty and not df_on_all.empty:
        print("\n  -- Direction breakdown (F&G OFF vs ON) --")
        for d in ["LONG", "SHORT"]:
            g_off = df_off_all[df_off_all["direction"] == d]
            g_on  = df_on_all[df_on_all["direction"] == d]
            if g_off.empty and g_on.empty:
                continue
            wr_off = (g_off["pnl_usd"] > 0).mean() * 100 if len(g_off) else float("nan")
            wr_on  = (g_on["pnl_usd"] > 0).mean() * 100  if len(g_on)  else float("nan")
            n_off, n_on = len(g_off), len(g_on)
            print(f"  {d:<6}  OFF: {n_off:>4} trades WR={wr_off:.1f}%  "
                  f"ON: {n_on:>4} trades WR={wr_on:.1f}%  "
                  f"(skipped {n_off - n_on})")

    # save
    if not df_off_all.empty:
        df_off_all.to_csv(BOT_DIR / "fng_compare_off.csv", index=False)
    if not df_on_all.empty:
        df_on_all.to_csv(BOT_DIR / "fng_compare_on.csv", index=False)
    print("\n  Saved: fng_compare_off.csv / fng_compare_on.csv")


if __name__ == "__main__":
    main()
