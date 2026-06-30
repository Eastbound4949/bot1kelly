"""
config.py — All settings in one place.
"""

import os

# ─── Exchange ──────────────────────────────────────────────────
EXCHANGE = os.environ.get("EXCHANGE", "kucoin")   # kucoin works on Railway US servers

# ─── Exchange API keys ─────────────────────────────────────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY",    "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BYBIT_API_KEY      = os.environ.get("BYBIT_API_KEY",      "")
BYBIT_API_SECRET   = os.environ.get("BYBIT_API_SECRET",   "")
OKX_API_KEY        = os.environ.get("OKX_API_KEY",        "")
OKX_API_SECRET     = os.environ.get("OKX_API_SECRET",     "")
OKX_PASSPHRASE     = os.environ.get("OKX_PASSPHRASE",     "")

# ─── Telegram ──────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─── Trading settings ──────────────────────────────────────────
SYMBOL   = os.environ.get("HEARTBEAT_SYMBOL", "DOGEUSDT")  # heartbeat pair — DOGE has tiny minimum notional vs BTC
SYMBOLS  = [                                      # All pairs to scan each tick — matches bot-1-quant
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BNBUSDT", "XRPUSDT", "AVAXUSDT",
    "DOGEUSDT", "UNIUSDT", "ADAUSDT", "DOTUSDT",
]
INTERVAL = "1h"                                   # Candle size: 1m/5m/15m/1h/4h/1d
LOOKBACK = "730 day ago UTC"

# 0.50 — ETHUSDT hold-out showed 77.3% precision at this level.
BUY_THRESHOLD   = float(os.environ.get("BUY_THRESHOLD",   "0.50"))
SELL_THRESHOLD  = float(os.environ.get("SELL_THRESHOLD",  "0.45"))  # exit long when prob drops below here
SHORT_THRESHOLD = float(os.environ.get("SHORT_THRESHOLD", "0.55"))  # raised from 0.50 — 90% precision at 0.55+ vs 50% at 0.50
COVER_THRESHOLD = float(os.environ.get("COVER_THRESHOLD", "0.45"))  # exit short when prob drops below here

# Minimum walk-forward precision to allow new LONG entries (skip weak models)
MIN_LONG_WF_PRECISION = float(os.environ.get("MIN_LONG_WF_PRECISION", "0.42"))
ENABLE_SHORTING = os.environ.get("ENABLE_SHORTING", "true").lower() == "true"
ENABLE_LONGING  = os.environ.get("ENABLE_LONGING",  "true").lower() == "true"

# ─── Fear & Greed filter ──────────────────────────────────────────────────────
# Block shorts at Extreme Fear (squeeze risk) and Extreme Greed (strong uptrend).
# Allow shorts in 25–75: trend is in motion, ML confirms direction.
# Block longs at Extreme Greed (overbought) and Extreme Fear already falling.
FNG_SHORT_MIN = int(os.environ.get("FNG_SHORT_MIN", "25"))   # below = block short
FNG_SHORT_MAX = int(os.environ.get("FNG_SHORT_MAX", "75"))   # above = block short
FNG_LONG_MIN  = int(os.environ.get("FNG_LONG_MIN",  "0"))    # below = block long
FNG_LONG_MAX  = int(os.environ.get("FNG_LONG_MAX",  "45"))   # above = block long
ENABLE_FNG    = os.environ.get("ENABLE_FNG", "false").lower() == "true"

# ─── Paper trading ─────────────────────────────────────────────
PAPER_STARTING_BALANCE = 10_000   # Simulated USDT
RISK_PER_TRADE         = float(os.environ.get("RISK_PER_TRADE", "0.20"))  # Risk % of free balance per trade
MAX_POSITION_PCT       = 0.90     # Cap position at 90% of balance
STOP_LOSS_PCT          = 0.025    # Hard stop: 2.5% below/above entry (safety floor)
TAKE_PROFIT_ATR_MULT   = 3.5      # TP at entry ± 3.5 × ATR (was 2.5)
TRAIL_STOP_ATR_MULT    = 2.0      # Trail stop 2.0 × ATR from running high/low (was 1.5 — too tight)
# Minimum R:R enforced at trade entry. When hard stop (STOP_LOSS_PCT) is the
# effective SL, TP is scaled to stop_distance × MIN_RR_RATIO rather than
# the raw ATR-based TP — fixes the sub-1:1 R:R on low-ATR pairs (XRP, ADA).
# At 32% WR breakeven requires 2.125:1; 2.25 gives ~6% safety margin.
MIN_RR_RATIO           = float(os.environ.get("MIN_RR_RATIO", "2.25"))
TRADE_SIZE_PCT         = 0.95     # Kept for backward compat (not used in v3)
LOG_FILE               = "trades_log.csv"

# ─── Regime filter ─────────────────────────────────────────────
TREND_FILTER            = True    # Require price > EMA200
ADX_TREND_THRESHOLD     = 20      # Min ADX to consider market trending
EMA_ALIGN_THRESHOLD     = 1       # Min EMA alignment score (0–3) to allow BUY (relaxed from 2)
REQUIRE_HTF_TREND       = False   # Disabled — was blocking all entries in sideways market
REGIME_BYPASS_THRESHOLD = float(os.environ.get("REGIME_BYPASS_THRESHOLD", "0.75"))  # skip regime if ML prob >= this

# ─── Model ─────────────────────────────────────────────────────
MODEL_FILE     = "model.pkl"  # Used when SYMBOLS has a single entry
RETRAIN_DAYS   = 7            # Retrain every N days

# ─── Live trading ──────────────────────────────────────────────
# Set LIVE_TRADING=true + FUTURES_EXCHANGE=binanceusdm + BINANCE_API_KEY/SECRET env vars.
# Binance/Bybit are IP-blocked (451/403) on Railway — this bot runs from a home PC/VPS
# with a non-datacenter IP instead. OKX was tried but is account/KYC-blocked for this user.
LIVE_TRADING          = os.environ.get("LIVE_TRADING",          "false").lower() == "true"
LEVERAGE              = int(os.environ.get("LEVERAGE",              "3"))
MAX_LEVERAGE          = int(os.environ.get("MAX_LEVERAGE",          "10"))   # cap dynamic leverage (was 40 — too high for small accounts)
MAX_CONCURRENT_TRADES = int(os.environ.get("MAX_CONCURRENT_TRADES", "2"))
FUTURES_EXCHANGE      = os.environ.get("FUTURES_EXCHANGE",      "binanceusdm")
MIN_FREE_BALANCE      = float(os.environ.get("MIN_FREE_BALANCE",    "15"))   # skip new entries if free USDT < this

