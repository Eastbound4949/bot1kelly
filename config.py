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
SYMBOL   = "BTCUSDT"                              # Primary / fallback pair
SYMBOLS  = [                                      # All pairs to scan each tick
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BNBUSDT", "AVAXUSDT",
    "DOGEUSDT", "UNIUSDT", "ADAUSDT", "DOTUSDT",
]  # XRPUSDT removed: net -$36 across 4 trades (trail stop whipsaw)
INTERVAL = "1h"                                   # Candle size: 1m/5m/15m/1h/4h/1d
LOOKBACK = "730 day ago UTC"

# 0.50 — ETHUSDT hold-out showed 77.3% precision at this level.
BUY_THRESHOLD   = float(os.environ.get("BUY_THRESHOLD",   "0.50"))
SELL_THRESHOLD  = float(os.environ.get("SELL_THRESHOLD",  "0.38"))
SHORT_THRESHOLD = float(os.environ.get("SHORT_THRESHOLD", "0.50"))
COVER_THRESHOLD = float(os.environ.get("COVER_THRESHOLD", "0.38"))
ENABLE_SHORTING = os.environ.get("ENABLE_SHORTING", "true").lower() == "true"

# ─── Paper trading ─────────────────────────────────────────────
PAPER_STARTING_BALANCE = 10_000   # Simulated USDT
RISK_PER_TRADE         = 0.015    # Risk 1.5% of portfolio per trade (+50% vs 1%)
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
TREND_FILTER        = True    # Require price > EMA200
ADX_TREND_THRESHOLD = 20      # Min ADX to consider market trending
EMA_ALIGN_THRESHOLD = 2       # Min EMA alignment score (0–3) to allow BUY
REQUIRE_HTF_TREND   = False   # Disabled — was blocking all entries in sideways market

# ─── Model ─────────────────────────────────────────────────────
MODEL_FILE     = "model.pkl"  # Used when SYMBOLS has a single entry
RETRAIN_DAYS   = 7            # Retrain every N days

# ─── Live trading ──────────────────────────────────────────────
# Set LIVE_TRADING=true + FUTURES_EXCHANGE=okx + OKX_API_KEY/SECRET/PASSPHRASE env vars.
# OKX has no geo-block on Railway. Binance(451) + Bybit(CloudFront 403) both block Railway.
LIVE_TRADING          = os.environ.get("LIVE_TRADING",          "false").lower() == "true"
MAX_CONCURRENT_TRADES = int(os.environ.get("MAX_CONCURRENT_TRADES", "2"))
LEVERAGE              = int(os.environ.get("LEVERAGE",              "3"))
FUTURES_EXCHANGE      = os.environ.get("FUTURES_EXCHANGE",      "okx")

# ─── Reverse Kelly / risk-tapering bands ──────────────────────
# Band 1 $100–$200:  20% risk, aggressive — die cheap if it fails
# Band 2 $200–$400:  10% risk — still pushing
# Band 3 $400–$700:  5% risk — protect the progress
# Band 4 $700–$1000: 2.5% risk — conservative close-out
KELLY_BANDS = [
    {"band": 1, "min": 0,   "max": 200,  "risk_pct": 0.20,  "label": "aggressive"},
    {"band": 2, "min": 200, "max": 400,  "risk_pct": 0.10,  "label": "pushing"},
    {"band": 3, "min": 400, "max": 700,  "risk_pct": 0.05,  "label": "protect"},
    {"band": 4, "min": 700, "max": 1000, "risk_pct": 0.025, "label": "conservative"},
]
