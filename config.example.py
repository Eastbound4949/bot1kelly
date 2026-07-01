"""
config.example.py — Copy this to config.py and fill in your keys.
On Railway/server: set these as environment variables (recommended — never commit real keys).

cp config.example.py config.py
# then edit config.py with your real values
"""

import os

# ─── Binance API (READ-ONLY permissions sufficient for paper trading) ──────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY",    "YOUR_BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")

# ─── Telegram (@BotFather for token, @userinfobot for chat ID) ─────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

# ─── Trading settings ──────────────────────────────────────────────────────────
# kucoin/gateio/kraken work from any IP including Railway US servers.
# binance/bybit are blocked on US-based servers.
EXCHANGE = os.environ.get("EXCHANGE", "kucoin")
SYMBOL   = os.environ.get("SYMBOL",   "BTCUSDT")
SYMBOLS  = os.environ.get("SYMBOLS",  "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,DOGEUSDT,UNIUSDT,ADAUSDT,DOTUSDT").split(",")  # XRPUSDT removed: net -$36 in 4 trades
INTERVAL = os.environ.get("INTERVAL", "1h")
LOOKBACK = os.environ.get("LOOKBACK", "730 day ago UTC")

BUY_THRESHOLD    = float(os.environ.get("BUY_THRESHOLD",    "0.50"))
SELL_THRESHOLD   = float(os.environ.get("SELL_THRESHOLD",   "0.38"))
SHORT_THRESHOLD  = float(os.environ.get("SHORT_THRESHOLD",  "0.50"))
COVER_THRESHOLD  = float(os.environ.get("COVER_THRESHOLD",  "0.38"))
ENABLE_SHORTING  = os.environ.get("ENABLE_SHORTING", "false").lower() == "true"

# ─── Risk management ───────────────────────────────────────────────────────────
PAPER_STARTING_BALANCE = float(os.environ.get("PAPER_STARTING_BALANCE", "10000"))
RISK_PER_TRADE         = float(os.environ.get("RISK_PER_TRADE",         "0.015"))  # 1.5% risk (+50% vs 1%)
MAX_POSITION_PCT       = float(os.environ.get("MAX_POSITION_PCT",       "0.90"))
STOP_LOSS_PCT          = float(os.environ.get("STOP_LOSS_PCT",          "0.025"))
TAKE_PROFIT_ATR_MULT   = float(os.environ.get("TAKE_PROFIT_ATR_MULT",   "3.5"))   # was 2.5
TRAIL_STOP_ATR_MULT    = float(os.environ.get("TRAIL_STOP_ATR_MULT",    "2.0"))   # was 1.5 — too tight
# Minimum R:R at entry. TP is scaled to max(ATR-based, stop_distance × MIN_RR_RATIO).
MIN_RR_RATIO           = float(os.environ.get("MIN_RR_RATIO",           "2.25"))
TRADE_SIZE_PCT         = float(os.environ.get("TRADE_SIZE_PCT",         "0.95"))
LOG_FILE               = os.environ.get("LOG_FILE", "trades_log.csv")

# ─── Regime filter ─────────────────────────────────────────────────────────────
TREND_FILTER        = os.environ.get("TREND_FILTER",       "true").lower() == "true"
ADX_TREND_THRESHOLD = int(os.environ.get("ADX_TREND_THRESHOLD", "20"))
EMA_ALIGN_THRESHOLD = int(os.environ.get("EMA_ALIGN_THRESHOLD", "1"))
REQUIRE_HTF_TREND   = os.environ.get("REQUIRE_HTF_TREND",  "false").lower() == "true"
REGIME_BYPASS_THRESHOLD = float(os.environ.get("REGIME_BYPASS_THRESHOLD", "0.75"))  # skip regime if ML prob >= this

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_FILE   = "model.pkl"
RETRAIN_DAYS = int(os.environ.get("RETRAIN_DAYS", "7"))
