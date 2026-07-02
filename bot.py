"""
bot.py v3 — Regime-aware, multi-pair, adaptive trading bot.

Upgrades over v2:
  - ATR-based take-profit (2.5× ATR above entry)
  - Trailing stop (1.5× ATR below running high, hard floor at -2.5%)
  - ATR-based position sizing: risk exactly 1% of portfolio per trade
  - Regime gate: only trade when ADX trending + EMA stack aligned + HTF uptrend
  - Multi-pair scanner: picks highest-confidence signal across all SYMBOLS
  - Per-symbol model files (model_BTCUSDT_1h.pkl, etc.)
"""

import csv
import json
import os
import pickle
import time
from datetime import datetime, timezone

_BOT_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

import ccxt
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

from apscheduler.schedulers.blocking import BlockingScheduler
import config
from model_trainer import (
    add_features,
    fetch_binance_data,
    fetch_htf_data,
    model_file_for,
    short_model_file_for,
    train_and_save,
    train_and_save_short,
)


# ─── Telegram ────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"[telegram] Failed: {e}")


# ─── Fear & Greed index ──────────────────────────────────────────────────────
_fng_cache: dict = {"value": None, "ts": 0}

def fetch_fear_greed() -> int:
    """Return current crypto Fear & Greed index (0–100). Cached 1h."""
    now = datetime.now(timezone.utc).timestamp()
    if _fng_cache["value"] is not None and now - _fng_cache["ts"] < 3600:
        return _fng_cache["value"]
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        val = int(r.json()["data"][0]["value"])
        _fng_cache["value"] = val
        _fng_cache["ts"] = now
        return val
    except Exception as e:
        print(f"[fng] fetch failed: {e} — defaulting to 50 (neutral)")
        return _fng_cache["value"] if _fng_cache["value"] is not None else 50


def _fng_allows_short(fng: int) -> bool:
    """Block shorts only at extremes where trend reversal risk is high."""
    return config.FNG_SHORT_MIN <= fng <= config.FNG_SHORT_MAX


def _fng_allows_long(fng: int) -> bool:
    """Block longs only at extremes where trend reversal risk is high."""
    return config.FNG_LONG_MIN <= fng <= config.FNG_LONG_MAX


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_latest_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Returns (base_tf_df, htf_df) using ccxt. Exchange set by EXCHANGE env var."""
    df     = fetch_binance_data(symbol, config.INTERVAL, config.LOOKBACK)
    htf_df = fetch_htf_data(symbol, config.INTERVAL, config.LOOKBACK)
    return df, htf_df


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(symbol: str) -> tuple:
    """Load (model, feature_cols) for symbol. Trains if stale or missing."""
    mf = model_file_for(symbol)

    if not os.path.exists(mf):
        print(f"[bot] No model for {symbol} — training now...")
        train_and_save(symbol, mf)

    with open(mf, "rb") as f:
        payload = pickle.load(f)

    trained_dt = datetime.fromisoformat(payload["trained_at"])
    if trained_dt.tzinfo is None:
        trained_dt = trained_dt.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - trained_dt).days
    if age_days >= config.RETRAIN_DAYS:
        print(f"[bot] {symbol} model is {age_days}d old — retraining...")
        train_and_save(symbol, mf)
        with open(mf, "rb") as f:
            payload = pickle.load(f)

    return payload["model"], payload["feature_cols"]


def load_short_model(symbol: str) -> tuple | None:
    """Load (short_model, feature_cols). Returns None if ENABLE_SHORTING is false."""
    if not config.ENABLE_SHORTING:
        return None
    mf = short_model_file_for(symbol)
    if not os.path.exists(mf):
        print(f"[bot] No short model for {symbol} — training now...")
        train_and_save_short(symbol, mf)
    with open(mf, "rb") as f:
        payload = pickle.load(f)
    trained_dt = datetime.fromisoformat(payload["trained_at"])
    if trained_dt.tzinfo is None:
        trained_dt = trained_dt.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - trained_dt).days
    if age_days >= config.RETRAIN_DAYS:
        print(f"[bot] {symbol} short model is {age_days}d old — retraining...")
        train_and_save_short(symbol, mf)
        with open(mf, "rb") as f:
            payload = pickle.load(f)
    return payload["model"], payload["feature_cols"]


# ─── Regime filter ────────────────────────────────────────────────────────────

def _regime_ok(df: pd.DataFrame, prob: float = 0.0) -> bool:
    """Return True only when market conditions justify a long entry."""
    # High-confidence override: skip regime check if ML prob >= 75%
    if prob >= config.REGIME_BYPASS_THRESHOLD:
        return True

    row = df.iloc[-1]

    def get(col, default):
        return row[col] if col in df.columns else default

    above_ema200   = bool(get("above_ema200", 1))
    # Relaxed: ema_alignment >= 1 (was 2) — catches trending pairs with mixed EMAs
    adx_trending   = bool(get("regime_trending", 0)) or get("ema_alignment", 0) >= 1
    no_extreme_vol = not bool(get("regime_high_vol", 0))
    htf_trend      = not config.REQUIRE_HTF_TREND or bool(get("htf_trend_up", 1))

    return (
        (not config.TREND_FILTER or above_ema200) and
        adx_trending and
        no_extreme_vol and
        htf_trend
    )


def _bearish_regime_ok(df: pd.DataFrame, prob: float = 0.0) -> bool:
    """Return True only when market conditions justify a short entry."""
    # High-confidence override: skip regime check if ML prob >= 75%
    if prob >= config.REGIME_BYPASS_THRESHOLD:
        return True

    row = df.iloc[-1]

    def get(col, default):
        return row[col] if col in df.columns else default

    # Removed below_ema200 requirement — kills all shorts in bull markets
    ema_bearish    = int(get("ema_alignment", 3)) <= 2  # relaxed: 2 of 3 EMAs bearish (was 1)
    adx_trending   = bool(get("regime_trending", 0))
    no_extreme_vol = not bool(get("regime_high_vol", 0))
    htf_trend_down = not bool(get("htf_trend_up", 1))

    return (
        ema_bearish and
        adx_trending and
        no_extreme_vol and
        (not config.REQUIRE_HTF_TREND or htf_trend_down)
    )


def _futures_symbol(symbol: str) -> str:
    """'BTCUSDT' → 'BTC/USDT:USDT' (ccxt binanceusdm format)."""
    base = symbol.replace("USDT", "")
    return f"{base}/USDT:USDT"


# ─── Paper trade state ────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulates trading without real money.

    Position sizing: risk RISK_PER_TRADE% of portfolio per trade, sized by ATR.
    Exit logic:      take-profit at entry + TAKE_PROFIT_ATR_MULT × ATR
                     trailing stop 1.5 × ATR below running high
                     hard stop floor at entry × (1 - STOP_LOSS_PCT)
                     ML SELL signal also exits
    """

    _log_file = os.path.join(_BOT_DIR, config.LOG_FILE)

    def __init__(self):
        self.balance         = config.PAPER_STARTING_BALANCE
        self.position        = 0.0    # long: coin units held
        self.entry_price     = 0.0
        self.take_profit_price = 0.0
        self.trail_stop_price  = 0.0
        self.position_symbol = ""
        # Short position state (paper short: receive proceeds at entry, pay at cover)
        self.short_position    = 0.0  # units shorted
        self.short_entry_price = 0.0
        self.short_take_profit = 0.0  # TP is below entry for shorts
        self.short_trail_stop  = 0.0  # trailing stop is above entry, moves down
        self.short_symbol      = ""
        self.trades            = 0
        self.wins              = 0
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self._log_file):
            return
        try:
            df = pd.read_csv(self._log_file)
            if len(df):
                last = df.iloc[-1]
                self.balance           = float(last["balance_usdt"])
                self.position          = float(last["position_units"])
                self.entry_price       = float(last.get("entry_price", 0))
                self.take_profit_price = float(last.get("take_profit_price", 0))
                self.trail_stop_price  = float(last.get("trail_stop_price", 0))
                self.position_symbol   = str(last.get("position_symbol", ""))
                self.short_position    = float(last.get("short_units", 0))
                self.short_entry_price = float(last.get("short_entry_price", 0))
                self.short_take_profit = float(last.get("short_take_profit", 0))
                self.short_trail_stop  = float(last.get("short_trail_stop", 0))
                self.short_symbol      = str(last.get("short_symbol", ""))
                self.trades            = int(last.get("total_trades", 0))
                self.wins              = int(last.get("total_wins", 0))
                print(
                    f"[bot] Restored: ${self.balance:,.2f} USDT | "
                    f"{self.position:.6f} {self.position_symbol or 'none'}"
                )
        except Exception as e:
            print(f"[bot] Could not restore state: {e}")

    # ── Exit helpers ──────────────────────────────────────────────────────────

    def _close_position(self, price: float, reason: str) -> str:
        proceeds = self.position * price
        pnl      = proceeds - (self.position * self.entry_price)
        pnl_pct  = (price / self.entry_price - 1) * 100
        if pnl > 0:
            self.wins += 1
        self.trades           += 1
        self.balance          += proceeds
        result = (
            f"{reason} {self.position:.6f} {self.position_symbol} "
            f"@ ${price:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        self.position          = 0.0
        self.entry_price       = 0.0
        self.take_profit_price = 0.0
        self.trail_stop_price  = 0.0
        self.position_symbol   = ""
        return result

    def _close_short(self, price: float, reason: str) -> str:
        # At entry: balance += short_units × entry_price (received proceeds)
        # At cover: balance -= short_units × cover_price (pay to buy back)
        cost_to_cover = self.short_position * price
        pnl           = self.short_position * (self.short_entry_price - price)
        pnl_pct       = (self.short_entry_price / price - 1) * 100
        if pnl > 0:
            self.wins  += 1
        self.trades    += 1
        self.balance   -= cost_to_cover
        result = (
            f"{reason} {self.short_position:.6f} {self.short_symbol} "
            f"@ ${price:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        self.short_position    = 0.0
        self.short_entry_price = 0.0
        self.short_take_profit = 0.0
        self.short_trail_stop  = 0.0
        self.short_symbol      = ""
        return result

    # ── Main execute ──────────────────────────────────────────────────────────

    def execute(self, signal: str, price: float, atr: float, symbol: str) -> str:
        atr = max(atr, price * 0.001)  # floor: min 0.1% of price

        # ── Short position exit ───────────────────────────────────────────────
        if self.short_position > 0 and self.short_symbol == symbol:
            # Trailing stop for shorts moves DOWN as price falls, never up.
            # Gate: only tighten to breakeven-or-better — otherwise it parks
            # between entry and the original stop, and a normal retrace closes
            # the trade at a small loss before TP is ever reached.
            new_trail = price + config.TRAIL_STOP_ATR_MULT * atr
            if new_trail < self.short_trail_stop and new_trail <= self.short_entry_price:
                self.short_trail_stop = new_trail

            hard_stop      = self.short_entry_price * (1 + config.STOP_LOSS_PCT)
            effective_stop = min(hard_stop, self.short_trail_stop)
            stop_type      = "SHORT-TRAIL" if self.short_trail_stop < hard_stop else "SHORT-STOP"

            if price >= effective_stop:
                return self._close_short(price, stop_type)
            if price <= self.short_take_profit:
                return self._close_short(price, "SHORT-TP")
            if signal == "COVER":
                return self._close_short(price, "ML-COVER")
            return "HOLD — monitoring short"

        # ── Long position exit ────────────────────────────────────────────────
        if self.position > 0 and self.position_symbol == symbol:
            # Update trailing stop: only moves up, never down.
            # Gate: only tighten to breakeven-or-better — same reasoning as the
            # short side above: a stop parked between entry and the original
            # floor turns a near-TP retrace into a needless loss.
            new_trail = price - config.TRAIL_STOP_ATR_MULT * atr
            if new_trail > self.trail_stop_price and new_trail >= self.entry_price:
                self.trail_stop_price = new_trail

            hard_stop      = self.entry_price * (1 - config.STOP_LOSS_PCT)
            effective_stop = max(hard_stop, self.trail_stop_price)
            stop_type      = "TRAIL-STOP" if self.trail_stop_price > hard_stop else "STOP-LOSS"

            if price <= effective_stop:
                return self._close_position(price, stop_type)
            if price >= self.take_profit_price:
                return self._close_position(price, "TAKE-PROFIT")
            if signal == "SELL":
                return self._close_position(price, "ML-SELL")
            return "HOLD — monitoring position"

        # ── Short entry ───────────────────────────────────────────────────────
        if signal == "SHORT" and self.short_position == 0 and self.position == 0 and self.balance > 10:
            portfolio_val  = self.portfolio_value(price)
            risk_amount    = portfolio_val * config.RISK_PER_TRADE
            # When hard stop (STOP_LOSS_PCT) is wider than ATR-based stop, TP must
            # scale with it — otherwise R:R collapses below 1:1 on low-ATR pairs.
            stop_distance  = max(config.TRAIL_STOP_ATR_MULT * atr, price * config.STOP_LOSS_PCT)
            tp_distance    = max(config.TAKE_PROFIT_ATR_MULT * atr, stop_distance * config.MIN_RR_RATIO)
            rr             = tp_distance / stop_distance
            if rr < config.MIN_RR_RATIO:
                return f"HOLD — SHORT R:R {rr:.2f}:1 below min {config.MIN_RR_RATIO}"
            position_units = risk_amount / stop_distance
            notional       = position_units * price
            notional       = min(notional, self.balance * config.MAX_POSITION_PCT)
            position_units = notional / price

            self.short_position    = position_units
            self.balance          += notional          # receive proceeds from short sale
            self.short_entry_price = price
            self.short_take_profit = price - tp_distance
            self.short_trail_stop  = price + config.TRAIL_STOP_ATR_MULT * atr
            self.short_symbol      = symbol
            self.trades           += 1

            return (
                f"SHORTED {position_units:.6f} {symbol} @ ${price:,.2f} | "
                f"TP: ${self.short_take_profit:,.2f} | "
                f"SL: ${self.short_trail_stop:,.2f} | "
                f"R:R: {rr:.2f} | "
                f"Risk: ${risk_amount:.2f}"
            )

        # ── Long entry ────────────────────────────────────────────────────────
        if signal == "BUY" and self.position == 0 and self.balance > 10:
            portfolio_val  = self.portfolio_value(price)
            risk_amount    = portfolio_val * config.RISK_PER_TRADE
            # When hard stop (STOP_LOSS_PCT) is wider than ATR-based stop, TP must
            # scale with it — otherwise R:R collapses below 1:1 on low-ATR pairs.
            stop_distance  = max(config.TRAIL_STOP_ATR_MULT * atr, price * config.STOP_LOSS_PCT)
            tp_distance    = max(config.TAKE_PROFIT_ATR_MULT * atr, stop_distance * config.MIN_RR_RATIO)
            rr             = tp_distance / stop_distance
            if rr < config.MIN_RR_RATIO:
                return f"HOLD — BUY R:R {rr:.2f}:1 below min {config.MIN_RR_RATIO}"
            position_units = risk_amount / stop_distance
            notional       = position_units * price
            notional       = min(notional, self.balance * config.MAX_POSITION_PCT)
            position_units = notional / price

            self.position           = position_units
            self.balance           -= notional
            self.entry_price        = price
            self.take_profit_price  = price + tp_distance
            self.trail_stop_price   = price - config.TRAIL_STOP_ATR_MULT * atr
            self.position_symbol    = symbol
            self.trades            += 1

            return (
                f"BOUGHT {position_units:.6f} {symbol} @ ${price:,.2f} | "
                f"TP: ${self.take_profit_price:,.2f} | "
                f"SL: ${self.trail_stop_price:,.2f} | "
                f"R:R: {rr:.2f} | "
                f"Risk: ${risk_amount:.2f}"
            )

        return "HOLD — no action"

    def portfolio_value(self, price: float) -> float:
        # balance already contains short sale proceeds; subtract liability to cover
        return self.balance + self.position * price - self.short_position * price

    def log_trade(self, timestamp, symbol, signal, price, buy_prob, action):
        portfolio_val = self.portfolio_value(price)
        win_rate      = (self.wins / max(self.trades, 1)) * 100
        row = {
            "timestamp":         timestamp,
            "symbol":            symbol,
            "price":             round(price, 4),
            "buy_prob":          round(buy_prob, 4),
            "signal":            signal,
            "action":            action,
            "balance_usdt":      round(self.balance, 2),
            "position_units":    round(self.position, 8),
            "position_symbol":   self.position_symbol,
            "entry_price":       round(self.entry_price, 4),
            "take_profit_price": round(self.take_profit_price, 4),
            "trail_stop_price":  round(self.trail_stop_price, 4),
            "short_units":       round(self.short_position, 8),
            "short_symbol":      self.short_symbol,
            "short_entry_price": round(self.short_entry_price, 4),
            "short_take_profit": round(self.short_take_profit, 4),
            "short_trail_stop":  round(self.short_trail_stop, 4),
            "portfolio_usd":     round(portfolio_val, 2),
            "total_trades":      self.trades,
            "total_wins":        self.wins,
            "win_rate_pct":      round(win_rate, 1),
        }
        file_exists = os.path.exists(self._log_file)
        with open(self._log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


# ─── Live trader (Binance USDT-M Futures) ────────────────────────────────────

class LiveTrader:
    """
    Executes real orders on Binance USDT-M Futures via ccxt.
    Position sizing: Kelly-band risk% of free USDT balance.
    SL + TP placed as exchange stop-market orders immediately after entry.
    State is authoritative from exchange — no in-memory tracking.
    """

    _log_file = os.path.join(_BOT_DIR, "live_trades_log.csv")

    def __init__(self):
        exch = config.FUTURES_EXCHANGE
        self._is_bybit = exch == "bybit"
        self._is_okx   = exch == "okx"

        if self._is_okx:
            api_key, api_secret = config.OKX_API_KEY, config.OKX_API_SECRET
        elif self._is_bybit:
            api_key, api_secret = config.BYBIT_API_KEY, config.BYBIT_API_SECRET
        else:
            api_key, api_secret = config.BINANCE_API_KEY, config.BINANCE_API_SECRET

        exchange_params = {
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {"defaultType": "swap" if self._is_okx else "future"},
        }
        if self._is_okx:
            exchange_params["password"] = config.OKX_PASSPHRASE

        exchange_class = getattr(ccxt, exch)
        self._exchange = exchange_class(exchange_params)
        self._exchange.has["fetchCurrencies"] = False  # skip geo-blocked private endpoint
        self._state_file = os.path.join(_BOT_DIR, "live_trail_state.json")
        self._live_state: dict = self._load_live_state()
        self._entries, self._wins, self._losses = self._load_live_stats()
        _delay = 30
        for _attempt in range(1, 9999):
            try:
                self._exchange.load_markets()
                break
            except Exception as e:
                print(f"[live] load_markets attempt {_attempt} failed: {e}. Retry in {_delay}s...")
                time.sleep(_delay)
                _delay = min(int(_delay * 1.5), 300)

    # ── Exchange helpers ──────────────────────────────────────────────────────

    def _free_balance(self) -> float:
        b = self._exchange.fetch_balance()
        return float(b.get("USDT", {}).get("free", 0))

    def _open_positions(self) -> list:
        return [p for p in self._exchange.fetch_positions() if float(p.get("contracts", 0)) != 0]

    def _set_leverage(self, fsym: str, lev: int):
        if not self._is_bybit and not self._is_okx:
            # Binance defaults new symbols to ISOLATED, which gets margin-called
            # on adverse moves; CROSSED shares the whole account balance as margin.
            try:
                self._exchange.set_margin_mode("cross", fsym)
            except Exception as e:
                print(f"[live] set_margin_mode {fsym} cross: {e}")
        try:
            if self._is_bybit:
                params = {"buyLeverage": str(lev), "sellLeverage": str(lev)}
            elif self._is_okx:
                params = {"mgnMode": "isolated"}
            else:
                params = {}
            self._exchange.set_leverage(lev, fsym, params=params)
        except Exception as e:
            print(f"[live] set_leverage {fsym} {lev}x: {e}")

    def _sl_params(self, trigger: float) -> dict:
        if self._is_bybit:
            return {"triggerPrice": trigger, "reduceOnly": True}
        if self._is_okx:
            return {"stopPrice": trigger, "reduceOnly": True}
        # Binance USDM rejects reduceOnly combined with closePosition (-1106)
        return {"stopPrice": trigger, "closePosition": True}

    def _tp_params(self, trigger: float) -> dict:
        if self._is_bybit:
            return {"triggerPrice": trigger, "reduceOnly": True}
        if self._is_okx:
            return {"stopPrice": trigger, "reduceOnly": True}
        # Binance USDM rejects reduceOnly combined with closePosition (-1106)
        return {"stopPrice": trigger, "closePosition": True}

    def _cancel_open_orders(self, fsym: str) -> bool:
        """Cancel every open order for fsym by explicit ID and poll until Binance
        confirms none remain. cancel_all_orders + a fixed sleep is not reliable —
        it can return success while closePosition STOP_MARKET/TAKE_PROFIT_MARKET
        orders are still being processed, causing -4130 on the immediate re-place."""
        try:
            orders = self._exchange.fetch_open_orders(fsym)
        except Exception as e:
            print(f"[live] fetch_open_orders {fsym}: {e}")
            orders = []

        for o in orders:
            try:
                self._exchange.cancel_order(o["id"], fsym)
            except Exception as e:
                print(f"[live] cancel_order {fsym} {o.get('id')}: {e}")

        for _ in range(6):
            time.sleep(0.5)
            try:
                remaining = self._exchange.fetch_open_orders(fsym)
            except Exception:
                continue
            if not remaining:
                return True
            for o in remaining:
                try:
                    self._exchange.cancel_order(o["id"], fsym)
                except Exception as e:
                    print(f"[live] cancel_order retry {fsym} {o.get('id')}: {e}")

        print(f"[live] cancel_open_orders {fsym}: orders still open after retries")
        return False

    def _place_close_order(self, fsym: str, order_type: str, close_side: str,
                            units: float, trigger: float, tries: int = 3, delay: float = 0.5):
        """Place a closePosition SL/TP order, retrying on Binance's -4130 (stale
        closePosition order still settling server-side after cancel)."""
        params = self._sl_params(trigger) if order_type == "stop_market" else self._tp_params(trigger)
        last_err = None
        for attempt in range(tries):
            try:
                return self._exchange.create_order(fsym, order_type, close_side, units, params=params), None
            except Exception as e:
                last_err = e
                if attempt < tries - 1:
                    time.sleep(delay)
        return None, last_err

    # ── Trail state persistence ───────────────────────────────────────────────

    def _load_live_state(self) -> dict:
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_live_stats(self) -> tuple:
        """Read trades/wins/losses from CSV log. Returns (entries, wins, losses)."""
        try:
            if not os.path.exists(self._log_file):
                return 0, 0, 0
            df = pd.read_csv(self._log_file)
            if df.empty:
                return 0, 0, 0
            entries = int(df["action"].str.contains(r"^LIVE (BUY|SELL|SHORT)", na=False, regex=True).sum())
            closes = df[df["action"].str.contains("ML-CLOSE", na=False)]
            if "pnl_usd" in df.columns:
                wins   = int((closes["pnl_usd"] > 0).sum())
                losses = int((closes["pnl_usd"] <= 0).sum() - (closes["pnl_usd"] == 0).sum())
            else:
                wins, losses = 0, 0
            return entries, wins, losses
        except Exception:
            return 0, 0, 0

    def _fmt_stats(self) -> str:
        """Return compact stats string for Telegram."""
        wr = (self._wins / max(self._wins + self._losses, 1)) * 100 if (self._wins + self._losses) else 0
        return f"T:{self._entries} W:{self._wins} L:{self._losses} WR:{wr:.0f}%"

    def _save_live_state(self, symbol: str, state: dict):
        self._live_state[symbol] = state
        with open(self._state_file, "w") as f:
            json.dump(self._live_state, f)

    def _clear_live_state(self, symbol: str):
        self._live_state.pop(symbol, None)
        with open(self._state_file, "w") as f:
            json.dump(self._live_state, f)

    def _update_trail_stop(self, symbol: str, price: float, atr: float) -> str | None:
        """Advance ATR trailing stop once price reaches breakeven-or-better.
        Cancel old SL, place new one. Re-place TP so it isn't lost."""
        state = self._live_state.get(symbol)
        if not state:
            return None

        fsym         = _futures_symbol(symbol)
        side         = state["side"]
        entry_price  = state["entry_price"]
        current_trail = state["trail_stop"]
        tp_price     = state["tp_price"]
        units        = state["units"]
        atr          = max(atr, price * 0.001)

        if side == "long":
            new_trail = price - config.TRAIL_STOP_ATR_MULT * atr
            if not (new_trail > current_trail and new_trail >= entry_price):
                return None
            close_side = "sell"
        else:
            new_trail = price + config.TRAIL_STOP_ATR_MULT * atr
            if not (new_trail < current_trail and new_trail <= entry_price):
                return None
            close_side = "buy"

        sl_prec = float(self._exchange.price_to_precision(fsym, new_trail))
        tp_prec = float(self._exchange.price_to_precision(fsym, tp_price))

        if not self._cancel_open_orders(fsym):
            print(f"[live] trail skip {symbol}: cancel failed, state unchanged")
            return None

        sl_order, sl_err = self._place_close_order(fsym, "stop_market", close_side, units, sl_prec)
        if sl_order is None:
            # New SL failed after retries and old one is already cancelled — position
            # is naked. Restore the PRIOR stop immediately rather than leave it unprotected
            # until the next bar (up to 1h away on this schedule).
            print(f"[live] trail SL order failed after retries: {sl_err}")
            fallback_prec = float(self._exchange.price_to_precision(fsym, current_trail))
            restore_order, restore_err = self._place_close_order(fsym, "stop_market", close_side, units, fallback_prec)
            if restore_order is None:
                print(f"[live] CRITICAL: {symbol} has NO STOP LOSS — restore also failed: {restore_err}")
                send_telegram(f"CRITICAL: {symbol} live position has NO STOP LOSS attached — manual check needed. {restore_err}")
            else:
                print(f"[live] trail SL restore OK: kept prior SL ${fallback_prec:,.2f}")
            return None

        tp_order, tp_err = self._place_close_order(fsym, "take_profit_market", close_side, units, tp_prec)
        if tp_order is None:
            print(f"[live] trail TP re-place failed after retries: {tp_err}")
            send_telegram(f"WARNING: {symbol} trail update lost TP order (SL ok @ ${sl_prec:,.2f}). {tp_err}")

        state["trail_stop"] = sl_prec
        self._save_live_state(symbol, state)
        return f"TRAIL → SL ${sl_prec:,.2f} (was ${current_trail:,.2f})"

    # ── Main execute ──────────────────────────────────────────────────────────

    def execute(self, signal: str, price: float, atr: float, symbol: str) -> str:
        atr = max(atr, price * 0.001)
        fsym = _futures_symbol(symbol)

        open_pos_map = {p["symbol"]: p for p in self._open_positions()}
        pos = open_pos_map.get(fsym)

        # ── Exit: existing position ───────────────────────────────────────────
        if pos is not None:
            side = pos["side"]
            contracts = abs(float(pos["contracts"]))
            upnl = float(pos.get("unrealizedPnl", 0))
            should_exit = (
                (side == "long" and signal == "SELL") or
                (side == "short" and signal == "COVER")
            )
            if should_exit:
                self._cancel_open_orders(fsym)
                close_side = "sell" if side == "long" else "buy"
                self._exchange.create_order(
                    fsym, "market", close_side, contracts,
                    params={"reduceOnly": True},
                )
                result = f"ML-CLOSE {side} {contracts:.6f} {symbol} @ ~${price:,.2f} | uPnL: ${upnl:+,.2f}"
                self._log(symbol, signal, price, result)
                return result
            return f"HOLD — monitoring {side} {contracts:.6f} {symbol} | uPnL: ${upnl:+,.2f}"

        # ── No position: check for SL/TP-closed scenario ──────────────────────
        if signal in ("SELL", "COVER", "HOLD"):
            return "HOLD — no open position (SL/TP may have closed it)"

        # ── New entry ─────────────────────────────────────────────────────────
        balance = self._free_balance()
        if balance < config.MIN_FREE_BALANCE:
            return f"HOLD — free balance ${balance:.2f} below minimum ${config.MIN_FREE_BALANCE}"

        risk_amount = balance * config.RISK_PER_TRADE

        # Fetch actual Binance price before sizing — KuCoin signal price may be stale/diverged
        try:
            _ticker = self._exchange.fetch_ticker(fsym)
            live_price = float(_ticker.get("last") or _ticker.get("bid") or 0) or price
        except Exception:
            live_price = price
            print(f"[live] WARNING: Binance price fetch failed for {symbol}, sizing may be off")

        # Scale KuCoin ATR to Binance price level, then compute stop/tp
        atr_live      = atr * live_price / max(price, 1e-9)
        stop_distance = max(config.TRAIL_STOP_ATR_MULT * atr_live, live_price * config.STOP_LOSS_PCT)
        tp_distance   = max(config.TAKE_PROFIT_ATR_MULT * atr_live, stop_distance * config.MIN_RR_RATIO)
        rr = tp_distance / stop_distance
        if rr < config.MIN_RR_RATIO:
            return f"HOLD — R:R {rr:.2f}:1 below min {config.MIN_RR_RATIO}"

        # Dynamic leverage: max where stop fires before liquidation (0.80 safety buffer for MMR+slippage)
        stop_pct = stop_distance / live_price
        dynamic_lev = max(1, min(int(0.80 / stop_pct), config.MAX_LEVERAGE))

        units = risk_amount / stop_distance
        notional = units * live_price
        max_notional = balance * config.MAX_POSITION_PCT * dynamic_lev
        if notional > max_notional:
            units = max_notional / live_price

        units = float(self._exchange.amount_to_precision(fsym, units))
        if units <= 0:
            return "HOLD — position too small for exchange minimums"

        self._set_leverage(fsym, dynamic_lev)
        entry_side = "buy" if signal == "BUY" else "sell"
        order = self._exchange.create_order(fsym, "market", entry_side, units)

        # Use Binance fill price; fall back to pre-fetched live_price (never KuCoin signal price)
        entry_price = float(order.get("average") or order.get("price") or 0) or live_price

        if signal == "BUY":
            sl_price = entry_price - stop_distance
            tp_price = entry_price + tp_distance
            close_side = "sell"
        else:
            sl_price = entry_price + stop_distance
            tp_price = entry_price - tp_distance
            close_side = "buy"

        sl_str = float(self._exchange.price_to_precision(fsym, sl_price))
        tp_str = float(self._exchange.price_to_precision(fsym, tp_price))

        sl_order, sl_err = self._place_close_order(fsym, "stop_market", close_side, units, sl_str)
        if sl_order is None:
            # No stop-loss could be attached — do not run a leveraged position naked.
            # Emergency-close at market immediately.
            print(f"[live] SL order failed after retries: {sl_err} — emergency-closing position")
            try:
                self._exchange.create_order(fsym, "market", close_side, units, params={"reduceOnly": True})
                send_telegram(f"WARNING: {symbol} entry SL failed ({sl_err}) — position emergency-closed, no trade taken.")
            except Exception as e2:
                print(f"[live] CRITICAL: emergency close also failed: {e2}")
                send_telegram(f"CRITICAL: {symbol} has NO STOP LOSS and emergency close FAILED — manual check needed. {e2}")
            return f"ABORTED {signal} {symbol} @ ${entry_price:,.2f} — SL failed, position closed"

        tp_order, tp_err = self._place_close_order(fsym, "take_profit_market", close_side, units, tp_str)
        if tp_order is None:
            print(f"[live] TP order failed after retries: {tp_err}")
            send_telegram(f"WARNING: {symbol} entry has no TP order (SL ok @ ${sl_str:,.2f}). {tp_err}")

        result = (
            f"LIVE {signal} {units} {symbol} @ ${entry_price:,.2f} | "
            f"SL: ${sl_str:,.2f} | TP: ${tp_str:,.2f} | R:R: {rr:.2f} | "
            f"Lev: {dynamic_lev}x | Risk: {config.RISK_PER_TRADE:.0%} = ${risk_amount:.2f}"
        )
        self._save_live_state(symbol, {
            "side":             "long" if signal == "BUY" else "short",
            "entry_price":      entry_price,
            "entry_atr":        atr,
            "trail_stop":       float(sl_str),
            "tp_price":         float(tp_str),
            "units":            units,
            "risk_dollar":      risk_amount,
            "balance_at_entry": balance,
        })
        self._log(symbol, signal, entry_price, result)
        return result

    def run_heartbeat_trade(self) -> str:
        """Open + immediately close the smallest possible position. Tries configured SYMBOL first,
        then falls back through DOGEUSDT / XRPUSDT if insufficient margin (-2019)."""
        fallback_order = [config.SYMBOL, "DOGEUSDT", "XRPUSDT", "ADAUSDT"]
        tried = []
        for sym in dict.fromkeys(fallback_order):  # deduplicate, preserve order
            fsym = _futures_symbol(sym)
            tried.append(sym)
            try:
                market   = self._exchange.markets.get(fsym, {})
                min_amt  = market.get("limits", {}).get("amount", {}).get("min") or 0
                min_cost = market.get("limits", {}).get("cost", {}).get("min") or 0

                ticker = self._exchange.fetch_ticker(fsym)
                price  = float(ticker["last"])

                units = max(min_amt, (min_cost / price) if price else 0)
                units = float(self._exchange.amount_to_precision(fsym, units * 1.05))
                if units <= 0:
                    raise ValueError("could not determine minimum order size")

                self._set_leverage(fsym, 1)
                open_order  = self._exchange.create_order(fsym, "market", "buy", units)
                entry_price = float(open_order.get("average") or open_order.get("price") or price)

                close_order = self._exchange.create_order(fsym, "market", "sell", units, params={"reduceOnly": True})
                exit_price  = float(close_order.get("average") or close_order.get("price") or price)

                msg = (
                    f"HEARTBEAT OK — {sym} open+close {units} "
                    f"@ ~${entry_price:,.4f} -> ~${exit_price:,.4f} (fees only, flat)"
                )
                print(f"[live] {msg}")
                send_telegram(f"*Startup Heartbeat*\n{msg}")
                return msg
            except Exception as e:
                if "-2019" in str(e) or "Margin is insufficient" in str(e):
                    print(f"[live] Heartbeat {sym}: insufficient margin, trying next...")
                    continue
                msg = f"HEARTBEAT FAILED — {sym}: {e}"
                print(f"[live] {msg}")
                send_telegram(f"*Startup Heartbeat*\n{msg}")
                return msg

        msg = f"HEARTBEAT SKIPPED — all symbols ({', '.join(tried)}) have insufficient margin (balance too low)"
        print(f"[live] {msg}")
        send_telegram(f"*Startup Heartbeat*\n{msg}")
        return msg

    def portfolio_value(self, _price: float = 0.0) -> float:
        try:
            b = self._exchange.fetch_balance()
            # Binance USDT-M: USDT["total"] = totalMarginBalance (walletBalance + uPnL already).
            # Do NOT add unrealizedPnL separately — it would double-count uPnL.
            return float(b.get("USDT", {}).get("total", 0))
        except Exception as e:
            print(f"[live] portfolio_value: {e}")
            return 0.0

    def log_trade(self, timestamp, symbol, signal, price, prob, action, pnl_usd: float = 0.0):
        balance = self._free_balance()
        row = {
            "timestamp":     timestamp,
            "symbol":        symbol,
            "price":         round(price, 4),
            "ml_prob":       round(prob, 4),
            "signal":        signal,
            "action":        action,
            "pnl_usd":       round(pnl_usd, 4),
            "balance_usdt":  round(balance, 2),
            "portfolio_usd": round(self.portfolio_value(), 2),
            "risk_pct":      config.RISK_PER_TRADE,
        }
        file_exists = os.path.exists(self._log_file)
        with open(self._log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _log(self, symbol, signal, price, action, pnl_usd: float = 0.0):
        self.log_trade(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            symbol, signal, price, 0.0, action, pnl_usd=pnl_usd,
        )


# ─── Multi-pair scanner ───────────────────────────────────────────────────────

def _scan_symbols() -> list[tuple]:
    """
    Scan all configured symbols. Return list of
    (symbol, buy_prob, short_prob, price, atr, df) sorted by buy_prob descending.
    """
    results = []
    for sym in config.SYMBOLS:
        try:
            df, htf_df = fetch_latest_data(sym)
            df         = add_features(df, htf_df)
            model, fcols = load_model(sym)

            for c in fcols:
                if c not in df.columns:
                    df[c] = np.nan  # XGBoost treats NaN as missing — acceptable fallback

            buy_prob  = float(model.predict_proba(df[fcols].iloc[[-1]])[0][1])

            short_prob = 0.0
            if config.ENABLE_SHORTING:
                short_result = load_short_model(sym)
                if short_result:
                    sm, sfcols = short_result
                    for c in sfcols:
                        if c not in df.columns:
                            df[c] = np.nan
                    short_prob = float(sm.predict_proba(df[sfcols].iloc[[-1]])[0][1])

            price = float(df["close"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])
            results.append((sym, buy_prob, short_prob, price, atr, df))
        except Exception as e:
            print(f"[scanner] {sym}: {e}")
    return sorted(results, key=lambda x: x[1], reverse=True)


# ─── Main loop ────────────────────────────────────────────────────────────────

paper_trader: PaperTrader | LiveTrader = LiveTrader() if config.LIVE_TRADING else PaperTrader()


def _run_live_bot():
    """Live-mode tick: monitor open futures positions, then scan for entries."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'-'*55}")
    print(f"[live] Running at {now}")

    try:
        open_positions = paper_trader._open_positions()
        balance = paper_trader._free_balance()
        portfolio_val = paper_trader.portfolio_value()

        print(
            f"[live] Balance: ${balance:,.2f} | Portfolio: ${portfolio_val:,.2f} | "
            f"Risk: {config.RISK_PER_TRADE:.0%}/trade | Open positions: {len(open_positions)}"
        )

        # Detect SL/TP-closed positions: in state but not in open_positions anymore
        open_syms = {p["symbol"].split("/")[0] + "USDT" for p in open_positions}
        for sym in list(paper_trader._live_state.keys()):
            if sym not in open_syms:
                state = paper_trader._live_state.get(sym, {})
                side = state.get("side", "?")
                entry_p = state.get("entry_price", 0)
                sl_p    = state.get("trail_stop", 0)
                tp_p    = state.get("tp_price", 0)
                paper_trader._entries += 1 if entry_p else 0
                paper_trader._clear_live_state(sym)
                dir_emoji = "📈" if side == "long" else "📉"
                sl_str = f"SL≈${sl_p:,.4f}" if sl_p else "SL?"
                tp_str = f"TP≈${tp_p:,.4f}" if tp_p else "TP?"
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                close_msg = (
                    f"🔔 *Position Closed by Exchange*\n"
                    f"{dir_emoji} {sym} {side.upper()} — SL/TP hit\n"
                    f"Entry: ${entry_p:,.4f} | {sl_str} | {tp_str}\n"
                    f"Balance: ${balance:,.2f} | Portfolio: ${portfolio_val:,.2f}\n"
                    f"{paper_trader._fmt_stats()}"
                )
                send_telegram(close_msg)
                paper_trader.log_trade(now_str, sym, "SL/TP", entry_p, 0.0, f"CLOSED by exchange ({side})")

        # Phase 1: monitor open positions for ML-driven exits
        for pos in open_positions:
            # ccxt returns 'BTC/USDT:USDT' — convert back to 'BTCUSDT'
            fsym = pos["symbol"]
            base = fsym.split("/")[0]
            sym = base + "USDT"
            side = pos["side"]

            if sym not in config.SYMBOLS:
                continue

            try:
                df, htf_df = fetch_latest_data(sym)
                df = add_features(df, htf_df)
                price = float(df["close"].iloc[-1])
                atr   = float(df["atr"].iloc[-1])
                upnl  = float(pos.get("unrealizedPnl", 0))

                trail_msg = paper_trader._update_trail_stop(sym, price, atr)
                if trail_msg:
                    print(f"[live] {sym} {trail_msg}")
                    send_telegram(f"*Trail Updated*\n{sym} @ ${price:,.4f}\n{trail_msg}")

                if side == "long":
                    if not config.ENABLE_LONGING:
                        signal = "SELL"
                        prob = 0.0
                    else:
                        model, fcols = load_model(sym)
                        for c in fcols:
                            if c not in df.columns:
                                df[c] = np.nan
                        prob = float(model.predict_proba(df[fcols].iloc[[-1]])[0][1])
                        signal = "SELL" if prob <= config.SELL_THRESHOLD else "HOLD"
                else:
                    short_result = load_short_model(sym)
                    if short_result:
                        sm, sfcols = short_result
                        for c in sfcols:
                            if c not in df.columns:
                                df[c] = np.nan
                        prob = float(sm.predict_proba(df[sfcols].iloc[[-1]])[0][1])
                    else:
                        prob = 0.5
                    signal = "COVER" if prob <= config.COVER_THRESHOLD else "HOLD"

                action = paper_trader.execute(signal, price, atr, sym)
                print(
                    f"[live] {sym} {side} ${price:,.4f} | ML: {prob:.1%} | "
                    f"uPnL: ${upnl:+,.2f} | {signal} → {action}"
                )
                if action.startswith("ML-CLOSE"):
                    paper_trader._clear_live_state(sym)
                    if upnl >= 0:
                        paper_trader._wins += 1
                        outcome_emoji = "✅ WIN"
                    else:
                        paper_trader._losses += 1
                        outcome_emoji = "❌ LOSS"
                    dir_emoji = "📈" if side == "long" else "📉"
                    close_telegram = (
                        f"{outcome_emoji} *ML-Exit*\n"
                        f"{dir_emoji} {sym} {side.upper()} @ ${price:,.4f}\n"
                        f"uPnL: ${upnl:+,.2f}\n"
                        f"Portfolio: ${portfolio_val:,.2f} | Balance: ${balance:,.2f}\n"
                        f"{paper_trader._fmt_stats()}"
                    )
                    send_telegram(close_telegram)
                    paper_trader.log_trade(now, sym, signal, price, prob, action, pnl_usd=upnl)
                else:
                    paper_trader.log_trade(now, sym, signal, price, prob, action)

            except Exception as e:
                print(f"[live] monitor {sym}: {e}")

        # Phase 2: scan for new entries across all symbols (no concurrent trade limit)
        open_positions = paper_trader._open_positions()
        n_open = len(open_positions)
        open_syms = {pos["symbol"].split("/")[0] + "USDT" for pos in open_positions}
        open_upnl = sum(float(p.get("unrealizedPnl", 0)) for p in open_positions)

        msg_header = (
            f"*Kelly Bot — {now}*\n"
            f"💰 Portfolio: ${portfolio_val:,.2f} | Free: ${balance:,.2f}\n"
            f"📊 {paper_trader._fmt_stats()} | Open: {n_open} (uPnL ${open_upnl:+,.2f})\n"
            f"Risk: {config.RISK_PER_TRADE:.0%}/trade"
        )

        utc_hour = datetime.now(timezone.utc).hour
        if not (5 <= utc_hour < 20):
            print(f"[live] Outside entry session ({utc_hour:02d} UTC) — skipping scan")
            return

        fng = fetch_fear_greed()
        print(f"[live] Fear & Greed: {fng}")

        candidates = _scan_symbols()
        if not candidates:
            print("[live] No scan results.")
            return

        any_signal = False
        for sym, buy_prob, sh_prob, sym_price, sym_atr, sym_df in candidates:
            if sym in open_syms:
                continue

            signal    = "HOLD"
            exec_prob = 0.0
            fng_long_ok  = not config.ENABLE_FNG or _fng_allows_long(fng)
            fng_short_ok = not config.ENABLE_FNG or _fng_allows_short(fng)
            if config.ENABLE_LONGING and fng_long_ok and buy_prob >= config.BUY_THRESHOLD and _regime_ok(sym_df, buy_prob):
                # Walk-fwd precision guard: skip LONG if model is below threshold
                mf = model_file_for(sym)
                wf_prec = 0.0
                if os.path.exists(mf):
                    try:
                        with open(mf, "rb") as _f:
                            wf_prec = pickle.load(_f).get("walk_fwd_precision", 0.0)
                    except Exception:
                        wf_prec = 0.0
                if wf_prec < config.MIN_LONG_WF_PRECISION:
                    print(f"[live] {sym}: LONG skipped — walk-fwd {wf_prec:.1%} < {config.MIN_LONG_WF_PRECISION:.0%}")
                else:
                    signal = "BUY"
                    exec_prob = buy_prob
            if signal == "HOLD" and (
                config.ENABLE_SHORTING
                and fng_short_ok
                and sh_prob >= config.SHORT_THRESHOLD
                and _bearish_regime_ok(sym_df, sh_prob)
            ):
                signal = "SHORT"
                exec_prob = sh_prob

            if signal == "HOLD":
                continue

            any_signal = True
            action = paper_trader.execute(signal, sym_price, sym_atr, sym)
            print(f"[live] Signal: {signal} → {action}")

            if not action.startswith("HOLD"):
                paper_trader._entries += 1
                dir_emoji = "📈" if signal == "BUY" else "📉"
                entry_telegram = (
                    f"{dir_emoji} *{signal} Signal* — {sym}\n"
                    f"Price: ${sym_price:,.4f} | ML: {exec_prob:.1%}\n"
                    f"{action}\n"
                    f"{msg_header}"
                )
                send_telegram(entry_telegram)
                open_syms.add(sym)
            else:
                send_telegram(f"{msg_header}\n{sym}: {signal} ({exec_prob:.1%}) — {action}")
            paper_trader.log_trade(now, sym, signal, sym_price, exec_prob, action)

        if not any_signal:
            fng_label = "Extreme Fear" if fng < 25 else "Fear" if fng < 45 else "Neutral" if fng < 55 else "Greed" if fng < 75 else "Extreme Greed"
            print(f"[live] Signal: HOLD → HOLD — no qualifying signals (F&G: {fng} {fng_label})")
            send_telegram(f"{msg_header}\nF&G: {fng} ({fng_label})\n-- HOLD — no qualifying signals --")

    except Exception as e:
        err = f"[live] ERROR: {e}"
        print(err)
        send_telegram(f"Live bot error at {now}:\n{err}")


def run_bot():
    if config.LIVE_TRADING:
        _run_live_bot()
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'-'*55}")
    print(f"[bot] Running at {now}")

    try:
        # ── Phase 1: Monitor open SHORT position (exit checks) ────────────────
        if paper_trader.short_position > 0:
            sym = paper_trader.short_symbol
            df, htf_df = fetch_latest_data(sym)
            df    = add_features(df, htf_df)
            price = float(df["close"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])

            short_prob = 0.0
            short_result = load_short_model(sym)
            if short_result:
                sm, sfcols = short_result
                for c in sfcols:
                    if c not in df.columns:
                        df[c] = np.nan
                short_prob = float(sm.predict_proba(df[sfcols].iloc[[-1]])[0][1])

            signal = "COVER" if short_prob <= config.COVER_THRESHOLD else "HOLD"
            action = paper_trader.execute(signal, price, atr, sym)
            portfolio_val = paper_trader.portfolio_value(price)
            total_return  = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100
            win_rate      = (paper_trader.wins / max(paper_trader.trades, 1)) * 100

            print(f"[bot] {sym} ${price:,.4f} | short_prob={short_prob:.1%} | {signal} → {action}")
            print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%) | WR: {win_rate:.0f}%")

            msg = (
                f"*{sym} Short Monitor — {now}*\n"
                f"Price: ${price:,.4f} | ML: {short_prob:.1%}\n"
                f"Action: {action}\n"
                f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
                f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
            )
            send_telegram(msg)
            paper_trader.log_trade(now, sym, signal, price, short_prob, action)
            return

        # ── Phase 2: Monitor open LONG position (exit checks) ─────────────────
        if paper_trader.position > 0:
            sym  = paper_trader.position_symbol
            df, htf_df = fetch_latest_data(sym)
            df    = add_features(df, htf_df)
            price = float(df["close"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])

            model, fcols = load_model(sym)
            for c in fcols:
                if c not in df.columns:
                    df[c] = np.nan
            buy_prob = float(model.predict_proba(df[fcols].iloc[[-1]])[0][1])

            signal = "SELL" if buy_prob <= config.SELL_THRESHOLD else "HOLD"
            action = paper_trader.execute(signal, price, atr, sym)
            portfolio_val = paper_trader.portfolio_value(price)
            total_return  = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100
            win_rate      = (paper_trader.wins / max(paper_trader.trades, 1)) * 100

            print(f"[bot] {sym} ${price:,.4f} | prob={buy_prob:.1%} | {signal} → {action}")
            print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%) | "
                  f"WR: {win_rate:.0f}%")

            msg = (
                f"*{sym} Monitor — {now}*\n"
                f"Price: ${price:,.4f} | ML: {buy_prob:.1%}\n"
                f"Action: {action}\n"
                f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
                f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
            )
            send_telegram(msg)
            paper_trader.log_trade(now, sym, signal, price, buy_prob, action)
            return

        # ── Phase 3: Scan for best entry opportunity (LONG or SHORT) ──────────
        # Session gate: only enter 05:00-20:00 UTC (active liquidity hours)
        utc_hour = datetime.now(timezone.utc).hour
        if not (5 <= utc_hour < 20):
            print(f"[bot] Outside entry session ({utc_hour:02d} UTC) — skipping scan")
            return

        candidates = _scan_symbols()
        if not candidates:
            print("[bot] No scan results.")
            return

        # Best BUY: highest buy_prob (already sorted)
        best_sym, best_prob, _, best_price, best_atr, best_df = candidates[0]
        # Best SHORT: highest short_prob across all candidates
        best_short = max(candidates, key=lambda x: x[2])
        sh_sym, _, sh_prob, sh_price, sh_atr, sh_df = best_short

        trend_tag  = "↑" if _regime_ok(best_df, best_prob) else "↓regime"
        print(f"[bot] Best BUY:   {best_sym} prob={best_prob:.1%} ${best_price:,.4f} {trend_tag}")
        for sym, prob, s_prob, price, _, _ in candidates[1:]:
            s_tag = f" | short={s_prob:.1%}" if config.ENABLE_SHORTING else ""
            print(f"      {sym} buy={prob:.1%} ${price:,.4f}{s_tag}")
        if config.ENABLE_SHORTING:
            sh_tag = "↓" if _bearish_regime_ok(sh_df, sh_prob) else "↑regime"
            print(f"[bot] Best SHORT: {sh_sym} prob={sh_prob:.1%} ${sh_price:,.4f} {sh_tag}")

        fng = fetch_fear_greed()
        fng_label = "Extreme Fear" if fng < 25 else "Fear" if fng < 45 else "Neutral" if fng < 55 else "Greed" if fng < 75 else "Extreme Greed"
        fng_long_ok  = not config.ENABLE_FNG or _fng_allows_long(fng)
        fng_short_ok = not config.ENABLE_FNG or _fng_allows_short(fng)
        print(f"[bot] Fear & Greed: {fng} ({fng_label}) | long_ok={fng_long_ok} short_ok={fng_short_ok}")

        signal     = "HOLD"
        action_sym = best_sym
        action_price, action_atr = best_price, best_atr
        exec_prob  = best_prob

        if config.ENABLE_LONGING and fng_long_ok and best_prob >= config.BUY_THRESHOLD and _regime_ok(best_df, best_prob):
            signal = "BUY"
        elif config.ENABLE_SHORTING and fng_short_ok and sh_prob >= config.SHORT_THRESHOLD and _bearish_regime_ok(sh_df, sh_prob):
            signal = "SHORT"
            action_sym, action_price, action_atr = sh_sym, sh_price, sh_atr
            exec_prob = sh_prob

        action = paper_trader.execute(signal, action_price, action_atr, action_sym)
        portfolio_val = paper_trader.portfolio_value(action_price)
        total_return  = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100
        win_rate      = (paper_trader.wins / max(paper_trader.trades, 1)) * 100

        print(f"[bot] Signal: {signal} → {action}")
        print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%) | WR: {win_rate:.0f}%")

        msg = (
            f"*{action_sym} Signal — {now}*\n"
            f"Price: ${action_price:,.4f}\n"
            f"ML signal: *{signal}* ({exec_prob:.1%}) {trend_tag}\n"
            f"F&G: {fng} ({fng_label})\n"
            f"Action: {action}\n"
            f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
            f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
        )
        send_telegram(msg)
        paper_trader.log_trade(now, action_sym, signal, action_price, exec_prob, action)

    except Exception as e:
        err = f"[bot] ERROR: {e}"
        print(err)
        send_telegram(f"Bot error at {now}:\n{err}")


# ─── Scheduler ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    mode_label = "LIVE FUTURES" if config.LIVE_TRADING else "paper"
    print(f" Crypto ML Bot v3 — Multi-pair, Regime-aware [{mode_label}]")
    print(f" Pairs:    {', '.join(config.SYMBOLS)}")
    print(f" Interval: {config.INTERVAL}")
    if config.LIVE_TRADING:
        print(f" Exchange: {config.FUTURES_EXCHANGE} | Leverage: dynamic (max before liquidation, cap 40x)")
        print(f" Max concurrent trades: {config.MAX_CONCURRENT_TRADES}")
    else:
        print(f" Balance:  ${config.PAPER_STARTING_BALANCE:,.0f} (paper)")
    print(f" BUY threshold: {config.BUY_THRESHOLD:.0%} | Risk/trade: {config.RISK_PER_TRADE:.0%}")
    if config.ENABLE_SHORTING:
        print(f" SHORT threshold: {config.SHORT_THRESHOLD:.0%} | Shorting: ENABLED")
    print("=" * 55)

    if config.LIVE_TRADING and isinstance(paper_trader, LiveTrader):
        paper_trader.run_heartbeat_trade()

    run_bot()

    scheduler = BlockingScheduler()
    interval_map = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "1d": 1440,
    }
    minutes = interval_map.get(config.INTERVAL, 60)
    scheduler.add_job(run_bot, "interval", minutes=minutes)

    print(f"\n[bot] Scheduler running every {minutes} min. Ctrl+C to stop.\n")
    send_telegram(
        f"Kelly Binance Started\n"
        f"Pairs: {', '.join(config.SYMBOLS)}\n"
        f"Interval: {config.INTERVAL} | BUY≥{config.BUY_THRESHOLD:.0%}"
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[bot] Stopped.")
        send_telegram("Bot stopped.")
