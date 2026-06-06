"""
tests/test_bot.py — Automated tests.
GitHub Actions runs these on every push.
If any test fails, the deploy is blocked.

Run locally: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model_trainer import add_features, FEATURE_COLS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_fake_ohlcv(n=200):
    """Generate fake price data so tests don't need a real Binance connection."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 30000 + np.cumsum(np.random.randn(n) * 200)
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.randint(500, 2000, n).astype(float),
    }, index=dates)
    return df


# ─── Feature engineering tests ───────────────────────────────────────────────

def test_add_features_returns_dataframe():
    df = make_fake_ohlcv()
    result = add_features(df)
    assert isinstance(result, pd.DataFrame)


def test_all_feature_cols_present():
    df = make_fake_ohlcv()
    result = add_features(df)
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing feature: {col}"


def test_no_nans_in_features():
    df = make_fake_ohlcv(300)
    result = add_features(df)
    for col in FEATURE_COLS:
        assert result[col].isna().sum() == 0, f"NaN in feature: {col}"


def test_rsi_in_valid_range():
    df = make_fake_ohlcv(300)
    result = add_features(df)
    assert result["rsi"].between(0, 100).all(), "RSI out of 0-100 range"


def test_vol_ratio_positive():
    df = make_fake_ohlcv(300)
    result = add_features(df)
    assert (result["vol_ratio"] > 0).all(), "Volume ratio has non-positive values"


# ─── Paper trader tests ───────────────────────────────────────────────────────

def test_paper_trader_import():
    from bot import PaperTrader
    trader = PaperTrader.__new__(PaperTrader)
    trader.balance = config.PAPER_STARTING_BALANCE
    trader.position = 0.0
    trader.entry_price = 0.0
    trader.trades = 0
    trader.wins = 0
    assert trader.balance == config.PAPER_STARTING_BALANCE


def test_paper_trader_buy():
    from bot import PaperTrader
    trader = PaperTrader.__new__(PaperTrader)
    trader.balance = 10_000.0
    trader.position = 0.0
    trader.entry_price = 0.0
    trader.trades = 0
    trader.wins = 0

    result = trader.execute("BUY", price=50_000)
    assert trader.position > 0, "Position should be > 0 after BUY"
    assert trader.balance < 10_000, "Balance should decrease after BUY"
    assert "BOUGHT" in result


def test_paper_trader_buy_then_sell():
    from bot import PaperTrader
    trader = PaperTrader.__new__(PaperTrader)
    trader.balance = 10_000.0
    trader.position = 0.0
    trader.entry_price = 0.0
    trader.trades = 0
    trader.wins = 0

    trader.execute("BUY", price=50_000)
    result = trader.execute("SELL", price=55_000)  # sell higher → profit
    assert trader.position == 0.0, "Position should be 0 after SELL"
    assert trader.balance > 10_000, "Balance should be higher after profitable SELL"
    assert "SOLD" in result


def test_paper_trader_hold_when_no_position():
    from bot import PaperTrader
    trader = PaperTrader.__new__(PaperTrader)
    trader.balance = 10_000.0
    trader.position = 0.0
    trader.entry_price = 0.0
    trader.trades = 0
    trader.wins = 0

    result = trader.execute("SELL", price=50_000)
    assert "HOLD" in result, "Should HOLD when SELL signal but no position"


# ─── Config validation tests ──────────────────────────────────────────────────

def test_buy_threshold_in_valid_range():
    assert 0.5 <= config.BUY_THRESHOLD <= 0.95, "BUY_THRESHOLD should be between 0.5 and 0.95"


def test_trade_size_sane():
    assert 0.1 <= config.TRADE_SIZE_PCT <= 1.0, "TRADE_SIZE_PCT should be between 0.1 and 1.0"


def test_starting_balance_positive():
    assert config.PAPER_STARTING_BALANCE > 0, "Starting balance must be positive"


# ─── Performance monitor tests ────────────────────────────────────────────────

def test_performance_monitor_import():
    from performance_monitor import PerformanceMonitor
    pm = PerformanceMonitor(log_file="nonexistent.csv")
    assert pm is not None


def test_sharpe_calculation():
    from performance_monitor import PerformanceMonitor
    pm = PerformanceMonitor.__new__(PerformanceMonitor)
    returns = pd.Series([0.01, -0.005, 0.02, 0.003, -0.01, 0.015])
    sharpe = pm._sharpe(returns)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
