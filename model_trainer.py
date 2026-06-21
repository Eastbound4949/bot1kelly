"""
model_trainer.py v3 — multi-timeframe, regime-aware, recency-weighted XGBoost.

Upgrades over v2:
  - Higher-timeframe (HTF) features for trend confirmation
  - Regime features: ADX classification, volatility percentile, EMA alignment
  - Reward/risk label filter (forward upside must be >= 1.5x forward downside)
  - Exponential recency sample weights (recent candles matter more)
  - CalibratedClassifierCV for reliable probability estimates
  - Shorter label horizon (4 candles) and higher threshold (1%) for precision
"""

import os
import pickle
import warnings
from datetime import datetime

import time

import ccxt
import numpy as np
import pandas as pd
import ta
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import TimeSeriesSplit

import config

warnings.filterwarnings("ignore")

# ─── Label config ─────────────────────────────────────────────────────────────
LABEL_HORIZON       = 6      # Look ahead N candles
LABEL_THRESHOLD     = 0.005  # Price must rise >0.5% — less sparse, model learns better
STOP_LOSS_THRESHOLD = 0.025  # Mirrors config.STOP_LOSS_PCT
# REWARD_RISK_MIN removed — the bot's ATR trailing stop handles R:R at execution time.
# Pre-filtering labels by R:R just starves the model of positive samples.
TOP_N_FEATURES      = 25     # Fewer features → less overfitting
RECENCY_DECAY       = 0.9997 # Per-candle exponential weight decay

# Maps each base interval to its higher timeframe for trend confirmation
HIGHER_TF_MAP = {
    "1m": "5m",  "3m": "15m", "5m": "15m",
    "15m": "1h", "30m": "2h", "1h": "4h",
    "2h": "6h",  "4h": "1d",  "6h": "1d", "1d": "1d",
}

FEATURE_COLS: list[str] = []  # Populated at training time; bot reads from pickle


# ─── Exchange client ──────────────────────────────────────────────────────────
# EXCHANGE env var controls which exchange to use.
# "bybit" works from any region including Railway US servers.
# Exchange geo-restriction matrix:
#   binance → blocked on US IPs (Railway US, most VPS)
#   bybit   → blocked on US IPs via CloudFront
#   kucoin  → available globally, no geo-restrictions ← Railway default
#   gateio  → available globally, no geo-restrictions
#   kraken  → available globally, no geo-restrictions
# Set EXCHANGE env var in Railway to override.

_CCXT_INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "1d": "1d",
}

# KuCoin symbol format: BTC/USDT (same as our conversion logic) ✓
# Gate.io symbol format: BTC/USDT ✓
# Kraken symbol format: BTC/USDT ✓ (for USDT pairs)
_EXCHANGE_SYMBOL_OVERRIDES: dict[str, dict[str, str]] = {
    "kraken": {
        "BTCUSDT": "XBT/USDT",  "ETHUSDT": "ETH/USDT",  "SOLUSDT":  "SOL/USDT",
        "BNBUSDT": "BNB/USDT",  "XRPUSDT": "XRP/USDT",  "AVAXUSDT": "AVAX/USDT",
        "DOGEUSDT": "DOGE/USDT","LINKUSDT": "LINK/USDT", "ADAUSDT":  "ADA/USDT",
        "DOTUSDT": "DOT/USDT",
    },
}


def _get_exchange() -> ccxt.Exchange:
    exchange_id = os.environ.get("EXCHANGE", "kucoin")
    exchange_class = getattr(ccxt, exchange_id)
    # OHLCV is public data — no credentials needed.
    # binanceusdm = Binance USDT-M futures; needs defaultType=future to access futures OHLCV.
    # All other exchanges use spot.
    default_type = "future" if exchange_id in ("binanceusdm", "binancecoinm") else "spot"
    ex = exchange_class({"options": {"defaultType": default_type}})
    ex.load_markets()
    return ex


def _to_ccxt_symbol(exchange: ccxt.Exchange, symbol: str) -> str:
    """Convert BTCUSDT → BTC/USDT, with exchange-specific overrides."""
    exchange_id = exchange.id
    overrides = _EXCHANGE_SYMBOL_OVERRIDES.get(exchange_id, {})
    if symbol in overrides:
        return overrides[symbol]
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    return symbol


def _lookback_to_since_ms(lookback: str) -> int:
    """Convert '730 day ago UTC' → Unix timestamp in milliseconds."""
    import re
    m = re.match(r"(\d+)\s+day", lookback)
    days = int(m.group(1)) if m else 365
    return int((time.time() - days * 86400) * 1000)


def _ohlcv_to_df(ohlcv: list, symbol: str) -> pd.DataFrame:
    """Convert ccxt OHLCV list [[ts, o, h, l, c, v], ...] to DataFrame."""
    df = pd.DataFrame(ohlcv, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    # taker_buy_base not available via ccxt OHLCV — use 50% volume as neutral proxy
    df["taker_buy_base"] = df["volume"] * 0.5
    return df[["open", "high", "low", "close", "volume", "taker_buy_base"]]


def _fetch_full_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Paginate ccxt fetch_ohlcv until we have all candles since `since_ms`."""
    ccxt_sym = _to_ccxt_symbol(exchange, symbol)
    all_ohlcv = []
    limit = 1000
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(ccxt_sym, timeframe, since=since, limit=limit)
        if not batch:
            break
        all_ohlcv.extend(batch)
        if len(batch) < limit:
            break
        since = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    return _ohlcv_to_df(all_ohlcv, symbol)


def fetch_binance_data(symbol: str, interval: str, lookback: str) -> pd.DataFrame:
    """Fetch base-TF OHLCV. Exchange determined by EXCHANGE env var."""
    exchange = _get_exchange()
    since_ms = _lookback_to_since_ms(lookback)
    tf = _CCXT_INTERVAL_MAP.get(interval, interval)
    df = _fetch_full_ohlcv(exchange, symbol, tf, since_ms)
    print(f"[trainer] {symbol} {interval} ({os.environ.get('EXCHANGE','kucoin')}): {len(df)} candles")
    return df


def fetch_htf_data(symbol: str, base_interval: str, lookback: str) -> pd.DataFrame | None:
    """Fetch higher-timeframe candles. Returns None if HTF == base interval."""
    htf = HIGHER_TF_MAP.get(base_interval, base_interval)
    if htf == base_interval:
        return None
    exchange = _get_exchange()
    since_ms = _lookback_to_since_ms(lookback)
    tf = _CCXT_INTERVAL_MAP.get(htf, htf)
    df = _fetch_full_ohlcv(exchange, symbol, tf, since_ms)
    print(f"[trainer] {symbol} {htf} HTF: {len(df)} candles")
    return df


# ─── Feature engineering ──────────────────────────────────────────────────────

def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # Momentum
    df["rsi"]        = ta.momentum.RSIIndicator(close, 14).rsi()
    df["rsi_6"]      = ta.momentum.RSIIndicator(close, 6).rsi()
    df["stoch_k"]    = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df["stoch_d"]    = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
    df["cci"]        = ta.trend.CCIIndicator(high, low, close).cci()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    df["mfi"]        = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index()

    # Trend
    macd = ta.trend.MACD(close)
    df["macd_diff"]   = macd.macd_diff()
    df["macd_diff_n"] = macd.macd_diff() / close
    adx = ta.trend.ADXIndicator(high, low, close)
    df["adx"]         = adx.adx()
    df["adx_pos"]     = adx.adx_pos()
    df["adx_neg"]     = adx.adx_neg()
    df["ema_9"]       = ta.trend.EMAIndicator(close, 9).ema_indicator()
    df["ema_21"]      = ta.trend.EMAIndicator(close, 21).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(close, 50).ema_indicator()
    df["ema_200"]     = ta.trend.EMAIndicator(close, 200).ema_indicator()

    # Volatility
    bb = ta.volatility.BollingerBands(close, 20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb.bollinger_mavg()
    df["bb_pos"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["atr_pct"]  = df["atr"] / close

    # Volume
    df["vol_ratio"] = vol / vol.rolling(20).mean()
    df["vol_z"]     = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)
    obv = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["obv_diff"]  = (obv - obv.ewm(span=20).mean()) / (obv.ewm(span=20).mean().abs() + 1e-9)

    # Taker buy pressure (crypto-specific aggression signal)
    taker = df["taker_buy_base"]
    df["taker_ratio"]   = taker / (vol + 1e-9)
    df["taker_ratio_z"] = (df["taker_ratio"] - df["taker_ratio"].rolling(20).mean()) / (
        df["taker_ratio"].rolling(20).std() + 1e-9
    )

    # Price-derived
    df["returns"]    = close.pct_change()
    df["log_ret"]    = np.log(close / close.shift(1))
    df["range_pct"]  = (high - low) / close
    df["body_pct"]   = (close - df["open"]).abs() / (high - low + 1e-9)
    df["upper_wick"] = (high - np.maximum(close, df["open"])) / (high - low + 1e-9)
    df["lower_wick"] = (np.minimum(close, df["open"]) - low) / (high - low + 1e-9)

    # Price vs EMAs (normalized — scale-independent across symbols)
    df["price_vs_ema9"]   = (close - df["ema_9"])   / df["ema_9"]
    df["price_vs_ema21"]  = (close - df["ema_21"])  / df["ema_21"]
    df["price_vs_ema50"]  = (close - df["ema_50"])  / df["ema_50"]
    df["price_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]
    df["above_ema200"]    = (close > df["ema_200"]).astype(int)
    df["ema9_vs_ema21"]   = (df["ema_9"] - df["ema_21"]) / df["ema_21"]
    df["ema21_vs_ema50"]  = (df["ema_21"] - df["ema_50"]) / df["ema_50"]

    # Multi-period returns
    df["ret_3"]  = close.pct_change(3)
    df["ret_6"]  = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)
    df["ret_24"] = close.pct_change(24)

    # Rolling stats
    df["ret_mean_12"] = df["returns"].rolling(12).mean()
    df["ret_std_12"]  = df["returns"].rolling(12).std()
    df["ret_mean_24"] = df["returns"].rolling(24).mean()

    # Time features — crypto has strong session/weekly patterns
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Lag features — short-term momentum memory
    for col in ["rsi", "macd_diff", "vol_ratio", "returns", "bb_pos"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market regime: trending vs ranging vs volatile."""
    close = df["close"]

    # ADX regime: strong trend when ADX > 25
    df["regime_trending"] = (df["adx"] > 25).astype(int)

    # Realized vol percentile (annualised, rolling 20-bar)
    rvol = df["returns"].rolling(20).std() * np.sqrt(252 * 24)
    df["rvol_pct"]        = rvol.rank(pct=True)
    df["regime_low_vol"]  = (df["rvol_pct"] < 0.4).astype(int)
    df["regime_high_vol"] = (df["rvol_pct"] > 0.7).astype(int)

    # EMA stack alignment: 0 (fully bearish) → 3 (fully bullish)
    df["ema_alignment"] = (
        (df["ema_9"]  > df["ema_21"]).astype(int) +
        (df["ema_21"] > df["ema_50"]).astype(int) +
        (df["ema_50"] > df["ema_200"]).astype(int)
    )

    # Distance from 200 EMA (trend strength proxy)
    df["trend_strength"] = (close - df["ema_200"]) / df["ema_200"]

    return df


def _htf_features(df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    """Merge higher-timeframe indicators onto base-TF index (forward-fill)."""
    c = htf_df["close"]
    h = htf_df["high"]
    lo = htf_df["low"]

    htf = pd.DataFrame(index=htf_df.index)
    htf["htf_rsi"]         = ta.momentum.RSIIndicator(c, 14).rsi()
    m = ta.trend.MACD(c)
    htf["htf_macd_diff_n"] = m.macd_diff() / c
    adx_h = ta.trend.ADXIndicator(h, lo, c)
    htf["htf_adx"]         = adx_h.adx()
    htf["htf_adx_pos"]     = adx_h.adx_pos()
    htf["htf_adx_neg"]     = adx_h.adx_neg()
    ema50_h  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    ema200_h = ta.trend.EMAIndicator(c, 200).ema_indicator()
    htf["htf_above_ema50"]  = (c > ema50_h).astype(int)
    htf["htf_above_ema200"] = (c > ema200_h).astype(int)
    bb_h = ta.volatility.BollingerBands(c, 20)
    htf["htf_bb_pos"] = (c - bb_h.bollinger_lband()) / (
        bb_h.bollinger_hband() - bb_h.bollinger_lband() + 1e-9
    )
    # Primary HTF gate: both EMAs stacked and price above both
    htf["htf_trend_up"] = ((c > ema50_h) & (ema50_h > ema200_h)).astype(int)
    htf["htf_regime_trending"] = (adx_h.adx() > 20).astype(int)

    # Forward-fill HTF onto base TF (HTF bar holds until the next HTF bar closes)
    merged = htf.reindex(df.index, method="ffill")
    for col in merged.columns:
        df[col] = merged[col]
    return df


def add_features(df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full feature pipeline. htf_df is optional; skip HTF block if None."""
    df = df.copy()
    df = _base_features(df)
    df = _regime_features(df)
    if htf_df is not None:
        df = _htf_features(df, htf_df)
    df.dropna(inplace=True)
    return df


# ─── Labels ───────────────────────────────────────────────────────────────────

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUY (1) when ALL of:
    1. Price rises > LABEL_THRESHOLD% at exactly LABEL_HORIZON candles ahead
    2. Price never breaches stop-loss during the hold window
    3. Max upside / max downside >= REWARD_RISK_MIN (quality filter — avoids marginal setups)
    """
    close = df["close"]
    fwd_ret = close.shift(-LABEL_HORIZON) / close - 1

    # Minimum return during the hold window — stop-loss trip check.
    # R:R is handled by the bot's ATR trailing stop at execution time.
    min_fwd = pd.Series(np.inf, index=df.index)
    for i in range(1, LABEL_HORIZON + 1):
        ret = close.shift(-i) / close - 1
        min_fwd = np.fmin(min_fwd, ret)

    df["label"] = (
        (fwd_ret > LABEL_THRESHOLD) &
        (min_fwd > -STOP_LOSS_THRESHOLD)
    ).astype(int)

    df.dropna(subset=["label"], inplace=True)
    return df.iloc[:-LABEL_HORIZON]


# ─── Sample weights ───────────────────────────────────────────────────────────

def recency_weights(n: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    """Exponential decay — recent candles receive more training weight."""
    w = decay ** np.arange(n - 1, -1, -1)
    return (w / w.sum() * n).astype(np.float32)


# ─── Isotonic calibration wrapper ────────────────────────────────────────────

class _IsotonicCalibrated:
    """
    Wraps a fitted XGBoost classifier and applies isotonic regression to
    map raw probabilities to empirically calibrated values.
    Works with any sklearn version — no cv="prefit" dependency.
    """

    def __init__(self, base_model):
        self._base = base_model
        self._iso  = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, X, y):
        raw = self._base.predict_proba(X)[:, 1]
        self._iso.fit(raw, y)
        self._fitted = True
        return self

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)[:, 1]
        if self._fitted:
            cal = self._iso.transform(raw)
        else:
            cal = raw
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Pass-through so feature_importances_ still works if needed
    @property
    def feature_importances_(self):
        return self._base.feature_importances_


# ─── Feature selection ────────────────────────────────────────────────────────

def _select_top_features(model, cols: list, top_n: int) -> list:
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    selected = [cols[i] for i in idx]
    print(f"\n[trainer] Top 10 features by importance:")
    for rank, col in enumerate(selected[:10], 1):
        print(f"  {rank:2d}. {col:38s} {imp[idx[rank-1]]:.4f}")
    return selected


# ─── Walk-forward validation ──────────────────────────────────────────────────

def _walk_forward_precision(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, threshold: float | None = None
) -> tuple[float, float, float]:
    """Returns (precision_default, precision_at_threshold, std_default)."""
    th = threshold if threshold is not None else config.BUY_THRESHOLD
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores_default, scores_thresh = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        w_tr = recency_weights(len(y_tr))
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=8, gamma=2,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        preds   = m.predict(X_te)
        probs   = m.predict_proba(X_te)[:, 1]
        mask_th = probs >= th
        scores_default.append(precision_score(y_te, preds, pos_label=1, zero_division=0))
        if mask_th.sum() > 0:
            scores_thresh.append((y_te[mask_th] == 1).mean())
    prec_def = float(np.mean(scores_default))
    prec_thr = float(np.mean(scores_thresh)) if scores_thresh else 0.0
    return prec_def, prec_thr, float(np.std(scores_default))


# ─── Model file helper ────────────────────────────────────────────────────────

# DATA_DIR env var → Railway volume path (e.g. /data).
# Falls back to script directory so local usage is unchanged.
_BOT_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

def model_file_for(symbol: str) -> str:
    name = f"model_{symbol}_{config.INTERVAL}.pkl" if len(config.SYMBOLS) > 1 else config.MODEL_FILE
    return os.path.join(_BOT_DIR, name)


def short_model_file_for(symbol: str) -> str:
    return os.path.join(_BOT_DIR, f"short_model_{symbol}_{config.INTERVAL}.pkl")


# ─── Short labels ─────────────────────────────────────────────────────────────

def add_short_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    SHORT (1) when ALL of:
    1. Price falls > LABEL_THRESHOLD% at exactly LABEL_HORIZON candles ahead
    2. Price never rallies > STOP_LOSS_THRESHOLD during the hold window (no stop-out)
    """
    close   = df["close"]
    fwd_ret = close.shift(-LABEL_HORIZON) / close - 1

    max_fwd = pd.Series(-np.inf, index=df.index)
    for i in range(1, LABEL_HORIZON + 1):
        ret = close.shift(-i) / close - 1
        max_fwd = np.fmax(max_fwd, ret)

    df["label"] = (
        (fwd_ret < -LABEL_THRESHOLD) &
        (max_fwd < STOP_LOSS_THRESHOLD)
    ).astype(int)

    df.dropna(subset=["label"], inplace=True)
    return df.iloc[:-LABEL_HORIZON]


# ─── Train & save ─────────────────────────────────────────────────────────────

def train_and_save(symbol: str | None = None, model_file: str | None = None) -> dict:
    sym = symbol or config.SYMBOL
    mf  = model_file or model_file_for(sym)

    print(f"[trainer] Fetching {sym} {config.INTERVAL} | {config.LOOKBACK}...")
    df = fetch_binance_data(sym, config.INTERVAL, config.LOOKBACK)

    print("[trainer] Fetching higher-TF data...")
    htf_df = fetch_htf_data(sym, config.INTERVAL, config.LOOKBACK)

    print("[trainer] Engineering features...")
    df = add_features(df, htf_df)
    df = add_labels(df)

    skip = {
        "label", "open", "high", "low", "close", "volume", "taker_buy_base",
        "ema_9", "ema_21", "ema_50", "ema_200",   # use price_vs_ema* instead
        "bb_upper", "bb_lower",                    # use bb_pos, bb_width instead
        "macd", "macd_signal",                     # use macd_diff_n instead
        "atr",                                     # use atr_pct instead
    }
    all_cols = [c for c in df.columns if c not in skip]

    X = df[all_cols]
    y = df["label"]
    print(f"[trainer] Rows: {len(df)} | BUY rate: {y.mean():.1%} | Features: {len(all_cols)}")

    # 3-way split: train 70% / val 15% / test 15%
    # val  → early stopping only (never shown to calibrator or final eval)
    # test → calibration + honest hold-out metrics (zero model decisions used this data)
    n = len(X)
    tr_end  = int(n * 0.70)
    val_end = int(n * 0.85)
    X_tr,  y_tr  = X.iloc[:tr_end],       y.iloc[:tr_end]
    X_val, y_val = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    X_te,  y_te  = X.iloc[val_end:],       y.iloc[val_end:]

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    w_tr = recency_weights(len(y_tr))
    print(f"[trainer] scale_pos_weight: {pos_weight:.1f} | "
          f"train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    # Pass 1: all features, early stop on val (clean — val never influences test eval)
    m1 = xgb.XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, gamma=2,
        reg_alpha=0.5, reg_lambda=2,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42, verbosity=0,
        early_stopping_rounds=50,
    )
    m1.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    selected = _select_top_features(m1, all_cols, TOP_N_FEATURES)

    # Pass 2: retrain on train+val combined, fixed n_estimators from pass 1
    X_tv = pd.concat([X_tr, X_val])
    y_tv = pd.concat([y_tr, y_val])
    w_tv = recency_weights(len(y_tv))
    m2 = xgb.XGBClassifier(
        n_estimators=m1.best_iteration + 1,
        max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, gamma=2,
        reg_alpha=0.5, reg_lambda=2,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    m2.fit(X_tv[selected], y_tv, sample_weight=w_tv)

    # Calibrate on test set (first time this data is used — no leakage)
    calibrated = _IsotonicCalibrated(m2)
    if int(y_te.sum()) >= 10:
        calibrated.fit(X_te[selected], y_te)
        print("[trainer] Isotonic calibration fitted on test set.")
    else:
        print("[trainer] Skipping calibration (too few BUY samples in test set)")

    y_pred = calibrated.predict(X_te[selected])
    y_prob = calibrated.predict_proba(X_te[selected])[:, 1]

    print(f"\n[trainer] === Hold-out test (honest — zero leakage) ===")
    print(classification_report(y_te, y_pred, target_names=["HOLD", "BUY"]))

    print(f"\n[trainer] Precision by confidence threshold (TRUE out-of-sample):")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Signals':>10} {'Coverage':>10}")
    for thresh in [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]:
        mask = y_prob >= thresh
        if mask.sum() > 0:
            prec = (y_te[mask] == 1).mean()
            cov  = mask.sum() / len(y_te)
            flag = " <- TARGET" if thresh == config.BUY_THRESHOLD else ""
            print(f"  {thresh:>10.0%} {prec:>10.1%} {mask.sum():>10d} {cov:>10.1%}{flag}")

    print(f"\n[trainer] Walk-forward precision (5-fold)...")
    wf_def, wf_thr, wf_std = _walk_forward_precision(X[selected], y)
    # Break-even: TP=2.5×ATR, SL=1.5×ATR → R:R=1.67 → need win rate > 1/(1+1.67) = 37.5%
    break_even = 1.0 / (1.0 + config.TAKE_PROFIT_ATR_MULT / config.TRAIL_STOP_ATR_MULT)
    flag_def = "ok" if wf_def >= break_even else f"LOSING (need >{break_even:.0%})"
    flag_thr = "ok" if wf_thr >= break_even else f"LOSING (need >{break_even:.0%})"
    print(f"  @ default 0.5  threshold: {wf_def:.1%} ± {wf_std:.1%}  [{flag_def}]")
    print(f"  @ BUY threshold {config.BUY_THRESHOLD:.0%}: {wf_thr:.1%}  [{flag_thr}]")
    wf_mean = wf_thr if wf_thr > 0 else wf_def

    payload = {
        "model":            calibrated,
        "feature_cols":     selected,
        "trained_at":       datetime.utcnow().isoformat(),
        "label_horizon":    LABEL_HORIZON,
        "label_threshold":  LABEL_THRESHOLD,
        "symbol":           sym,
        "walk_fwd_precision": wf_mean,
    }
    with open(mf, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[trainer] Saved -> {mf}  (walk-fwd precision: {wf_mean:.1%})")
    return payload


def train_and_save_short(symbol: str | None = None, model_file: str | None = None) -> dict:
    """Train a SHORT/bearish model using inverted labels. Same architecture as BUY model."""
    sym = symbol or config.SYMBOL
    mf  = model_file or short_model_file_for(sym)

    print(f"[trainer] Fetching {sym} {config.INTERVAL} | {config.LOOKBACK}...")
    df = fetch_binance_data(sym, config.INTERVAL, config.LOOKBACK)

    print("[trainer] Fetching higher-TF data...")
    htf_df = fetch_htf_data(sym, config.INTERVAL, config.LOOKBACK)

    print("[trainer] Engineering features (SHORT labels)...")
    df = add_features(df, htf_df)
    df = add_short_labels(df)

    skip = {
        "label", "open", "high", "low", "close", "volume", "taker_buy_base",
        "ema_9", "ema_21", "ema_50", "ema_200",
        "bb_upper", "bb_lower",
        "macd", "macd_signal",
        "atr",
    }
    all_cols = [c for c in df.columns if c not in skip]

    X = df[all_cols]
    y = df["label"]
    print(f"[trainer] Rows: {len(df)} | SHORT rate: {y.mean():.1%} | Features: {len(all_cols)}")

    n = len(X)
    tr_end  = int(n * 0.70)
    val_end = int(n * 0.85)
    X_tr,  y_tr  = X.iloc[:tr_end],       y.iloc[:tr_end]
    X_val, y_val = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    X_te,  y_te  = X.iloc[val_end:],       y.iloc[val_end:]

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    w_tr = recency_weights(len(y_tr))
    print(f"[trainer] scale_pos_weight: {pos_weight:.1f} | "
          f"train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    m1 = xgb.XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, gamma=2,
        reg_alpha=0.5, reg_lambda=2,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42, verbosity=0,
        early_stopping_rounds=50,
    )
    m1.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    selected = _select_top_features(m1, all_cols, TOP_N_FEATURES)

    X_tv = pd.concat([X_tr, X_val])
    y_tv = pd.concat([y_tr, y_val])
    w_tv = recency_weights(len(y_tv))
    m2 = xgb.XGBClassifier(
        n_estimators=m1.best_iteration + 1,
        max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, gamma=2,
        reg_alpha=0.5, reg_lambda=2,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    m2.fit(X_tv[selected], y_tv, sample_weight=w_tv)

    calibrated = _IsotonicCalibrated(m2)
    if int(y_te.sum()) >= 10:
        calibrated.fit(X_te[selected], y_te)
        print("[trainer] Isotonic calibration fitted on test set.")
    else:
        print("[trainer] Skipping calibration (too few SHORT samples in test set)")

    y_pred = calibrated.predict(X_te[selected])
    y_prob = calibrated.predict_proba(X_te[selected])[:, 1]

    print(f"\n[trainer] === Hold-out test SHORT (honest — zero leakage) ===")
    print(classification_report(y_te, y_pred, target_names=["HOLD", "SHORT"]))

    print(f"\n[trainer] Precision by confidence threshold (TRUE out-of-sample):")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Signals':>10} {'Coverage':>10}")
    for thresh in [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]:
        mask = y_prob >= thresh
        if mask.sum() > 0:
            prec = (y_te[mask] == 1).mean()
            cov  = mask.sum() / len(y_te)
            flag = " <- TARGET" if thresh == getattr(config, "SHORT_THRESHOLD", 0.50) else ""
            print(f"  {thresh:>10.0%} {prec:>10.1%} {mask.sum():>10d} {cov:>10.1%}{flag}")

    print(f"\n[trainer] Walk-forward precision (5-fold)...")
    _, wf_thr, wf_std = _walk_forward_precision(X[selected], y, threshold=getattr(config, "SHORT_THRESHOLD", 0.50))
    break_even = 1.0 / (1.0 + config.TAKE_PROFIT_ATR_MULT / config.TRAIL_STOP_ATR_MULT)
    flag = "ok" if wf_thr >= break_even else f"LOSING (need >{break_even:.0%})"
    print(f"  @ SHORT threshold {getattr(config, "SHORT_THRESHOLD", 0.50):.0%}: {wf_thr:.1%} ± {wf_std:.1%}  [{flag}]")

    payload = {
        "model":              calibrated,
        "feature_cols":       selected,
        "trained_at":         datetime.utcnow().isoformat(),
        "label_horizon":      LABEL_HORIZON,
        "label_threshold":    LABEL_THRESHOLD,
        "symbol":             sym,
        "model_type":         "short",
        "walk_fwd_precision": wf_thr,
    }
    with open(mf, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[trainer] Saved (SHORT) -> {mf}  (walk-fwd precision: {wf_thr:.1%})")
    return payload


if __name__ == "__main__":
    import sys
    # python model_trainer.py              → trains all symbols in config.SYMBOLS
    # python model_trainer.py ETHUSDT      → trains just ETHUSDT
    # python model_trainer.py BTCUSDT ETHUSDT SOLUSDT → trains all three
    symbols = sys.argv[1:] if len(sys.argv) > 1 else config.SYMBOLS
    for sym in symbols:
        print(f"\n{'='*55}")
        print(f" Training: {sym}")
        print(f"{'='*55}")
        train_and_save(sym)
