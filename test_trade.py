"""
test_trade.py — manual connectivity check: open + immediately close a minimal
BTC/USDT:USDT futures position. Proves the live order-execution path works.
Reads BINANCE_API_KEY / BINANCE_API_SECRET from environment (no hardcoded secrets).
"""
import os
import time
import ccxt

exchange = ccxt.binanceusdm({
    "apiKey": os.environ["BINANCE_API_KEY"],
    "secret": os.environ["BINANCE_API_SECRET"],
    "options": {"defaultType": "future"},
})
exchange.has["fetchCurrencies"] = False
exchange.load_markets()

symbol = "BTC/USDT:USDT"
price = exchange.fetch_ticker(symbol)["last"]

notional = 10.0
min_qty = exchange.markets[symbol]["limits"]["amount"]["min"] or 0.001
qty = float(exchange.amount_to_precision(symbol, max(notional / price, min_qty)))
print(f"BTC price: ${price:,.2f} | test qty: {qty} BTC (~${qty*price:.2f} notional)")

try:
    exchange.set_leverage(2, symbol)
except Exception as e:
    print(f"set_leverage warning: {e}")

try:
    hedged = bool(exchange.fapiPrivateGetPositionSideDual().get("dualSidePosition"))
except Exception:
    hedged = False
print(f"account mode: {'hedge' if hedged else 'one-way'}")

open_params = {"positionSide": "LONG"} if hedged else {}
close_params = {"positionSide": "LONG"} if hedged else {"reduceOnly": True}

order = exchange.create_order(symbol, "market", "buy", qty, params=open_params)
entry_price = float(order.get("average") or order.get("price") or price)
print(f"OPENED: bought {qty} BTC @ ~${entry_price:,.2f}")

time.sleep(3)

close = exchange.create_order(symbol, "market", "sell", qty, params=close_params)
exit_price = float(close.get("average") or close.get("price") or price)
print(f"CLOSED: sold {qty} BTC @ ~${exit_price:,.2f}")

pnl = (exit_price - entry_price) * qty
print(f"Gross P&L: ${pnl:+.4f} (before fees) | round-trip notional: ~${qty*entry_price:.2f}")
print("TEST COMPLETE — live order execution confirmed working.")
