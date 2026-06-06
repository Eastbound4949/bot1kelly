"""
performance_monitor.py — Reads the trade log, computes metrics, and
automatically promotes the bot to live trading when targets are hit.

This runs as a separate scheduled job alongside the main bot.
It checks performance weekly and sends you a Telegram report.

Usage: called automatically by bot.py — no need to run manually.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

import config


class PerformanceMonitor:
    """
    Watches trades_log.csv and decides when the bot is ready to go live.

    Promotion requires ALL of:
    - At least 4 weeks of paper trading
    - At least 50 completed trades
    - Sharpe ratio > 1.0
    - Win rate > 52%
    - Max drawdown < 20%
    """

    TARGETS = {
        "min_weeks":     4,
        "min_trades":    50,
        "min_sharpe":    1.0,
        "min_win_rate":  0.52,
        "max_drawdown":  0.20,
    }

    def __init__(self, log_file=None):
        self.log_file = log_file or config.LOG_FILE
        self.promotion_file = "promoted_to_live.flag"

    def _send_telegram(self, msg: str):
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        except Exception as e:
            print(f"[monitor] Telegram failed: {e}")

    def _load_log(self) -> pd.DataFrame:
        if not os.path.exists(self.log_file):
            return pd.DataFrame()
        df = pd.read_csv(self.log_file, parse_dates=["timestamp"])
        return df

    def _sharpe(self, returns: pd.Series, periods_per_year: int = 8760) -> float:
        """Annualised Sharpe ratio. periods_per_year=8760 for hourly data."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

    def _max_drawdown(self, portfolio_series: pd.Series) -> float:
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        return float(abs(drawdown.min()))

    def compute_metrics(self) -> dict:
        df = self._load_log()
        if df.empty or len(df) < 10:
            return {"error": "Not enough data yet"}

        # Time in operation
        start = df["timestamp"].min()
        end   = df["timestamp"].max()
        weeks_running = (end - start).days / 7

        # Returns: difference in portfolio value between rows
        df = df.sort_values("timestamp")
        df["port_return"] = df["portfolio_usd"].pct_change().fillna(0)

        # Completed round-trips (buy + sell = 1 trade)
        completed_trades = int(df["total_trades"].max())
        total_wins = int(df["total_wins"].max())
        win_rate = total_wins / max(completed_trades, 1)

        sharpe    = self._sharpe(df["port_return"])
        max_dd    = self._max_drawdown(df["portfolio_usd"])
        total_ret = (df["portfolio_usd"].iloc[-1] / config.PAPER_STARTING_BALANCE - 1) * 100

        return {
            "weeks_running":    round(weeks_running, 1),
            "completed_trades": completed_trades,
            "win_rate":         round(win_rate, 4),
            "sharpe":           round(sharpe, 3),
            "max_drawdown":     round(max_dd, 4),
            "total_return_pct": round(total_ret, 2),
            "portfolio_usd":    round(df["portfolio_usd"].iloc[-1], 2),
            "as_of":            str(end),
        }

    def check_promotion(self, metrics: dict) -> tuple[bool, list[str]]:
        """
        Returns (ready_to_promote, list_of_unmet_conditions).
        """
        if "error" in metrics:
            return False, [metrics["error"]]

        t = self.TARGETS
        unmet = []

        if metrics["weeks_running"] < t["min_weeks"]:
            remaining = t["min_weeks"] - metrics["weeks_running"]
            unmet.append(f"Need {remaining:.1f} more weeks of paper trading")

        if metrics["completed_trades"] < t["min_trades"]:
            remaining = t["min_trades"] - metrics["completed_trades"]
            unmet.append(f"Need {remaining} more completed trades")

        if metrics["sharpe"] < t["min_sharpe"]:
            unmet.append(f"Sharpe {metrics['sharpe']:.2f} < target {t['min_sharpe']}")

        if metrics["win_rate"] < t["min_win_rate"]:
            unmet.append(f"Win rate {metrics['win_rate']:.1%} < target {t['min_win_rate']:.1%}")

        if metrics["max_drawdown"] > t["max_drawdown"]:
            unmet.append(f"Max drawdown {metrics['max_drawdown']:.1%} > limit {t['max_drawdown']:.1%}")

        return len(unmet) == 0, unmet

    def promote_to_live(self, metrics: dict):
        """
        Write a flag file that bot.py checks.
        The bot will switch PAPER_MODE=False on next run.
        Also sends Telegram alert requiring manual confirmation.
        """
        if os.path.exists(self.promotion_file):
            return  # Already promoted

        with open(self.promotion_file, "w") as f:
            f.write(f"Promoted at {datetime.utcnow().isoformat()}\n")
            f.write(f"Metrics: {metrics}\n")

        msg = (
            "PROMOTION ALERT\n"
            "Your bot has hit all paper trading targets!\n\n"
            f"Sharpe: {metrics['sharpe']}\n"
            f"Win rate: {metrics['win_rate']:.1%}\n"
            f"Max drawdown: {metrics['max_drawdown']:.1%}\n"
            f"Weeks: {metrics['weeks_running']}\n"
            f"Trades: {metrics['completed_trades']}\n"
            f"Total return: {metrics['total_return_pct']:+.1f}%\n\n"
            "To go live: set LIVE_TRADING=true in Railway environment variables.\n"
            "WARNING: Only do this if you are comfortable with real risk."
        )
        self._send_telegram(msg)
        print("[monitor] Promotion flag written. Telegram alert sent.")

    def weekly_report(self):
        """Send a weekly performance summary to Telegram."""
        metrics = self.compute_metrics()
        if "error" in metrics:
            self._send_telegram(f"Weekly report: {metrics['error']}")
            return

        ready, unmet = self.check_promotion(metrics)

        status = "READY FOR LIVE" if ready else "Paper trading"
        unmet_str = "\n".join(f"  - {u}" for u in unmet) if unmet else "  All targets met!"

        msg = (
            f"Weekly bot report\n"
            f"Status: {status}\n\n"
            f"Portfolio: ${metrics['portfolio_usd']:,.2f} ({metrics['total_return_pct']:+.1f}%)\n"
            f"Sharpe: {metrics['sharpe']:.2f} (target >1.0)\n"
            f"Win rate: {metrics['win_rate']:.1%} (target >52%)\n"
            f"Max drawdown: {metrics['max_drawdown']:.1%} (limit <20%)\n"
            f"Trades: {metrics['completed_trades']} | Weeks: {metrics['weeks_running']}\n\n"
            f"Remaining targets:\n{unmet_str}"
        )
        self._send_telegram(msg)
        print(f"[monitor] Weekly report sent.\n{msg}")

        if ready:
            self.promote_to_live(metrics)

        return metrics


# Called by bot.py weekly scheduler
def run_weekly_check():
    monitor = PerformanceMonitor()
    return monitor.weekly_report()


if __name__ == "__main__":
    monitor = PerformanceMonitor()
    metrics = monitor.compute_metrics()
    print("\nCurrent metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    ready, unmet = monitor.check_promotion(metrics)
    print(f"\nReady for live trading: {ready}")
    if unmet:
        print("Unmet conditions:")
        for u in unmet:
            print(f"  - {u}")
