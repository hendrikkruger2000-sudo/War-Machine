# Trader.py
import asyncio
import inspect
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import math
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# IMPORTANT: ensure this package is installed and configured for your environment.
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

# === Config ===
SYMBOL = "EURUSD_otc"
STAKE = 20.0
MODE = "demo"  # "demo" or "real"

# === Stats ===
class Stats:
    def __init__(self):
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    @property
    def win_rate(self):
        return (self.wins / self.trades) * 100 if self.trades > 0 else 0.0

stats = Stats()

# === Helpers ===
async def _maybe_await(maybe):
    if inspect.isawaitable(maybe):
        return await maybe
    return maybe

# === Candle Builder ===
class CandleBuilder:
    def __init__(self, max_candles=300):
        self.candle_data: List[Dict[str, Any]] = []
        self.current_candle: Optional[Dict[str, Any]] = None
        self.candle_start: Optional[datetime] = None
        self.indicators: Dict[str, Any] = {}
        self.max_candles = max_candles

    def process_tick(self, tick_time: datetime, price: float):
        bucket = tick_time.replace(second=0, microsecond=0)

        # Start new candle if none exists
        if self.candle_start is None or self.current_candle is None:
            self.candle_start = bucket
            self.current_candle = {
                "time": bucket,
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "ticks": [(tick_time, float(price))]
            }
            return

        # If new minute, finalize and start fresh
        if bucket > self.candle_start:
            self.finalize_current_candle()
            self.candle_start = bucket
            self.current_candle = {
                "time": bucket,
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "ticks": [(tick_time, float(price))]
            }
        else:
            c = self.current_candle
            c["high"] = max(c["high"], float(price))
            c["low"] = min(c["low"], float(price))
            c["close"] = float(price)
            c["ticks"].append((tick_time, float(price)))

    def finalize_current_candle(self):
        if not self.current_candle:
            return
        self.candle_data.append(self.current_candle)
        if len(self.candle_data) > self.max_candles:
            self.candle_data.pop(0)
        self.current_candle = None
        self.update_indicators()

    def update_indicators(self):
        if len(self.candle_data) < 30:
            return
        closes = [c["close"] for c in self.candle_data]
        highs = [c["high"] for c in self.candle_data]
        lows = [c["low"] for c in self.candle_data]

        ema8 = _ema(closes, 8)
        ema21 = _ema(closes, 21)
        rsi = _rsi(closes, 14)
        boll = _bollinger(closes, 20, 2.0)
        chop = _choppiness(highs, lows, closes, 14)
        mom = _roc(closes, 6)

        self.indicators.update({
            "ema8": ema8, "ema21": ema21, "rsi": rsi,
            "boll_low": boll[0], "boll_mid": boll[1], "boll_high": boll[2],
            "boll_percent_b": boll[4], "choppiness": chop, "momentum": mom
        })

# === Indicators ===
def _ema(series, period):
    if len(series) < period: return None
    k = 2/(period+1)
    ema = sum(series[:period])/period
    for x in series[period:]:
        ema = x*k + ema*(1-k)
    return ema

def _rsi(prices, period=14):
    if len(prices) < period+1: return None
    deltas = [prices[i]-prices[i-1] for i in range(1,len(prices))]
    gains = [max(d,0) for d in deltas[:period]]
    losses = [max(-d,0) for d in deltas[:period]]
    avg_gain = sum(gains)/period
    avg_loss = sum(losses)/period
    for d in deltas[period:]:
        gain, loss = max(d,0), max(-d,0)
        avg_gain = (avg_gain*(period-1)+gain)/period
        avg_loss = (avg_loss*(period-1)+loss)/period
    if avg_loss==0: return 100
    rs = avg_gain/avg_loss
    return 100-(100/(1+rs))

def _bollinger(prices, period=20, k=2.0):
    if len(prices) < period: return (None,None,None,None,None,None)
    window = prices[-period:]
    mean = sum(window)/period
    var = sum((x-mean)**2 for x in window)/period
    std = var**0.5
    upper, lower = mean+k*std, mean-k*std
    percent_b = (window[-1]-lower)/(upper-lower) if upper!=lower else 0.5
    return lower, mean, upper, None, percent_b, None

def _choppiness(highs, lows, closes, period=14):
    n = len(closes)
    if n < period+1: return None
    start = n - period
    tr_sum = 0.0
    for i in range(start, n):
        hi = highs[i]; lo = lows[i]; prev_close = closes[i-1]
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        tr_sum += tr
    highest_high = max(highs[start:n]); lowest_low = min(lows[start:n])
    range_hl = highest_high - lowest_low
    if range_hl <= 1e-9 or tr_sum <= 1e-9: return 100.0
    return 100.0 * math.log10(tr_sum / range_hl) / math.log10(period)

def _roc(prices, period=6):
    if len(prices) < period+1: return None
    prev = prices[-1-period]
    if prev == 0: return 0.0
    return (prices[-1]-prev)/prev

# === Strategy ===
def Trade_Strategy(cb) -> Tuple[Optional[str], float]:
    i = cb.indicators
    required = ["ema8","ema21","rsi","momentum","choppiness","boll_percent_b"]
    if not all(i.get(k) is not None for k in required):
        return None, 0.0

    score = 0.0
    direction = None

    # EMA trend filter: only if clearly diverging
    if abs(i["ema8"] - i["ema21"]) > 0.0005:
        if i["ema8"] > i["ema21"]:
            direction = "call"; score += 0.4
        elif i["ema8"] < i["ema21"]:
            direction = "put"; score += 0.4

    # RSI filter: avoid mid‑range noise
    if i["rsi"] < 30 and direction == "call":
        score += 0.2
    elif i["rsi"] > 70 and direction == "put":
        score += 0.2

    # Bollinger confirmation
    if i["boll_percent_b"] < 0.2 and direction == "call":
        score += 0.2
    elif i["boll_percent_b"] > 0.8 and direction == "put":
        score += 0.2

    # Momentum filter
    if direction == "call" and i["momentum"] > 0:
        score += 0.2
    elif direction == "put" and i["momentum"] < 0:
        score += 0.2

    # Choppiness filter: avoid flat markets
    if i["choppiness"] is not None and (i["choppiness"] < 40 or i["choppiness"] > 60):
        score += 0.2

    # Final decision: stricter threshold
    if score >= 0.8:
        return direction, score
    else:
        return None, score

# === Trading Loop ===
class TraderLoop:
    def __init__(self, symbol):
        self.symbol = symbol
        self.api: Optional[PocketOptionAsync] = None
        self.cb = CandleBuilder()
        self._stop = asyncio.Event()
        self.last_trade_candle: Optional[datetime] = None   # NEW guard

    async def start(self):
        self.api = await Connect()
        sub = await _maybe_await(self.api.subscribe_symbol(self.symbol))
        # async or sync iterable
        if hasattr(sub, "__aiter__"):
            async for tick in sub:
                if self._stop.is_set(): break
                await self._handle_tick(tick)
        else:
            for tick in sub:
                if self._stop.is_set(): break
                await self._handle_tick(tick)

    async def _handle_tick(self, tick: Dict[str, Any]):
        price = tick.get("close") or tick.get("price") or tick.get("open")
        if price is None:
            return

        ts_raw = tick.get("timestamp") or tick.get("time") or tick.get("ts")
        try:
            ts = float(ts_raw)
            if ts > 1e12:
                ts /= 1000.0
            tick_time = datetime.utcfromtimestamp(ts)
        except:
            tick_time = datetime.utcnow()

        # Build candle from tick
        self.cb.process_tick(tick_time, float(price))

        # Only finalize and evaluate strategy on candle close
        if tick_time.second == 0:
            self.cb.finalize_current_candle()

            if len(self.cb.candle_data) >= 30:
                direction, score = Trade_Strategy(self.cb)
                candle_time = self.cb.candle_data[-1]["time"]

                # Guard: only one trade per candle
                if direction and (self.last_trade_candle != candle_time):
                    await self.take_live_trade(direction, tick_time)
                    self.last_trade_candle = candle_time

    async def take_live_trade(self, direction: str, tick_time: datetime):
        try:
            if direction == "call":
                res = await _maybe_await(self.api.buy(self.symbol, STAKE, 60, check_win=False))
            else:
                res = await _maybe_await(self.api.sell(self.symbol, STAKE, 60, check_win=False))

            trade_id = res[0] if isinstance(res, (tuple, list)) else res

            # schedule non-blocking result resolution
            asyncio.create_task(self._resolve_trade(trade_id, direction, tick_time))

        except Exception as e:
            print(f"[ERROR] Trade placement failed: {e}")

    async def _resolve_trade(self, trade_id: Any, direction: str, tick_time: datetime):
        try:
            result_data = await _maybe_await(self.api.check_win(trade_id))
            result = (result_data or {}).get("result", "UNKNOWN")
            exit_price = (result_data or {}).get("closePrice", 0.0)

            stats.trades += 1
            if result == "win":
                stats.wins += 1
            elif result == "loss":
                stats.losses += 1
            else:
                stats.draws += 1

            print(f"[TRADE] {tick_time.isoformat()} | {direction.upper()} | Result: {result} | Exit: {exit_price}")

        except Exception as e:
            print(f"[ERROR] Trade result check failed: {e}")

    def stop(self):
        if not self._stop.is_set():
            self._stop.set()

# === Connection ===
async def Connect() -> PocketOptionAsync:
    ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
    ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'
    ssid = ssid_demo if MODE == "demo" else ssid_real

    api = PocketOptionAsync(ssid)
    print("[CONNECT] API initializing...")
    await asyncio.sleep(3)
    try:
        balance = await api.balance()
        print(f"[CONNECTED] Balance: {float(balance):.2f} | Stake: {STAKE:.2f}")
    except Exception as e:
        print(f"[BALANCE ERROR] {type(e).__name__}: {e}")
    return api

# === GUI ===
class TraderUI:
    def __init__(self, root: tk.Tk, trader_loop: TraderLoop):
        self.root = root
        self.trader_loop = trader_loop
        self.loop_thread: Optional[threading.Thread] = None

        self.root.title("War Machine Trader")
        self.root.geometry("1000x700")

        self.stats_label = tk.Label(root, text=self._stats_text(), font=("Consolas", 12))
        self.stats_label.pack(pady=8)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.start_btn = tk.Button(btn_frame, text="Start Bot", command=self.start_bot, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=8)
        self.stop_btn = tk.Button(btn_frame, text="Stop Bot", command=self.stop_bot, width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        self.fig, (self.ax_price, self.ax_rsi) = plt.subplots(
            2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [3, 1]}
        )
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._schedule_updates()

    def _stats_text(self) -> str:
        return f"Trades: {stats.trades} | Wins: {stats.wins} | Losses: {stats.losses} | Draws: {stats.draws} | Win Rate: {stats.win_rate:.2f}%"

    def _schedule_updates(self):
        self.update_stats()
        self.update_chart()
        self.root.after(1000, self._schedule_updates)

    def update_stats(self):
        self.stats_label.config(text=self._stats_text())

    def update_chart(self):
        candles = list(self.trader_loop.cb.candle_data)
        current = self.trader_loop.cb.current_candle
        if current:
            candles.append(current)
        if len(candles) < 1:
            return

        # Price panel
        self.ax_price.clear()
        ohlc = []
        for c in candles[-80:]:
            t = mdates.date2num(c["time"])
            ohlc.append((t, c["open"], c["high"], c["low"], c["close"]))
        candlestick_ohlc(self.ax_price, ohlc, width=1/1440, colorup="green", colordown="red")
        self.ax_price.xaxis_date()
        self.ax_price.set_title("Candles with EMA/Bollinger")

        # Indicators (last values as reference lines)
        i = self.trader_loop.cb.indicators
        ema8 = i.get("ema8"); ema21 = i.get("ema21")
        boll_low = i.get("boll_low"); boll_mid = i.get("boll_mid"); boll_high = i.get("boll_high")

        price_has_labels = False
        if ema8 is not None:
            self.ax_price.axhline(ema8, color="blue", linestyle="--", label="EMA8"); price_has_labels = True
        if ema21 is not None:
            self.ax_price.axhline(ema21, color="purple", linestyle="--", label="EMA21"); price_has_labels = True
        if all(v is not None for v in (boll_low, boll_mid, boll_high)):
            self.ax_price.axhline(boll_low, color="green", alpha=0.6, label="Boll Low"); price_has_labels = True
            self.ax_price.axhline(boll_mid, color="gray", alpha=0.6, label="Boll Mid"); price_has_labels = True
            self.ax_price.axhline(boll_high, color="red", alpha=0.6, label="Boll High"); price_has_labels = True

        if price_has_labels:
            self.ax_price.legend(loc="upper left")

        # RSI panel
        self.ax_rsi.clear()
        rsi_has_labels = False
        rsi = i.get("rsi")
        if rsi is not None and len(candles) > 0:
            self.ax_rsi.plot([candles[-1]["time"]], [rsi], "o", color="teal", label=f"RSI {rsi:.1f}")
            rsi_has_labels = True
        self.ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
        self.ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
        self.ax_rsi.set_ylim(0, 100)
        if rsi_has_labels:
            self.ax_rsi.legend(loc="upper left")
        self.ax_rsi.set_title("RSI")

        self.canvas.draw()

    def start_bot(self):
        if self.loop_thread and self.loop_thread.is_alive():
            return

        def _run():
            asyncio.run(self.trader_loop.start())

        self.loop_thread = threading.Thread(target=_run, daemon=True)
        self.loop_thread.start()
        print("[UI] Bot started")

    def stop_bot(self):
        self.trader_loop.stop()
        print("[UI] Bot stop requested")

# === Entry ===
async def Connect() -> PocketOptionAsync:
    ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
    ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'
    ssid = ssid_demo if MODE == "demo" else ssid_real

    api = PocketOptionAsync(ssid)
    print("[CONNECT] API initializing...")
    await asyncio.sleep(3)
    try:
        balance = await api.balance()
        print(f"[CONNECTED] Balance: {float(balance):.2f} | Stake: {STAKE:.2f}")
    except Exception as e:
        print(f"[BALANCE ERROR] {type(e).__name__}: {e}")
    return api

def main():
    trader_loop = TraderLoop(SYMBOL)
    root = tk.Tk()
    ui = TraderUI(root, trader_loop)
    root.mainloop()

if __name__ == "__main__":
    main()