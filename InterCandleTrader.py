# InterCandleTrader.py
import asyncio, inspect, threading, math
from datetime import datetime
from typing import Optional, Dict, Any, List
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

# === Config ===
SYMBOL = "EURUSD_otc"
STAKE = 20.0
MODE = "demo"  # or "real"

# === Stats ===
class Stats:
    def __init__(self):
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.pnl = 0.0

    @property
    def win_rate(self):
        return (self.wins / self.trades) * 100 if self.trades > 0 else 0.0

stats = Stats()

# === Helpers ===
async def _maybe_await(maybe):
    return await maybe if inspect.isawaitable(maybe) else maybe

# === Candle Builder ===
class TickBuffer:
    def __init__(self, max_ticks=300):
        self.ticks: List[Tuple[datetime, float]] = []
        self.indicators: Dict[str, Any] = {}
        self.max_ticks = max_ticks

    def add_tick(self, tick_time: datetime, price: float):
        self.ticks.append((tick_time, price))
        if len(self.ticks) > self.max_ticks:
            self.ticks.pop(0)
        self.update_indicators()

    def update_indicators(self):
        if len(self.ticks) < 30:
            return
        prices = [p for _, p in self.ticks]
        ema8 = _ema(prices, 8)
        ema21 = _ema(prices, 21)
        rsi = _rsi(prices, 14)
        boll = _bollinger(prices, 20, 2.0)
        mom = _roc(prices, 6)
        self.indicators.update({
            "ema8": ema8, "ema21": ema21, "rsi": rsi,
            "boll_low": boll[0], "boll_mid": boll[1], "boll_high": boll[2],
            "boll_percent_b": boll[4], "momentum": mom
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
    std = (sum((x-mean)**2 for x in window)/period)**0.5
    upper, lower = mean+k*std, mean-k*std
    percent_b = (window[-1]-lower)/(upper-lower) if upper!=lower else 0.5
    return lower, mean, upper, None, percent_b, None

def _roc(prices, period=6):
    if len(prices) < period+1: return None
    prev = prices[-1-period]
    if prev == 0: return 0.0
    return (prices[-1]-prev)/prev

# === Strategy ===
def TickStrategy(tb: TickBuffer) -> Optional[str]:
    i = tb.indicators
    if not all(i.get(k) is not None for k in ["ema21","ema48","rsi","boll_percent_b","momentum"]):
        return None

    score = 0.0
    direction = None

    # EMA crossover
    if i["ema21"] > i["ema48"]:
        direction = "call"
        score += 0.3
    elif i["ema21"] < i["ema48"]:
        direction = "put"
        score += 0.3

    # Bollinger %B
    if direction == "call" and i["boll_percent_b"] < 0.2:
        score += 0.2
    elif direction == "put" and i["boll_percent_b"] > 0.8:
        score += 0.2

    # RSI
    if direction == "call" and i["rsi"] < 30:
        score += 0.2
    elif direction == "put" and i["rsi"] > 70:
        score += 0.2

    # Momentum
    if direction == "call" and i["momentum"] > 0:
        score += 0.2
    elif direction == "put" and i["momentum"] < 0:
        score += 0.2

    # Optional: Choppiness filter
    if i.get("choppiness") is not None and (i["choppiness"] < 40 or i["choppiness"] > 60):
        score += 0.1

    # Final decision
    if score >= 0.8:
        return direction
    return None

# === Trader ===
class InterCandleTrader:
    def __init__(self, symbol):
        self.symbol = symbol
        self.api: Optional[PocketOptionAsync] = None
        self.tb = TickBuffer()
        self._stop = asyncio.Event()
        self.last_trade_time: Optional[datetime] = None

    async def start(self):
        self.api = await Connect()
        sub = await _maybe_await(self.api.subscribe_symbol(self.symbol))
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
        ts_raw = tick.get("timestamp") or tick.get("time") or tick.get("ts")
        try:
            ts = float(ts_raw)
            if ts > 1e12: ts /= 1000.0
            tick_time = datetime.utcfromtimestamp(ts)
        except:
            tick_time = datetime.utcnow()

        self.tb.add_tick(tick_time, float(price))

        direction = TickStrategy(self.tb)
        if direction and self._can_trade(tick_time):
            await self._take_trade(direction, tick_time)

    def _can_trade(self, tick_time: datetime) -> bool:
        if not self.last_trade_time:
            return True
        delta = (tick_time - self.last_trade_time).total_seconds()
        return delta >= 5

    async def _take_trade(self, direction: str, tick_time: datetime):
        try:
            if direction == "call":
                res = await _maybe_await(self.api.buy(self.symbol, STAKE, 5, check_win=False))
            else:
                res = await _maybe_await(self.api.sell(self.symbol, STAKE, 5, check_win=False))
            trade_id = res[0] if isinstance(res, (tuple, list)) else res
            self.last_trade_time = tick_time
            asyncio.create_task(self._resolve_trade(trade_id, direction, tick_time))
        except Exception as e:
            print(f"[ERROR] Trade failed: {e}")

    async def _resolve_trade(self, trade_id: Any, direction: str, tick_time: datetime):
        try:
            result_data = await _maybe_await(self.api.check_win(trade_id))
            result = (result_data or {}).get("result", "UNKNOWN")
            exit_price = (result_data or {}).get("closePrice", 0.0)

            stats.trades += 1
            if result == "win":
                stats.wins += 1
                stats.pnl += STAKE * 0.8
            elif result == "loss":
                stats.losses += 1
                stats.pnl -= STAKE
            else:
                stats.draws += 1

            print(f"[TRADE] {tick_time.isoformat()} | {direction.upper()} | {result.upper()} | Exit: {exit_price:.5f} | PNL: {stats.pnl:.2f}")
        except Exception as e:
            print(f"[ERROR] Trade result check failed: {e}")

    def stop(self):
        self._stop.set()

# === Connection ===
# === Connection ===
async def Connect() -> PocketOptionAsync:
    ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
    ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'


    ssid = ssid_demo if MODE == "demo" else ssid_real

    api = PocketOptionAsync(ssid)
    print("[CONNECT] Connecting to PocketOption...")
    await asyncio.sleep(2)
    try:
        balance = await api.balance()
        print(f"[CONNECTED] Balance: {float(balance):.2f}")
    except Exception as e:
        print(f"[ERROR] Could not fetch balance: {e}")
    return api

# === GUI ===
class TraderUI:
    def __init__(self, root: tk.Tk, trader: InterCandleTrader):
        self.root = root
        self.trader = trader
        self.loop_thread: Optional[threading.Thread] = None

        self.root.title("InterCandle Trader")
        self.root.geometry("1000x700")

        self.stats_label = tk.Label(root, text=self._stats_text(), font=("Consolas", 12))
        self.stats_label.pack(pady=8)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.start_btn = tk.Button(btn_frame, text="Start Bot", command=self.start_bot, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=8)
        self.stop_btn = tk.Button(btn_frame, text="Stop Bot", command=self.stop_bot, width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._schedule_updates()

    def _stats_text(self) -> str:
        return f"Trades: {stats.trades} | Wins: {stats.wins} | Losses: {stats.losses} | Draws: {stats.draws} | Win Rate: {stats.win_rate:.2f}% | PNL: {stats.pnl:.2f}"

    def _schedule_updates(self):
        self.update_stats()
        self.update_chart()
        self.root.after(1000, self._schedule_updates)

    def update_stats(self):
        self.stats_label.config(text=self._stats_text())

    def update_chart(self):
        ticks = self.trader.tb.ticks
        if len(ticks) < 2:
            return

        self.ax.clear()
        times = [mdates.date2num(t[0]) for t in ticks]
        prices = [t[1] for t in ticks]
        self.ax.plot_date(times, prices, "-", label="Price", color="black", linewidth=1)

        i = self.trader.tb.indicators

        # EMA overlays
        if i.get("ema21"):
            self.ax.axhline(i["ema21"], color="green", linestyle="--", label="EMA21")
        if i.get("ema48"):
            self.ax.axhline(i["ema48"], color="orange", linestyle="--", label="EMA48")

        # Bollinger bands
        if i.get("boll_low") and i.get("boll_high"):
            self.ax.axhline(i["boll_low"], color="blue", linestyle="--", alpha=0.5, label="Boll Low")
            self.ax.axhline(i["boll_high"], color="red", linestyle="--", alpha=0.5, label="Boll High")

        # RSI label
        rsi = i.get("rsi")
        if rsi is not None:
            self.ax.text(times[-1], prices[-1] + 0.00002, f"RSI: {rsi:.1f}", fontsize=10, color="darkviolet",
                         ha="right")

        # Momentum label
        momentum = i.get("momentum")
        if momentum is not None:
            self.ax.text(times[-1], prices[-1] - 0.00002, f"Momentum: {momentum:.5f}", fontsize=10, color="darkgreen",
                         ha="right")

        # Bollinger %B label
        percent_b = i.get("boll_percent_b")
        if percent_b is not None:
            self.ax.text(times[-1], prices[-1] - 0.00004, f"%B: {percent_b:.2f}", fontsize=10, color="darkred",
                         ha="right")

        # Score and direction overlay
        direction = TickStrategy(self.trader.tb)
        score = 0.0

        if direction:
            if i["ema21"] > i["ema48"] and direction == "call":
                score += 0.3
            elif i["ema21"] < i["ema48"] and direction == "put":
                score += 0.3
            if direction == "call" and i["boll_percent_b"] < 0.2:
                score += 0.2
            elif direction == "put" and i["boll_percent_b"] > 0.8:
                score += 0.2
            if direction == "call" and i["rsi"] < 30:
                score += 0.2
            elif direction == "put" and i["rsi"] > 70:
                score += 0.2
            if direction == "call" and i["momentum"] > 0:
                score += 0.2
            elif direction == "put" and i["momentum"] < 0:
                score += 0.2
            if i.get("choppiness") is not None and (i["choppiness"] < 40 or i["choppiness"] > 60):
                score += 0.1

            self.ax.text(times[-1], prices[-1] + 0.00006,
                         f"Direction: {direction.upper()} | Score: {score:.2f}",
                         fontsize=10, color="darkorange", ha="right")

        # Final touches
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="upper left")
        self.ax.set_title("Live Tick Chart")
        self.canvas.draw()

    def start_bot(self):
        if self.loop_thread and self.loop_thread.is_alive():
            return
        def _run():
            asyncio.run(self.trader.start())
        self.loop_thread = threading.Thread(target=_run, daemon=True)
        self.loop_thread.start()
        print("[UI] Bot started")

    def stop_bot(self):
        self.trader.stop()
        print("[UI] Bot stop requested")

# === Entry Point ===
def main():
    trader = InterCandleTrader(SYMBOL)
    root = tk.Tk()
    ui = TraderUI(root, trader)
    root.mainloop()

if __name__ == "__main__":
    main()