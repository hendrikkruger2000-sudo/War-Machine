# war_machine_per_candle_fixed_window.py

import sys
import asyncio
import json
import os
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout,
    QComboBox, QGridLayout
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QTimer
from qasync import QEventLoop
import pyqtgraph as pg

# Replace with your actual library import
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                fv = float(v)
                if fv > 0:
                    out.append(datetime.fromtimestamp(fv).strftime("%H:%M"))
                else:
                    out.append("")
            except Exception:
                out.append("")
        return out


class WarMachineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("War Machine ⚔️ — Per-Candle")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color:#0d0d0d; color:#f2f2f2;")

        # Replace with your real SSIDs
        self.ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
        self.ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'
        self.live_trades = {}
        # API and balance
        self.api = None
        self.balance = 0.0
        self.trade_amount = 20.0  # Default until balance is fetched
        self.live_trades = {}
        self.trade_stats = {
            "wins": 0,
            "losses": 0,
            "total": 0,
            "pnl": 0.0
        }
        # Candle state
        self.candle_data = []           # finalized candles
        self.current_candle = None      # forming candle
        self.candle_start = None
        self.symbol = "EURUSD_otc"

        # Stream/task
        self.stream_task = None

        # Persistence
        self.storage_file = "war_machine_candles.json"
        self.max_candles = 200
        self.ensure_storage_file()

        # Trades and stats
        self.active_trade = None              # one live trade at a time
        self.last_trade_candle_time = None    # enforce one trade per candle
        self.stats = {"total": 0, "wins": 0, "losses": 0}
        self.pnl = 0.0
        self.trade_amount = 1.0

        # Strategy
        self.min_warmup_candles = 10
        self.expiry_otc_seconds = 5
        self.expiry_fx_seconds = 60

        # Simulation gate
        self.simulated_trades = []
        self.live_mode = False
        self.sim_backtest_window = 20
        self.sim_required_winrate = 0.60

        # Chart config
        self.window_seconds = 30 * 60    # fixed 30-minute window
        self.x_shift_seconds = 30         # reserve space for markers on right

        # Debug overlay
        self.last_overlay = ""
        self.debug_text_item = pg.TextItem(text="", color='w', anchor=(0, 0))

        # UI
        self.init_ui()
        self.showMaximized()

    # ------------- UI -------------

    def ensure_storage_file(self):
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w") as f:
                json.dump([], f)

    def init_ui(self):
        stat_font = QFont("Consolas", 14)

        self.mode_switch = QComboBox()
        self.mode_switch.addItems(["Demo", "Real"])
        self.mode_switch.setStyleSheet("background-color:#333; color:#fff;")

        self.balance_label = QLabel("Balance: $0.00")
        self.balance_label.setFont(stat_font)

        self.asset_selector = QComboBox()
        self.asset_selector.addItems(["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"])
        self.asset_selector.setCurrentText(self.symbol)
        self.asset_selector.setStyleSheet("background-color:#333; color:#fff;")
        self.asset_selector.currentTextChanged.connect(self.on_asset_change)

        self.stats_label = QLabel("Trades: 0 | Wins: 0 | Losses: 0 | Winrate: 0% | PNL: 0.00")
        self.stats_label.setFont(stat_font)

        self.duration_label = QLabel("Duration: --s")
        self.duration_label.setFont(stat_font)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Mode:"))
        top_layout.addWidget(self.mode_switch)
        top_layout.addSpacing(16)
        top_layout.addWidget(QLabel("Asset:"))
        top_layout.addWidget(self.asset_selector)
        top_layout.addStretch()
        top_layout.addWidget(self.balance_label)

        self.chart_widget = pg.PlotWidget(axisItems={'bottom': TimeAxis(orientation='bottom')})
        self.chart_widget.setBackground('#1a1a1a')
        self.chart_widget.showGrid(x=True, y=True)
        self.chart_widget.setLabel('left', 'Price')
        self.chart_widget.setLabel('bottom', 'Time (HH:MM)')
        self.chart_widget.addItem(self.debug_text_item)

        self.start_button = QPushButton("START WAR MACHINE ⚔️")
        self.stop_button = QPushButton("STOP WAR MACHINE 🛑")
        self.start_button.setStyleSheet("background-color:#2e7d32; color:white; font-weight:bold;")
        self.stop_button.setStyleSheet("background-color:#c62828; color:white; font-weight:bold;")
        self.start_button.clicked.connect(lambda: asyncio.create_task(self.start_war_machine()))
        self.stop_button.clicked.connect(lambda: asyncio.create_task(self.stop_war_machine()))

        main_layout = QGridLayout()
        main_layout.addLayout(top_layout, 0, 0, 1, 2)
        main_layout.addWidget(self.chart_widget, 1, 0, 1, 2)
        main_layout.addWidget(self.stats_label, 2, 0)
        main_layout.addWidget(self.duration_label, 2, 1)
        main_layout.addWidget(self.start_button, 3, 0)
        main_layout.addWidget(self.stop_button, 3, 1)
        self.setLayout(main_layout)

        # Balance refresh timer
        self.balance_timer = QTimer(self)
        self.balance_timer.timeout.connect(lambda: asyncio.create_task(self.update_balance()))
        self.balance_timer.start(5000)

    # ------------- API -------------

    async def connect_to_api(self):
        mode = self.mode_switch.currentText().lower()
        ssid = self.ssid_demo if mode == "demo" else self.ssid_real
        self.api = PocketOptionAsync(ssid)
        await asyncio.sleep(5)  # handshake
        await self.update_balance()

    async def update_balance(self):
        try:
            bal = await self.api.balance()
            self.balance = float(bal)
            self.balance_label.setText(f"Balance: ${self.balance:.2f}")
        except Exception:
            # keep UI stable if balance fetch fails
            pass

    async def update_trade_amount(self):
        try:
            balance = await self.api.get_balance()  # ✅ await if async
            raw_amount = balance * 0.2
            self.trade_amount = max(20, min(raw_amount, 2000))
            print(f"[BALANCE] {balance:.2f} → Trade amount set to {self.trade_amount:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch balance → {type(e).__name__}, {e}")
            self.trade_amount = 20.0

    def on_asset_change(self, asset):
        self.symbol = asset
        self.candle_data.clear()
        self.current_candle = None
        self.candle_start = None
        self.last_trade_candle_time = None
        self.chart_widget.clear()
        self.chart_widget.addItem(self.debug_text_item)
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
        asyncio.create_task(self.restart_stream())

    async def restart_stream(self):
        if not self.api:
            await self.connect_to_api()
        self.stream_task = asyncio.create_task(self.stream_ticks_and_build_candles(self.symbol))

    # --------- Streaming & candles ---------

    async def stream_ticks_and_build_candles(self, symbol):
        stream = await self.api.subscribe_symbol(symbol)
        print(f"Streaming {symbol}...")

        async for tick in stream:
            price = tick.get('close') or tick.get('price') or tick.get('open')
            if price is None:
                continue

            ts_raw = tick.get('timestamp') or tick.get('time') or tick.get('ts')
            ts = float(ts_raw)
            if ts > 1e12:
                ts /= 1000.0
            tick_time = datetime.utcfromtimestamp(ts)
            bucket = tick_time.replace(second=0, microsecond=0)

            if self.candle_start is None:
                self.candle_start = bucket
                self.current_candle = {
                    "time": bucket,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "ticks": [],
                    "entries": [],
                    "exits": []
                }

            elif bucket > self.candle_start:
                print(f"[CANDLE OPEN] {bucket.strftime('%H:%M:%S')}")
                try:
                    self.finalize_current_candle()
                except Exception as e:
                    print(f"[ERROR] finalize_current_candle failed → {type(e).__name__}, {e}")

                self.candle_start = bucket
                self.current_candle = {
                    "time": bucket,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "ticks": [],
                    "entries": [],
                    "exits": []
                }

            # Update forming candle
            self.current_candle["high"] = max(self.current_candle["high"], price)
            self.current_candle["low"] = min(self.current_candle["low"], price)
            self.current_candle["close"] = price
            self.current_candle["ticks"].append((tick_time, price))

            # 🔥 Per-tick micro momentum
            await self.evaluate_micro_momentum_tick(price, tick_time)

            # Lifecycle
            await self.check_simulated_exits(tick_time)
            await self.check_trade_exit_live(tick_time)

            # Overlay
            self.update_debug_overlay()
            self.draw_chart(forming=True)


    def finalize_current_candle(self):
        if not self.current_candle:
            return
        candle_copy = self.current_candle.copy()
        self.candle_data.append(candle_copy)
        if len(self.candle_data) > self.max_candles:
            self.candle_data.pop(0)

        # persist (optional; keeps history across sessions)
        try:
            with open(self.storage_file, "r+") as f:
                data = json.load(f)
                data.append({
                    "time": candle_copy["time"].isoformat(),
                    "open": candle_copy["open"],
                    "high": candle_copy["high"],
                    "low": candle_copy["low"],
                    "close": candle_copy["close"],
                    "ticks": candle_copy["ticks"],
                    "entries": candle_copy.get("entries", []),
                    "exits": candle_copy.get("exits", [])
                })
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception:
            pass

    # ------------- Strategy -------------

    def ema(self, series, period):
        if len(series) < period:
            return None
        k = 2 / (period + 1)
        ema_val = series[-period]
        for v in series[-period + 1:]:
            ema_val = v * k + ema_val * (1 - k)
        return ema_val

    def decide_direction_per_candle(self):
        # original logic: EMA20 filter, one trade per candle at open
        if len(self.candle_data) < self.min_warmup_candles:
            return None, "warm-up"
        closes = [c["close"] for c in self.candle_data]
        ema20 = self.ema(closes, 20)
        if ema20 is None:
            return None, "ema20-unavailable"

        price_open = self.current_candle["open"]
        trend_up = price_open >= ema20

        prev = self.candle_data[-1] if self.candle_data else None
        prev_color = None
        if prev:
            prev_color = "green" if prev["close"] >= prev["open"] else "red"

        if trend_up:
            return "buy", f"EMA20 UP (open {price_open:.5f} ≥ {ema20:.5f}); prev {prev_color}"
        else:
            return "sell", f"EMA20 DOWN (open {price_open:.5f} < {ema20:.5f}); prev {prev_color}"

    async def evaluate_micro_momentum_tick(self, price, tick_time):
        micro_dir, micro_conf = self.detect_micro_momentum(tick_window=10)
        if not micro_dir or micro_conf < 0.6:
            return

        # 🔍 Candle alignment filter
        is_bullish = self.current_candle["close"] > self.current_candle["open"]
        is_bearish = self.current_candle["close"] < self.current_candle["open"]

        # 🔥 Slope filter — INSERT HERE
        candle_slope = self.current_candle["close"] - self.current_candle["open"]
        if abs(candle_slope) < 0.0001:
            return  # Skip flat candles

        if micro_dir == "call" and not is_bullish:
            return
        if micro_dir == "put" and not is_bearish:
            return

        print(
            f"[TRADE DECISION] {tick_time.strftime('%H:%M:%S')} | dir={micro_dir.upper()} | conf={micro_conf:.2f} | candle={'BULL' if is_bullish else 'BEAR'}")

        self.simulate_trade(micro_dir, price, tick_time, duration=5)
        await self.place_live_trade(micro_dir, price, tick_time, duration=5)
        self.last_overlay = f"MICRO-{micro_dir.upper()} | conf:{int(micro_conf * 100)}%"

    def detect_micro_momentum(self, tick_window=10):
        if not self.current_candle or len(self.current_candle["ticks"]) < tick_window:
            return None, 0.0

        recent = self.current_candle["ticks"][-tick_window:]
        prices = [p for (_, p) in recent]
        tick_times = [ts for (ts, _) in recent]

        delta = prices[-1] - prices[0]
        if abs(delta) < 0.0002:
            return None, 0.0  # 🔥 Skip weak moves

        # Tick velocity filter
        intervals = [(tick_times[i + 1] - tick_times[i]).total_seconds() for i in range(len(tick_times) - 1)]
        avg_interval = sum(intervals) / len(intervals)
        if avg_interval > 1.5:
            return None, 0.0  # 🔥 Skip slow ticks

        direction = "call" if delta > 0 else "put" if delta < 0 else None
        moves = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        bias_count = sum(1 for m in moves if (m > 0 and direction == "call") or (m < 0 and direction == "put"))
        confidence = bias_count / len(moves) if direction else 0.0

        return direction, confidence


    def update_duration_label(self, duration):
        if duration is None:
            self.duration_label.setText("Duration: --s")
            self.duration_label.setStyleSheet("color: yellow;")
            return
        self.duration_label.setText(f"Duration: {duration}s")
        self.duration_label.setStyleSheet("color: white;" if self.live_mode else "color: yellow;")

    # ------------- Simulation -------------

    def simulate_trade(self, direction, price, tick_time, duration):
        expiry = tick_time + timedelta(seconds=duration)
        sim = {
            "time": tick_time, "direction": direction,
            "entry_price": price, "exit_time": expiry,
            "exit_price": None, "result": None
        }
        self.simulated_trades.append(sim)
        # mark sim entry
        self.current_candle.setdefault("entries", []).append({
            "direction": direction, "price": price, "sim": True
        })
        print(f"[SIM] {direction.upper()} at {price:.5f} | dur {duration}s | {tick_time.strftime('%H:%M:%S')}")

    async def check_simulated_exits(self, tick_time):
        for sim in self.simulated_trades:
            if sim["result"] is None and tick_time >= sim["exit_time"]:
                exit_price = self.current_candle["close"]
                sim["exit_price"] = exit_price
                if sim["direction"] == "buy":
                    sim["result"] = "win" if exit_price >= sim["entry_price"] else "loss"
                else:
                    sim["result"] = "win" if exit_price <= sim["entry_price"] else "loss"
                # mark sim exit
                self.current_candle.setdefault("exits", []).append({
                    "price": exit_price, "result": sim["result"], "sim": True
                })
                print(f"[SIM RESULT] {sim['result'].upper()} at {exit_price:.5f} | {tick_time.strftime('%H:%M:%S')}")

        # update mode from sim performance
        sim_wr = self.simulation_winrate()
        prev_mode = self.live_mode
        self.live_mode = (sim_wr >= self.sim_required_winrate)
        if self.live_mode != prev_mode:
            print(f"[GATE] {'LIVE ENABLED' if self.live_mode else 'LIVE DISABLED'} | SimWR {sim_wr:.2%}")

    def simulation_winrate(self):
        recent = [s for s in self.simulated_trades if s["result"] is not None][-self.sim_backtest_window:]
        if not recent:
            return 0.0
        wins = sum(1 for s in recent if s["result"] == "win")
        return wins / len(recent)

    # ------------- Live trading -------------

    async def place_live_trade(self, direction, price, tick_time, duration=5):
        await self.update_trade_amount()
        amount = self.trade_amount

        try:
            if direction == "call":
                trade_id, _ = await self.api.buy(
                    asset=self.symbol,
                    amount=amount,
                    time=duration,
                    check_win=False
                )
            elif direction == "put":
                trade_id, _ = await self.api.sell(
                    asset=self.symbol,
                    amount=amount,
                    time=duration,
                    check_win=False
                )
            else:
                print(f"[ERROR] Invalid direction: {direction}")
                return

            print(
                f"[LIVE] {direction.upper()} {trade_id} at {price:.5f} | dur {duration}s | amt {amount:.2f} | {tick_time.strftime('%H:%M:%S')}")
            self.live_trades[trade_id] = {
                "direction": direction,
                "entry_price": price,
                "entry_time": tick_time,
                "duration": duration
            }

        except Exception as e:
            print(f"[ERROR] Live trade failed → {type(e).__name__}, {e}")

    async def check_trade_exit_live(self, tick_time):
        expired = []
        for trade_id, trade in list(self.live_trades.items()):
            entry_time = trade["entry_time"]
            duration = trade["duration"]
            expiry_time = entry_time + timedelta(seconds=duration)

            if tick_time >= expiry_time:
                direction = trade["direction"]
                entry_price = trade["entry_price"]
                current_price = self.current_candle["close"]

                win = (direction == "call" and current_price > entry_price) or \
                      (direction == "put" and current_price < entry_price)

                self.trade_stats["total"] += 1
                if win:
                    self.trade_stats["wins"] += 1
                    self.trade_stats["pnl"] += self.trade_amount
                else:
                    self.trade_stats["losses"] += 1
                    self.trade_stats["pnl"] -= self.trade_amount
                self.update_stats_label()
                print(f"[RESULT] {direction.upper()} → {'WIN' if win else 'LOSS'} | PnL: {self.trade_stats['pnl']:.2f}")
                expired.append(trade_id)

        for trade_id in expired:
            del self.live_trades[trade_id]

    def update_stats_label(self):
        stats = self.trade_stats
        total = stats["total"]
        wins = stats["wins"]
        losses = stats["losses"]
        winrate = (wins / total * 100) if total > 0 else 0.0
        pnl = stats["pnl"]

        self.stats_label.setText(
            f"Trades: {total} | Wins: {wins} | Losses: {losses} | Winrate: {winrate:.1f}% | PNL: {pnl:.2f}"
        )

    # ------------- Overlay & drawing -------------

    def update_debug_overlay(self):
        sim_wr = self.simulation_winrate()
        mode = "LIVE" if self.live_mode else "SIM"
        text = f"{mode} | {self.last_overlay}" if self.last_overlay else f"{mode} | SimWR:{int(sim_wr*100)}%"
        self.debug_text_item.setText(text)

    def draw_chart(self, forming=False):
        candles = list(self.candle_data)
        if forming and self.current_candle:
            candles.append(self.current_candle.copy())
        if not candles:
            return

        self.chart_widget.clear()
        lows, highs = [], []

        # ✅ Fixed 30‑minute window with reserved marker space
        last_ts = candles[-1]['time'].timestamp()
        x_max = last_ts + self.x_shift_seconds
        x_min = x_max - self.window_seconds

        for c in candles:
            t = c['time'].timestamp()
            o, h, l, cl = c['open'], c['high'], c['low'], c['close']
            lows.append(l);
            highs.append(h)
            color = 'g' if cl >= o else 'r'
            self.chart_widget.plot([t, t], [l, h], pen=color)
            self.chart_widget.plot([t, t], [o, cl], pen=pg.mkPen(color, width=12))

            # ✅ Clamp entry/exit markers inside window
            for e in c.get("entries", []):
                pos_x = min(max(t + self.x_shift_seconds, x_min + 1), x_max - 1)
                arrow_angle = 90 if e["direction"] == "buy" else -90
                arrow_color = 'y' if e.get("sim") else ('g' if e["direction"] == "buy" else 'r')
                entry_arrow = pg.ArrowItem(
                    pos=(pos_x, e["price"]),
                    angle=arrow_angle,
                    brush=arrow_color, pen=pg.mkPen(arrow_color),
                    headLen=12, tipAngle=30, baseAngle=20
                )
                self.chart_widget.addItem(entry_arrow)

            for ex in c.get("exits", []):
                pos_x = min(max(t + self.x_shift_seconds, x_min + 1), x_max - 1)
                label_color = 'y' if ex.get("sim") else 'w'
                exit_marker = pg.ScatterPlotItem(
                    x=[pos_x], y=[ex["price"]],
                    pen=pg.mkPen(label_color), brush=label_color,
                    size=(6 if ex.get("sim") else 8), symbol='o'
                )
                self.chart_widget.addItem(exit_marker)

        # ✅ Y‑axis and X‑axis ranges
        y_min, y_max = min(lows), max(highs)
        pad = (y_max - y_min) * 0.03 if y_max > y_min else 0.001
        self.chart_widget.setYRange(y_min - pad, y_max + pad)
        self.chart_widget.setXRange(x_min, x_max, padding=0)

        self.chart_widget.addItem(self.debug_text_item)
        self.debug_text_item.setPos(x_min, y_max)

    # ------------- Start/Stop -------------

    async def start_war_machine(self):
        await self.connect_to_api()
        self.stream_task = asyncio.create_task(self.stream_ticks_and_build_candles(self.symbol))
        print("⚔️ War Machine activated: per-candle, simulation-first, fixed 30m window.")

    async def stop_war_machine(self):
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        print("🛑 War Machine deactivated.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    gui = WarMachineGUI()
    gui.show()

    with loop:
        loop.run_forever()