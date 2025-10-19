# Revised_WarMachine_GUI_Script.py
# Stripped chart version: stats, predictions, accuracy, tick stream and candle builder
import sys
import os
import json
import asyncio
import numpy as np
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout,
    QComboBox, QGridLayout, QVBoxLayout, QTextEdit
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QTimer
from qasync import QEventLoop

# Replace with your actual async API client
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync


class WarMachineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("War Machine ⚔️ — Dashboard")
        self.setGeometry(100, 100, 900, 420)
        self.setStyleSheet("background-color:#0d0d0d; color:#f2f2f2;")
        self.font_mono = QFont("Consolas", 11)

        # Credentials (replace with your strings)
        self.ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
        self.ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'

        # State
        self.api = None
        self.balance = 0.0
        self.trade_amount = 1.0
        self.symbol = "EURUSD_otc"
        self.trade_stats = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}

        # Candle / stream state
        self.candle_data = []
        self.current_candle = None
        self.candle_duration = 60  # seconds
        self.last_price = None
        self.ghost_history = []
        self.slope_history = []
        self.last_trade_time = None
        self.last_direction = None
        self.stream_task = None
        self.storage_file = "war_machine_candles.json"
        self.max_candles = 200
        self.ensure_storage_file()

        # Simulation / live gate
        self.simulated_trades = []
        self.live_trades_history = {}
        self.sim_backtest_window = 5
        self.sim_required_winrate = 0.60
        self.live_mode = False
        self.base_stake_ratio = 0.02  # 2% of balance
        self.max_stake = 2000.0

        # UI elements
        self.init_ui()
        self.show()

    def ensure_storage_file(self):
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w") as f:
                json.dump([], f)

    def init_ui(self):
        stat_font = QFont("Consolas", 12)

        # Controls
        self.mode_switch = QComboBox()
        self.mode_switch.addItems(["Demo", "Real"])
        self.mode_switch.setStyleSheet("background-color:#333; color:#fff;")

        self.asset_selector = QComboBox()
        self.asset_selector.addItems(["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"])
        self.asset_selector.setCurrentText(self.symbol)
        self.asset_selector.setStyleSheet("background-color:#333; color:#fff;")
        self.asset_selector.currentTextChanged.connect(self.on_asset_change)

        self.start_button = QPushButton("START WAR MACHINE ⚔️")
        self.stop_button = QPushButton("STOP WAR MACHINE 🛑")
        self.start_button.setStyleSheet("background-color:#2e7d32; color:white; font-weight:bold;")
        self.stop_button.setStyleSheet("background-color:#c62828; color:white; font-weight:bold;")
        self.start_button.clicked.connect(lambda: asyncio.create_task(self.start_war_machine()))
        self.stop_button.clicked.connect(lambda: asyncio.create_task(self.stop_war_machine()))

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Mode:"))
        top_layout.addWidget(self.mode_switch)
        top_layout.addSpacing(12)
        top_layout.addWidget(QLabel("Asset:"))
        top_layout.addWidget(self.asset_selector)
        top_layout.addStretch()
        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.stop_button)

        # Stats panel
        self.stats_label = QLabel("Live Trades: 0 | Wins: 0 | Losses: 0 | Winrate: 0.0% | PNL: 0.00")
        self.stats_label.setFont(stat_font)
        self.sim_stats_label = QLabel("Sim Trades: 0 | Wins: 0 | Losses: 0 | Winrate: 0.0%")
        self.sim_stats_label.setFont(stat_font)
        self.balance_label = QLabel("Balance: $0.00")
        self.balance_label.setFont(stat_font)
        self.indicators = {
            "ema8": None,
            "ema21": None,
            "momentum": None,
            "rsi": None,
            "choppiness": None,
            "boll_low": None,
            "boll_mid": None,
            "boll_high": None
        }
        self.last_tick_time = datetime.utcnow()
        self.loss_cluster_limit = 3

        self.tick_watchdog_timer = QTimer(self)
        self.tick_watchdog_timer.timeout.connect(self.check_tick_heartbeat)
        self.tick_watchdog_timer.start(30000)
        # Prediction panel
        self.prediction_label = QLabel("Prediction: N/A")
        self.prediction_label.setFont(stat_font)
        self.prediction_label.setStyleSheet("color: #ffd54f;")

        self.diagnostics = QTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setFont(self.font_mono)
        self.diagnostics.setStyleSheet("background-color:#121212; color:#e0e0e0;")

        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.balance_label)
        stats_layout.addWidget(self.stats_label)  # Live stats
        stats_layout.addWidget(self.sim_stats_label)  # Sim stats
        stats_layout.addWidget(self.prediction_label)

        main_layout = QGridLayout()
        main_layout.addLayout(top_layout, 0, 0, 1, 2)
        main_layout.addLayout(stats_layout, 1, 0)
        main_layout.addWidget(self.diagnostics, 1, 1)
        self.setLayout(main_layout)

        # Balance refresh
        self.balance_timer = QTimer(self)
        self.balance_timer.timeout.connect(lambda: asyncio.create_task(self.update_balance()))
        self.balance_timer.start(5000)

        # Simulated trade resolution failsafe
        self.sim_exit_timer = QTimer(self)
        self.sim_exit_timer.timeout.connect(lambda: asyncio.create_task(self.check_simulated_exits(datetime.utcnow())))
        self.sim_exit_timer.start(1000)

    # ---------------- API / Balance ----------------
    async def connect_to_api(self):
        mode = self.mode_switch.currentText().lower()
        ssid = self.ssid_demo if mode == "demo" else self.ssid_real
        self.api = PocketOptionAsync(ssid)
        await asyncio.sleep(3)
        await self.update_balance()

    async def update_balance(self):
        try:
            bal = await self.api.balance()
            self.balance = float(bal)
            self.balance_label.setText(f"Balance: ${self.balance:.2f}")
        except Exception:
            pass

    async def update_trade_amount(self, confidence=0.0):
        try:
            balance = await self.api.balance()
            base = balance * self.base_stake_ratio
            scaled = base * (1.0 + (confidence - 0.5) * 1.5)  # Boost up to +75% if confidence is 1.0
            self.trade_amount = max(1.0, min(scaled, self.max_stake))
        except Exception:
            self.trade_amount = 20.0

    def check_tick_heartbeat(self):
        now = datetime.utcnow()
        if (now - self.last_tick_time).total_seconds() > 60:
            self.append_diag("[ERROR] Tick stream stalled — restarting War Machine")
            asyncio.create_task(self.restart_war_machine())

    async def restart_war_machine(self):
        await self.stop_war_machine()
        await asyncio.sleep(2)
        await self.start_war_machine()

    # ---------------- Asset / Stream control ----------------
    def on_asset_change(self, asset):
        self.symbol = asset
        self.candle_data.clear()
        self.current_candle = None
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
        asyncio.create_task(self.start_war_machine())

    # ---------------- Streaming & candles ----------------
    async def stream_ticks_and_build_candles(self, symbol):
        stream = await self.api.subscribe_symbol(symbol)
        self.append_diag(f"Streaming {symbol}...")

        self.candle_start = None
        self.current_candle = None

        try:
            async for tick in stream:
                # --- Parse tick ---
                price = tick.get("close") or tick.get("price") or tick.get("open")
                if price is None:
                    continue

                ts_raw = tick.get("timestamp") or tick.get("time") or tick.get("ts")
                try:
                    ts = float(ts_raw)
                    if ts > 1e12:  # ms → s
                        ts /= 1000.0
                    tick_time = datetime.utcfromtimestamp(ts)
                except Exception:
                    tick_time = datetime.utcnow()

                # --- Candle bucketing ---
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
                    try:
                        self.finalize_current_candle()
                    except Exception as e:
                        self.append_diag(f"[ERROR] finalize_current_candle failed → {type(e).__name__}, {e}")
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

                # --- Update forming candle ---
                self.current_candle["high"] = max(self.current_candle["high"], price)
                self.current_candle["low"] = min(self.current_candle["low"], price)
                self.current_candle["close"] = price
                self.current_candle["ticks"].append((tick_time, price))

                # --- Per-tick handler ---
                try:
                    self.last_price = price
                    self.update_tick_indicators()
                    await self.evaluate_combined_trade(price, tick_time)
                    if tick_time.second != getattr(self, "last_ghost_check", -1):
                        self.last_ghost_check = tick_time.second
                        await self.check_ghost_exits()
                    await self.check_simulated_exits(tick_time)
                    await self.check_trade_exit_live(tick_time)
                except Exception as e:
                    self.append_diag(f"[HANDLE TICK ERROR] {e}")

                # --- Overlay update ---

        except asyncio.CancelledError:
            self.append_diag("[STREAM] subscription cancelled")
            raise
        except Exception as e:
            self.append_diag(f"[STREAM ITERATION ERROR] {type(e).__name__}: {e}")

    def update_tick_indicators(self, tick_window=8):
        try:
            if not self.current_candle or "ticks" not in self.current_candle:
                return
            ticks = self.current_candle["ticks"]
            if len(ticks) < max(21, tick_window):
                return

            prices = [p for (_, p) in ticks]

            self.indicators["ema8"] = self.ema(prices, period=8)
            self.indicators["ema21"] = self.ema(prices, period=21)
            self.indicators["rsi"] = self.rsi(prices, period=14)
            self.indicators["choppiness"] = self.choppiness_index(prices, period=14)
            low, mid, high = self.bollinger_bands(prices, period=20, k=2.0)
            self.indicators["boll_low"] = low
            self.indicators["boll_mid"] = mid
            self.indicators["boll_high"] = high

            momentum_dir, momentum_conf = self.detect_micro_momentum(tick_window=6)
            self.indicators["momentum"] = momentum_conf if momentum_dir else 0.0

        except Exception as e:
            self.append_diag(f"[INDICATORS ERROR] {e}")

    def finalize_current_candle(self):
        if self.current_candle:
            self.candle_data.append(self.current_candle)
            if len(self.candle_data) > self.max_candles:
                self.candle_data.pop(0)

    # ---------------- Utilities / UI updates ----------------
    def append_diag(self, text):
        important = any(tag in text for tag in [
            "⚔️", "🛑", "[LIVE]", "[RESULT]", "[SIM]", "[SIM RESULT]", "[SKIP]", "[OVERLAY]", "[ERROR]"
        ])
        if not important:
            return
        now = datetime.utcnow().strftime("%H:%M:%S")
        self.diagnostics.append(f"[{now}] {text}")
        if self.diagnostics.toPlainText().count("\n") > 500:
            self.diagnostics.clear()


    async def check_ghost_exits(self, _tick_time=None):
        tick_time = datetime.utcnow()
        self.last_price = self.current_candle["close"] if self.current_candle else self.last_price
        self.append_diag(f"[DEBUG] GHOST exit check | TickTime:{tick_time} | LastPrice:{self.last_price}")

        for ghost in list(self.ghost_history):
            expiry = ghost["expiry"]
            if ghost["result"] is None and tick_time >= expiry:
                exit_price = self.last_price if self.last_price is not None else ghost["entry_price"]
                ghost["exit_price"] = exit_price
                if ghost["direction"] in ("call", "buy"):
                    ghost["result"] = "win" if exit_price >= ghost["entry_price"] else "loss"
                else:
                    ghost["result"] = "win" if exit_price <= ghost["entry_price"] else "loss"

                duration = int((expiry - ghost["time"]).total_seconds())
                self.append_diag(
                    f"[GHOST RESULT] {ghost['result'].upper()} | Dir:{ghost['direction']} | "
                    f"Entry:{ghost['entry_price']:.5f} → Exit:{ghost['exit_price']:.5f} | "
                    f"Dur:{duration}s | Pstay:{ghost.get('p_stay', 0.0):.2%}"
                )

    async def evaluate_minimalist_trade(self, price, tick_time):
        # --- Cooldown gate ---
        cooldown_seconds = 10
        if self.last_trade_time and (tick_time - self.last_trade_time).total_seconds() < cooldown_seconds:
            return

        # --- Tick context ---
        candle = self.current_candle
        ticks = candle.get("ticks", []) if candle else []
        prices = [p for (_, p) in ticks]
        if len(prices) < 21:
            return

        # --- Trend detection ---
        ema_fast = self.ema(prices, period=8)
        ema_slow = self.ema(prices, period=21)
        if ema_fast is None or ema_slow is None:
            return

        trend = "call" if ema_fast > ema_slow else "put"
        pullback_ok = abs(ema_fast - price) < 0.00005

        # --- Momentum confirmation ---
        momentum_dir, momentum_conf = self.detect_micro_momentum(tick_window=6)
        if momentum_dir != trend or momentum_conf < 0.6:
            return

        # --- Fire trade ---
        if pullback_ok:
            self.simulate_trade(trend, price, tick_time, duration=5)
            self.last_trade_time = tick_time
            self.append_diag(
                f"[SIM] {trend.upper()} @ {price:.5f} | EMA8:{ema_fast:.5f} | EMA21:{ema_slow:.5f} | Momentum:{momentum_conf:.2f}"
            )

    def ema(self, prices, period=8):
        if len(prices) < period:
            return None
        k = 2 / (period + 1)
        ema_val = prices[0]
        for p in prices[1:]:
            ema_val = p * k + ema_val * (1 - k)
        return ema_val

    def rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, period + 1):
            delta = prices[-i] - prices[-i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def bollinger_bands(self, prices, period=20, k=2.0):
        if len(prices) < period:
            return None, None, None
        window = prices[-period:]
        ma = float(np.mean(window))
        std = float(np.std(window))
        return ma - k * std, ma, ma + k * std

    def choppiness_index(self, prices, period=14):
        if len(prices) < period + 1:
            return None
        window = prices[-period:]
        high, low = max(window), min(window)
        sum_tr = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
        if high == low or sum_tr <= 1e-12:
            return 100.0
        return 100.0 * np.log10(sum_tr / (high - low)) / np.log10(period)


    def detect_micro_momentum(self, tick_window=10):
        """Detect short-term momentum direction and confidence."""
        if not self.current_candle or len(self.current_candle.get("ticks", [])) < tick_window:
            return None, 0.0
        ticks = self.current_candle["ticks"][-tick_window:]
        prices = [p for (_, p) in ticks]
        if len(prices) < 2:
            return None, 0.0

        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
        down_moves = tick_window - 1 - up_moves
        if up_moves > down_moves:
            confidence = up_moves / (tick_window - 1)
            return "call", confidence
        elif down_moves > up_moves:
            confidence = down_moves / (tick_window - 1)
            return "put", confidence
        return None, 0.0


    def kalman_smooth(self, prices):
        n = len(prices)
        x = np.zeros(n)
        P = np.zeros(n)
        x[0] = prices[0]
        P[0] = 1.0
        Q = 0.00001
        R = 0.0001

        for t in range(1, n):
            x_pred = x[t - 1]
            P_pred = P[t - 1] + Q
            K = P_pred / (P_pred + R)
            x[t] = x_pred + K * (prices[t] - x_pred)
            P[t] = (1 - K) * P_pred

        return x.tolist()

    def compute_volatility(self, prices):
        if len(prices) < 2:
            return 0.0
        diffs = np.diff(prices)
        return np.std(diffs)

    async def evaluate_combined_trade(self, price, tick_time):
        cooldown_seconds = 10
        if self.last_trade_time and (tick_time - self.last_trade_time).total_seconds() < cooldown_seconds:
            return

        result = self.should_enter_trade(tick_time)
        if not result:
            return

        direction, confidence = result
        await self.update_trade_amount(confidence)

        if self.live_mode:
            try:
                await self.place_live_trade(direction, price, tick_time, duration=5)
                self.last_trade_time = tick_time
                self.append_diag(f"[STAKE] Confidence: {confidence:.2f} → Stake: {self.trade_amount:.2f}")
            except Exception as e:
                self.append_diag(f"[ERROR] live trade failed: {e}")
        else:
            self.simulate_trade(direction, price, tick_time, duration=5)
            self.last_trade_time = tick_time
            self.append_diag(
                f"[SIM] {direction.upper()} @ {price:.5f} | Confidence: {confidence:.2f} | Stake: {self.trade_amount:.2f}"
            )

    async def check_simulated_exits(self, tick_time):
        for sim in list(self.simulated_trades):
            if sim["result"] is None and tick_time >= sim["exit_time"]:
                exit_price = self.last_price if self.last_price is not None else sim["entry_price"]
                sim["exit_price"] = exit_price

                if sim["direction"] in ("call", "buy"):
                    sim["result"] = "win" if exit_price >= sim["entry_price"] else "loss"
                else:
                    sim["result"] = "win" if exit_price <= sim["entry_price"] else "loss"

                self.append_diag(
                    f"[SIM RESULT] {sim['result'].upper()} | Dir:{sim['direction']} | "
                    f"Entry:{sim['entry_price']:.5f} → Exit:{exit_price:.5f}"
                )

        # Update sim stats overlay
        self.update_sim_stats_label()

        # --- Live mode gate toggle ---
        sim_wr = self.simulation_winrate()
        threshold = self.sim_required_winrate
        prev = self.live_mode
        self.live_mode = sim_wr >= threshold

        self.append_diag(f"[GATE CHECK] SimWR: {sim_wr:.2%} | Threshold: {threshold:.2%} | LiveMode: {self.live_mode}")

        if prev != self.live_mode:
            self.append_diag(f"[GATE] {'LIVE ENABLED' if self.live_mode else 'LIVE DISABLED'}")

        # Update stats overlay
        self.update_sim_stats_label()

        # --- Live mode gate toggle ---
        sim_wr = self.simulation_winrate()
        threshold = self.sim_required_winrate
        prev = self.live_mode
        self.live_mode = sim_wr >= threshold

        self.append_diag(f"[GATE CHECK] SimWR: {sim_wr:.2%} | Threshold: {threshold:.2%} | LiveMode: {self.live_mode}")

        if prev != self.live_mode:
            self.append_diag(f"[GATE] {'LIVE ENABLED' if self.live_mode else 'LIVE DISABLED'}")

    def should_enter_trade(self, tick_time):
        """Loosened signal logic for rhythm and adaptive accuracy."""
        now = datetime.utcnow()
        if (now - tick_time).total_seconds() > 2.0:
            self.append_diag("[SKIP] Tick too old")
            return None

        i = self.indicators
        if None in (i["ema8"], i["ema21"], i["momentum"], i["rsi"], i["choppiness"], i["boll_low"], i["boll_high"]):
            self.append_diag("[SKIP] Missing indicator data")
            return None

        prices = [p for (_, p) in self.current_candle.get("ticks", [])]
        volatility = self.compute_volatility(prices)
        if volatility < 0.000012 or volatility > 0.00030:
            self.append_diag(f"[SKIP] Volatility out of range: {volatility:.6f}")
            return None

        ema_gap = abs(i["ema8"] - i["ema21"])
        if ema_gap < 0.000008:
            self.append_diag(f"[SKIP] EMA gap too small: {ema_gap:.5f}")
            return None

        if i["momentum"] < 0.45:
            self.append_diag(f"[SKIP] Weak momentum: {i['momentum']:.2f}")
            return None

        if i["choppiness"] > 75:
            self.append_diag(f"[SKIP] Choppy zone: {i['choppiness']:.1f}")
            return None

        # Loss cluster block (still active)
        recent = [s for s in self.simulated_trades if s.get("result") is not None][-5:]
        losses = sum(1 for s in recent if s["result"] == "loss")
        if losses >= self.loss_cluster_limit:
            self.append_diag("[BLOCK] Too many recent losses")
            return None

        price = self.last_price or prices[-1]
        near_upper = abs(price - i["boll_high"]) < 0.00007
        near_lower = abs(price - i["boll_low"]) < 0.00007

        if i["ema8"] < i["ema21"] and i["rsi"] > 53 and near_upper:
            confidence = (i["momentum"] + (i["rsi"] - 50) / 50 + ema_gap * 10000) / 3
            self.append_diag(
                f"[ENTRY] PUT | RSI:{i['rsi']:.1f} | Momentum:{i['momentum']:.2f} | BollHigh:{i['boll_high']:.5f}")
            return "put", confidence
        elif i["ema8"] > i["ema21"] and i["rsi"] < 47 and near_lower:
            confidence = (i["momentum"] + (50 - i["rsi"]) / 50 + ema_gap * 10000) / 3
            self.append_diag(
                f"[ENTRY] CALL | RSI:{i['rsi']:.1f} | Momentum:{i['momentum']:.2f} | BollLow:{i['boll_low']:.5f}")
            return "call", confidence
        else:
            self.append_diag("[SKIP] No valid signal")
            return None

    def simulation_winrate(self):
        """Return rolling winrate over the last N simulated trades."""
        recent = [s for s in self.simulated_trades if s.get("result") is not None][-self.sim_backtest_window:]
        if not recent:
            return 0.0
        wins = sum(1 for s in recent if s["result"] == "win")
        return wins / len(recent)

    def update_stats_label(self):
        """Update the stats panel using resolved live trades only."""
        resolved = [t for t in self.live_trades_history.values() if t.get("result") is not None]
        total = len(resolved)
        wins = sum(1 for t in resolved if t["result"] == "win")
        losses = total - wins
        winrate = (wins / total * 100.0) if total > 0 else 0.0

        # Sum actual PNL from broker results
        pnl = sum(t.get("pnl", 0.0) for t in resolved)

        self.trade_stats.update({
            "total": total,
            "wins": wins,
            "losses": losses,
            "pnl": pnl
        })

        self.stats_label.setText(
            f"Live Trades: {total} | Wins: {wins} | Losses: {losses} | "
            f"Winrate: {winrate:.1f}% | PNL: {pnl:.2f}"
        )

    def update_sim_stats_label(self):
        """Update the stats panel using resolved simulated trades only."""
        resolved = [s for s in self.simulated_trades if s.get("result") is not None]
        total = len(resolved)
        wins = sum(1 for s in resolved if s["result"] == "win")
        losses = total - wins
        winrate = (wins / total * 100.0) if total > 0 else 0.0

        self.sim_stats_label.setText(
            f"Sim Trades: {total} | Wins: {wins} | Losses: {losses} | Winrate: {winrate:.1f}%"
        )


    async def place_live_trade(self, direction, price, tick_time, duration=30):
        await self.update_trade_amount()
        amount = self.trade_amount

        try:
            duration_int = int(duration)  # 🔒 Force integer

            if direction == "call":
                trade_id, _ = await self.api.buy(
                    asset=self.symbol,
                    amount=amount,
                    time=duration_int,
                    check_win=False
                )
            elif direction == "put":
                trade_id, _ = await self.api.sell(
                    asset=self.symbol,
                    amount=amount,
                    time=duration_int,
                    check_win=False
                )
            else:
                self.append_diag(f"[ERROR] Invalid direction: {direction}")
                return

            self.append_diag(
                f"[LIVE] {direction.upper()} {trade_id} at {price:.5f} | dur {duration_int}s | amt {amount:.2f} | {tick_time.strftime('%H:%M:%S')}"
            )
            self.live_trades_history[trade_id] = {
                "direction": direction,
                "entry_price": price,
                "entry_time": tick_time,
                "duration": duration_int
            }

        except Exception as e:
            self.append_diag(f"[ERROR] Live trade failed → {type(e).__name__}, {e}")

    import asyncio

    async def check_trade_exit_live(self, tick_time):
        for trade_id, trade in list(self.live_trades_history.items()):
            entry_time = trade.get("entry_time")
            duration = trade.get("duration")
            if not entry_time or not duration:
                continue

            expiry = entry_time + timedelta(seconds=duration)
            if tick_time >= expiry and "result" not in trade:
                for attempt in range(3):
                    try:
                        result = await self.api.check_win(trade_id)

                        result_str = result.get("result")
                        stake = result.get("amount", 0.0)
                        profit = result.get("profit", 0.0)

                        if result_str in ("win", "loss"):
                            pnl = profit if result_str == "win" else -stake
                            trade["exit_price"] = self.last_price or trade["entry_price"]
                            trade["result"] = result_str
                            trade["pnl"] = pnl

                            self.append_diag(
                                f"[RESOLVED] Trade {trade_id} → {result_str.upper()} | "
                                f"PNL: {pnl:.2f} | Stake: {stake:.2f} | Profit: {profit:.2f}"
                            )
                            break  # ✅ Success — exit retry loop

                        else:
                            self.append_diag(f"[RETRY {attempt + 1}] Trade {trade_id} missing 'result'")
                            await asyncio.sleep(1.5)  # ⏳ Wait before retry

                    except Exception as e:
                        self.append_diag(f"[ERROR] check_win failed for {trade_id}: {e}")
                        break  # ❌ Don't retry on hard failure

        self.update_stats_label()

    # ---------------- Start / Stop ----------------
    async def start_war_machine(self):
        if not self.api:
            await self.connect_to_api()
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
        self.stream_task = asyncio.create_task(self.stream_ticks_and_build_candles(self.symbol))
        self.append_diag("⚔️ War Machine started")

    async def stop_war_machine(self):
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
                self.last_ghost_check = -1
            except asyncio.CancelledError:
                pass
        # ✅ Only sweep ghosts if loop is still running
        if asyncio.get_event_loop().is_running():
            await self.check_ghost_exits(datetime.utcnow() + timedelta(seconds=10))
            await asyncio.sleep(1)
        self.append_diag("🛑 War Machine stopped")
# ----------------- Entrypoint -----------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    gui = WarMachineGUI()
    gui.show()

    with loop:
        loop.run_forever()