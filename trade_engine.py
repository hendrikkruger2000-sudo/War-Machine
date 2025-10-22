from datetime import datetime, timedelta
from core_indicators import compute_volatility

class TradeEngine:
    def __init__(self, candle_builder, api=None, symbol="EURUSD_otc"):
        self.cb = candle_builder
        self.api = api
        self.symbol = symbol

        self.last_trade_time = None
        self.simulated_trades = []
        self.live_trades_history = {}
        self.sim_backtest_window = 5
        self.sim_required_winrate = 0.60
        self.live_mode = True
        self.loss_cluster_limit = 3
        self.trade_amount = 20.0
        self.base_stake_ratio = 0.02
        self.max_stake = 2000.0
        self.last_confidence = None
        from SimAudit import SimAudit
        from GateInversionEngine import GateInversionEngine

        self.inversion_engine = GateInversionEngine()

        self.audit = SimAudit()

    def should_enter_trade(self):
        if not hasattr(self.cb, "metrics") or not hasattr(self.cb, "indicators"):
            return None

        i = self.cb.indicators
        m = self.cb.metrics
        inversions = self.inversion_engine.evaluate(i, m)

        # 🔍 Setup classification
        direction = None
        if i["ema8"] > i["ema21"] and i["rsi"] > 53:
            direction = "call"
        elif i["ema8"] < i["ema21"] and i["rsi"] < 47:
            direction = "put"
        else:
            if inversions.get("misalignment"):
                direction = "call" if i["rsi"] > 50 else "put"

        if not direction:
            return None

        # ❌ Suppression logic
        if i["rsi"] > 20 and m["lower_proximity"] > 0.00025:
            return None
        if i["choppiness"] < 10 and i["momentum"] < 0.7:
            return None

        # 📊 Scoring system
        score = 0
        if m["ema_gap"] > 0.0002: score += 1
        if i["momentum"] > 0.65: score += 1
        if i["rsi"] < 18 or i["rsi"] > 53: score += 1
        if m["volatility"] < 0.0004: score += 1
        if m["lower_proximity"] < 0.00015: score += 2  # 🔥 Strong reversal zone
        if inversions.get("breakout_trigger") or inversions.get("compression_trigger"): score += 1
        if i["choppiness"] > 15: score += 1  # 🧠 Compression breakout zone

        # 🧠 Multi-candle confirmation
        recent = self.cb.candle_data[-3:]
        if direction == "call" and all(c["close"] > c["open"] for c in recent): score += 1
        if direction == "put" and all(c["close"] < c["open"] for c in recent): score += 1

        # 🔥 Confidence scaling
        if score < 4:
            return None

        confidence = 0.65 + (score * 0.05)
        confidence = min(confidence, 0.95)

        override = not (m["upper_proximity"] < 0.0005 or m["lower_proximity"] < 0.0005)

        return direction, confidence, override, inversions

    def simulation_winrate(self):
        recent = [s for s in self.simulated_trades if s.get("result") is not None][-self.sim_backtest_window:]
        if not recent:
            return 0.0
        wins = sum(1 for s in recent if s["result"] == "win")
        return wins / len(recent)

    def simulate_trade(self, direction, price, tick_time, duration=5, override=False, inversion=None):
        expiry = tick_time + timedelta(seconds=duration)
        self.simulated_trades.append({
            "direction": direction,
            "entry_price": price,
            "time": tick_time,
            "exit_time": expiry,
            "result": None,
            "override": override,
            "inversion": inversion or {}
        })

    async def evaluate_trade(self, tick_time):
        result = self.should_enter_trade()
        if not result:
            self.last_confidence = None
            return

        direction, confidence, override, inversion_flags = result
        self.last_confidence = confidence
        await self.update_trade_amount(confidence)
        price = self.cb.candle_data[-1]["close"]

        if self.live_mode and self.api:
            await self.place_live_trade(direction, price, tick_time, duration=5)
        else:
            self.simulate_trade(direction, price, tick_time, duration=5, override=override, inversion=inversion_flags)

    async def update_trade_amount(self, confidence=0.0):
        try:
            balance = await self.api.balance()
            base = balance * 0.20  # 🔥 20% of balance
            scaled = base * (1.0 + (confidence - 0.5) * 1.5)  # Optional scaling
            self.trade_amount = max(20.0, min(scaled, 2000.0))  # 🔒 Clamp between 20 and 2000
        except Exception:
            self.trade_amount = 20.0

    async def place_live_trade(self, direction, price, tick_time, duration=15):
        amount = self.trade_amount
        try:
            if direction == "call":
                trade_id, _ = await self.api.buy(self.symbol, amount, duration, check_win=False)
                print(trade_id)
            elif direction == "put":
                trade_id, _ = await self.api.sell(self.symbol, amount, duration, check_win=False)
                print(trade_id)
            else:
                return

            self.live_trades_history[trade_id] = {
                "direction": direction,
                "entry_price": price,
                "entry_time": tick_time,
                "duration": duration
            }

        except Exception as e:
            print(f"[ERROR] Live trade failed: {e}")

    async def resolve_simulated_trades(self, tick_time):
        for sim in list(self.simulated_trades):
            if sim["result"] is None and tick_time >= sim["exit_time"]:
                exit_price = self.cb.candle_data[-1]["close"]
                sim["exit_price"] = exit_price

                if sim["direction"] == "call":
                    sim["result"] = "win" if exit_price >= sim["entry_price"] else "loss"
                else:
                    sim["result"] = "win" if exit_price <= sim["entry_price"] else "loss"

                sim["confidence"] = self.last_confidence if self.last_confidence is not None else 0.0
                sim["override"] = sim.get("override", False)
                self.audit.log_trade(sim)

                i = self.cb.indicators
                required_keys = ["ema8", "ema21", "momentum", "rsi", "choppiness", "boll_low", "boll_high"]

                if all(k in i and i[k] is not None for k in required_keys):
                    conf = self.last_confidence if self.last_confidence is not None else 0.0
                    print(
                        f"[INDICATORS] EMA8:{i['ema8']:.5f} | EMA21:{i['ema21']:.5f} | "
                        f"RSI:{i['rsi']:.1f} | Momentum:{i['momentum']:.2f} | "
                        f"Choppiness:{i['choppiness']:.1f} | BollLow:{i['boll_low']:.5f} | BollHigh:{i['boll_high']:.5f} | "
                        f"Confidence:{conf:.2f}"
                    )
                else:
                    print("[INDICATORS] Not ready")
