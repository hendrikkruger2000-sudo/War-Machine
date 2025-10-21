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
        self.live_mode = False
        self.loss_cluster_limit = 3
        self.trade_amount = 20.0
        self.base_stake_ratio = 0.02
        self.max_stake = 2000.0
        self.last_confidence = None
        from SimAudit import SimAudit

        self.audit = SimAudit()

    def should_enter_trade(self):
        i = self.cb.indicators
        required_keys = ["ema8", "ema21", "momentum", "rsi", "choppiness", "boll_low", "boll_high"]
        if any(k not in i or i[k] is None for k in required_keys):
            return None

        closes = [c["close"] for c in self.cb.candle_data[-20:]]
        volatility = compute_volatility(closes)
        if volatility < 0.000008 or volatility > 0.00040:
            return None

        ema_gap = abs(i["ema8"] - i["ema21"])
        if ema_gap < 0.000005:
            return None

        if i["momentum"] < 0.35 or i["choppiness"] > 85:
            return None

        recent = [s for s in self.simulated_trades if s.get("result") is not None][-5:]
        losses = sum(1 for s in recent if s["result"] == "loss")
        if losses >= 5:
            return None

        price = self.cb.candle_data[-1]["close"]
        upper_proximity = abs(price - i["boll_high"])
        lower_proximity = abs(price - i["boll_low"])

        # Proximity override logic
        proximity_required = True
        if i["momentum"] >= 0.7 and ema_gap >= 0.0008:
            proximity_required = False

        if i["ema8"] < i["ema21"] and i["rsi"] > 53:
            if not proximity_required or upper_proximity < 0.00015:
                confidence = (i["momentum"] + (i["rsi"] - 50) / 50 + ema_gap * 10000) / 3
                return "put", confidence, not proximity_required  # ← override flag

        elif i["ema8"] > i["ema21"] and i["rsi"] < 47:
            if not proximity_required or lower_proximity < 0.00015:
                confidence = (i["momentum"] + (50 - i["rsi"]) / 50 + ema_gap * 10000) / 3
                return "call", confidence, not proximity_required

        return None

    def simulation_winrate(self):
        recent = [s for s in self.simulated_trades if s.get("result") is not None][-self.sim_backtest_window:]
        if not recent:
            return 0.0
        wins = sum(1 for s in recent if s["result"] == "win")
        return wins / len(recent)

    def simulate_trade(self, direction, price, tick_time, duration=5, override=False):
        expiry = tick_time + timedelta(seconds=duration)
        self.simulated_trades.append({
            "direction": direction,
            "entry_price": price,
            "time": tick_time,
            "exit_time": expiry,
            "result": None,
            "override": override
        })



    async def evaluate_trade(self, tick_time):
        cooldown_seconds = 10
        if self.last_trade_time and (tick_time - self.last_trade_time).total_seconds() < cooldown_seconds:
            return

        result = self.should_enter_trade()
        if not result:
            self.last_confidence = None
            return

        direction, confidence, override = result
        self.last_confidence = confidence
        await self.update_trade_amount(confidence)
        price = self.cb.candle_data[-1]["close"]

        if self.live_mode and self.api:
            await self.place_live_trade(direction, price, tick_time, duration=5)
        else:
            self.simulate_trade(direction, price, tick_time, duration=5, override=override)

        self.last_trade_time = tick_time
        print(
            f"[TRADE] {direction.upper()} @ {price:.5f} | Confidence: {confidence:.2f} | Mode: {'LIVE' if self.live_mode else 'SIM'}")
        sim_wr = self.simulation_winrate()
        print(
            f"[GATE] SimWR:{sim_wr:.2%} | Threshold:{self.sim_required_winrate:.2%} → LiveMode:{'ON' if self.live_mode else 'OFF'}")

        if sim_wr >= self.sim_required_winrate:
            self.live_mode = True

    async def update_trade_amount(self, confidence=0.0):
        try:
            balance = await self.api.balance()
            base = balance * 0.20  # 🔥 20% of balance
            scaled = base * (1.0 + (confidence - 0.5) * 1.5)  # Optional scaling
            self.trade_amount = max(20.0, min(scaled, 2000.0))  # 🔒 Clamp between 20 and 2000
        except Exception:
            self.trade_amount = 20.0

    async def place_live_trade(self, direction, price, tick_time, duration=5):
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
