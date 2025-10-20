from datetime import datetime, timedelta
from core_indicators import compute_volatility

class TradeEngine:
    def __init__(self, candle_builder, api=None):
        self.cb = candle_builder
        self.api = api
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

    def should_enter_trade(self):
        i = self.cb.indicators
        required_keys = ["ema8", "ema21", "momentum", "rsi", "choppiness", "boll_low", "boll_high"]
        if any(k not in i or i[k] is None for k in required_keys):
            return None

        closes = [c["close"] for c in self.cb.candle_data[-20:]]
        volatility = compute_volatility(closes)
        if volatility < 0.000012 or volatility > 0.00030:
            return None

        ema_gap = abs(i["ema8"] - i["ema21"])
        if ema_gap < 0.000008:
            return None

        if i["momentum"] < 0.45 or i["choppiness"] > 75:
            return None

        recent = [s for s in self.simulated_trades if s.get("result") is not None][-5:]
        losses = sum(1 for s in recent if s["result"] == "loss")
        if losses >= self.loss_cluster_limit:
            return None

        price = self.cb.candle_data[-1]["close"]
        near_upper = abs(price - i["boll_high"]) < 0.00007
        near_lower = abs(price - i["boll_low"]) < 0.00007

        if i["ema8"] < i["ema21"] and i["rsi"] > 53 and near_upper:
            confidence = (i["momentum"] + (i["rsi"] - 50) / 50 + ema_gap * 10000) / 3
            return "put", confidence
        elif i["ema8"] > i["ema21"] and i["rsi"] < 47 and near_lower:
            confidence = (i["momentum"] + (50 - i["rsi"]) / 50 + ema_gap * 10000) / 3
            return "call", confidence
        return None

    def simulation_winrate(self):
        recent = [s for s in self.simulated_trades if s.get("result") is not None][-self.sim_backtest_window:]
        if not recent:
            return 0.0
        wins = sum(1 for s in recent if s["result"] == "win")
        return wins / len(recent)

    def simulate_trade(self, direction, price, tick_time, duration=5):
        expiry = tick_time + timedelta(seconds=duration)
        self.simulated_trades.append({
            "direction": direction,
            "entry_price": price,
            "time": tick_time,
            "exit_time": expiry,
            "result": None
        })

    async def evaluate_trade(self, tick_time):
        cooldown_seconds = 10
        if self.last_trade_time and (tick_time - self.last_trade_time).total_seconds() < cooldown_seconds:
            return

        result = self.should_enter_trade()
        if not result:
            self.last_confidence = None
            return

        direction, confidence = result
        self.last_confidence = confidence

        direction, confidence = result
        await self.update_trade_amount(confidence)
        price = self.cb.candle_data[-1]["close"]

        if self.live_mode and self.api:
            await self.place_live_trade(direction, price, tick_time, duration=5)
        else:
            self.simulate_trade(direction, price, tick_time, duration=5)

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
            base = balance * self.base_stake_ratio
            scaled = base * (1.0 + (confidence - 0.5) * 1.5)
            self.trade_amount = max(1.0, min(scaled, self.max_stake))
        except Exception:
            self.trade_amount = 20.0

    async def place_live_trade(self, direction, price, tick_time, duration=5):
        amount = self.trade_amount
        try:
            if direction == "call":
                trade_id, _ = await self.api.buy(self.cb.symbol, amount, duration, check_win=False)
            elif direction == "put":
                trade_id, _ = await self.api.sell(self.cb.symbol, amount, duration, check_win=False)
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

                print(
                    f"[INDICATORS] EMA8:{self.cb.indicators['ema8']:.5f} | EMA21:{self.cb.indicators['ema21']:.5f} | "
                    f"RSI:{self.cb.indicators['rsi']:.1f} | Momentum:{self.cb.indicators['momentum']:.2f} | "
                    f"Confidence:{self.last_confidence:.2f}" if self.last_confidence is not None else
                    f"[INDICATORS] EMA8:{self.cb.indicators['ema8']:.5f} | EMA21:{self.cb.indicators['ema21']:.5f} | "
                    f"RSI:{self.cb.indicators['rsi']:.1f} | Momentum:{self.cb.indicators['momentum']:.2f}"
                )
