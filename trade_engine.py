from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
import json
import inspect
import asyncio

class NDJSONLogger:
    def __init__(self, path):
        self.path = path

    def log(self, data):
        try:
            safe_data = json.loads(json.dumps(data, default=str))
            with open(self.path, "a") as f:
                f.write(json.dumps(safe_data) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write NDJSON log: {e}")

class TradeEngine:
    COOLDOWN_S = 60  # one trade per candle

    def __init__(self, candle_builder, api=None, symbol: str = "EURUSD_otc"):
        self.cb = candle_builder
        self.api = api
        self.symbol = symbol
        self.trade_count = 0
        self.last_trade_time: Optional[datetime] = None
        self.simulated_trades: List[Dict[str, Any]] = []
        self.live_trades_history: Dict[str, Dict[str, Any]] = {}
        self.live_mode = True
        self.trade_amount = 20.0
        self.max_stake = 2000.0
        self.last_confidence: Optional[float] = None

        self.ndjson_logger = NDJSONLogger("C:/War-Machine/war_machine_exhaustive.ndjson")
        # Clear NDJSON log at start of session
        try:
            with open("C:/War-Machine/war_machine_exhaustive.ndjson", "w") as f:
                f.write("")  # wipe file
            print("[INIT] Cleared war_machine_exhaustive.ndjson for fresh session")
        except Exception as e:
            print(f"[ERROR] Failed to clear NDJSON log: {e}")

    def _cooldown_ok(self, now: datetime) -> bool:
        if self.last_trade_time is None:
            return True
        return (now - self.last_trade_time).total_seconds() >= self.COOLDOWN_S

    async def _maybe_await(self, maybe_awaitable: Any) -> Any:
        if maybe_awaitable is None:
            return None
        if inspect.isawaitable(maybe_awaitable):
            return await maybe_awaitable
        return maybe_awaitable

    async def observe_candle_behavior(self, open_price: float, duration: float = 5.0) -> Optional[str]:
        start = datetime.utcnow()
        stayed_above = True
        stayed_below = True

        while (datetime.utcnow() - start).total_seconds() < duration:
            await asyncio.sleep(0.2)

            # Use tick-level price if available
            current_price = getattr(self.cb, "latest_tick_price", None)
            if current_price is None:
                current_price = float(self.cb.candle_data[-1]["close"])

            if current_price < open_price:
                stayed_above = False
            if current_price > open_price:
                stayed_below = False

            if not stayed_above and not stayed_below:
                return None  # mixed behavior

        if stayed_above:
            return "call"
        if stayed_below:
            return "put"
        return None

    async def evaluate_trade(self, tick_time: datetime) -> None:
        if not self._cooldown_ok(tick_time):
            return

        self.last_trade_time = tick_time
        open_price = float(self.cb.candle_data[-1]["open"])
        close_price = float(self.cb.candle_data[-1]["close"])
        indicators = self.cb.indicators
        metrics = self.cb.metrics

        direction = await self.observe_candle_behavior(open_price, duration=5.0)
        if not direction:
            print("[DECISION] No clear direction after 5s")
            return

        # === Confidence Calculation ===
        ema_gap = abs(indicators["ema8"] - indicators["ema21"])
        momentum = indicators["momentum"]
        confidence = 0.5 + min(0.25, ema_gap * 1000) + min(0.1, abs(momentum) * 2)
        confidence = max(0.5, min(confidence, 0.85))
        self.last_confidence = confidence

        # === Strategy Match Check ===
        rsi_ok = 25 < indicators["rsi"] < 40
        momentum_ok = indicators["momentum"] != 0
        choppy_ok = indicators["choppiness"] < 40
        direction_ok = (
                (direction == "call" and indicators["ema8"] > indicators["ema21"] and indicators["momentum"] > 0) or
                (direction == "put" and indicators["ema8"] < indicators["ema21"] and indicators["momentum"] < 0)
        )

        strategy_match = rsi_ok and momentum_ok and choppy_ok and direction_ok

        # === Candle Context ===
        if len(self.cb.candle_data) >= 2:
            prev = self.cb.candle_data[-2]
            prev_dir = "call" if prev["close"] > prev["open"] else "put"
            body_size = abs(prev["close"] - prev["open"])
        else:
            prev_dir = None
            body_size = None

        print(f"[DECISION] Direction: {direction} | Confidence: {confidence:.2f} | StrategyMatch: {strategy_match}")

        await self.update_trade_amount(confidence)
        duration = 60

        trade_log = {
            "direction": direction,
            "entry_price": close_price,
            "entry_time": tick_time.isoformat(),
            "duration": duration,
            "confidence": confidence,
            "strategy_match": strategy_match,
            "confidence_components": {
                "ema_gap": ema_gap,
                "momentum": momentum
            },
            "conflicts": {
                "rsi_overbought": indicators["rsi"] > 85,
                "rsi_oversold": indicators["rsi"] < 15,
                "choppy": indicators["choppiness"] > 55
            },
            "candle_context": {
                "prev_dir": prev_dir,
                "body_size": body_size
            },
            "strategy_reason": {
                "rsi_ok": rsi_ok,
                "momentum_ok": momentum_ok,
                "choppy_ok": choppy_ok,
                "direction_ok": direction_ok
            },

            "indicators": indicators,
            "metrics": metrics

        }

        if self.live_mode and self.api:
            await self.place_live_trade(direction, close_price, tick_time, duration, trade_log)
            trade_log["live"] = True
        else:
            self.simulate_trade(direction, close_price, tick_time, duration, override=True)
            trade_log["live"] = False

        self.ndjson_logger.log(trade_log)
        self.trade_count += 1
        print(f"[DEBUG] Trade #{self.trade_count} logged at {tick_time.isoformat()}")

    async def update_trade_amount(self, confidence: float = 0.0) -> None:
        try:
            if self.api:
                bal = self.api.balance()
                balance = await self._maybe_await(bal)
            else:
                balance = 1000.0
            base = balance * 0.02
            scaled = base * (1.0 + (confidence - 0.5) * 1.5)
            self.trade_amount = max(20.0, min(scaled, self.max_stake))
        except Exception:
            self.trade_amount = 20.0

    def simulate_trade(self, direction: str, price: float, tick_time: datetime, duration: int = 5,
                       override: bool = False) -> None:
        expiry = tick_time + timedelta(seconds=duration)
        self.simulated_trades.append({
            "direction": direction,
            "entry_price": price,
            "time": tick_time,
            "exit_time": expiry,
            "result": None,
            "override": override
        })

    async def place_live_trade(self, direction: str, price: float, tick_time: datetime, duration: int = 5,
                               trade_log: dict = None) -> None:
        amount = self.trade_amount
        try:
            if direction == "call":
                call_res = self.api.buy(self.symbol, amount, duration, check_win=False)
                call_ret = await self._maybe_await(call_res)
                trade_id = call_ret[0] if isinstance(call_ret, (tuple, list)) else call_ret
            else:
                put_res = self.api.sell(self.symbol, amount, duration, check_win=False)
                put_ret = await self._maybe_await(put_res)
                trade_id = put_ret[0] if isinstance(put_ret, (tuple, list)) else put_ret

            self.live_trades_history[trade_id] = {
                "direction": direction,
                "entry_price": price,
                "entry_time": tick_time,
                "duration": duration,
                "confidence": self.last_confidence
            }

            # === Get result ===
            check_res = self.api.check_win(trade_id)
            result_data = await self._maybe_await(check_res)

            result = result_data.get("result", "UNKNOWN")
            exit_price = result_data.get("closePrice") or 0.0
            conf = self.last_confidence or 0.0

            print(f"[LIVE RESULT] ID:{trade_id} | Result:{result} | Exit:{exit_price:.5f} | Confidence:{conf:.2f}")

            # === Merge result into original trade_log ===
            if trade_log is None:
                trade_log = {}

            trade_log.update({
                "trade_id": trade_id,
                "result": result,
                "exit_price": exit_price,
                "time": datetime.utcnow().isoformat()
            })

            self.ndjson_logger.log(trade_log)

        except Exception as e:
            print(f"[ERROR] Live trade failed: {e}")