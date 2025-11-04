from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
import inspect
import asyncio

class TradeEngine:
    COOLDOWN_S = 60

    def __init__(self, candle_builder, api=None, symbol: str = "EURUSD_otc"):
        self.cb = candle_builder
        self.api = api
        self.symbol = symbol
        self.trade_count = 0
        self.last_trade_time: Optional[datetime] = None
        self.live_mode = True
        self.trade_amount = 20.0
        self.max_stake = 2000.0
        self.last_confidence: Optional[float] = None

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
            current_price = getattr(self.cb, "latest_tick_price", None)
            if current_price is None:
                current_price = float(self.cb.candle_data[-1]["close"])

            if current_price < open_price:
                stayed_above = False
            if current_price > open_price:
                stayed_below = False

            if not stayed_above and not stayed_below:
                return None

        if stayed_above:
            return "call"
        if stayed_below:
            return "put"
        return None

    def evaluate_strategy_match(self, direction: str) -> bool:
        indicators = self.cb.indicators
        required = ["ema8", "ema21", "rsi", "momentum", "choppiness", "boll_percent_b", "macd_histogram"]
        if not all(k in indicators and indicators[k] is not None for k in required):
            return False

        rsi = indicators["rsi"]
        momentum = indicators["momentum"]
        choppy = indicators["choppiness"]
        percent_b = indicators["boll_percent_b"]
        macd_hist = indicators["macd_histogram"]
        ema8 = indicators["ema8"]
        ema21 = indicators["ema21"]

        rsi_ok = 20 <= rsi <= 30
        momentum_ok = -0.1 <= momentum <= 0.1
        choppy_ok = 40 <= choppy <= 60
        boll_ok = percent_b < 0.2 if direction == "call" else percent_b > 0.8
        macd_ok = macd_hist > 0 if direction == "call" else macd_hist < 0
        ema_ok = (direction == "call" and ema8 > ema21) or (direction == "put" and ema8 < ema21)

        if len(self.cb.candle_data) >= 2:
            prev = self.cb.candle_data[-2]
            body_size = abs(prev["close"] - prev["open"])
            prev_dir = "call" if prev["close"] > prev["open"] else "put"
            zone_confirmed = body_size >= 0.0005 and prev_dir != direction
        else:
            zone_confirmed = False

        return all([rsi_ok, momentum_ok, choppy_ok, boll_ok, macd_ok, ema_ok, zone_confirmed])

    async def evaluate_trade(self, tick_time: datetime) -> None:
        if not self._cooldown_ok(tick_time):
            return

        indicators = self.cb.indicators
        required = ["ema8", "ema21", "rsi", "momentum", "choppiness", "boll_percent_b", "macd_histogram"]
        if not all(k in indicators and indicators[k] is not None for k in required):
            return

        self.last_trade_time = tick_time
        open_price = float(self.cb.candle_data[-1]["open"])
        close_price = float(self.cb.candle_data[-1]["close"])

        direction = await self.observe_candle_behavior(open_price, duration=5.0)
        if not direction:
            return

        if not self.evaluate_strategy_match(direction):
            return

        ema_gap = abs(indicators["ema8"] - indicators["ema21"])
        momentum = indicators["momentum"]
        confidence = 0.5 + min(0.25, ema_gap * 1000) + min(0.1, abs(momentum) * 2)
        confidence = max(0.5, min(confidence, 0.85))
        self.last_confidence = confidence

        await self.update_trade_amount(confidence)
        duration = 60

        if self.live_mode and self.api:
            await self.place_live_trade(direction, close_price, tick_time, duration)

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

    async def place_live_trade(self, direction: str, price: float, tick_time: datetime, duration: int = 5) -> None:
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

            check_res = self.api.check_win(trade_id)
            result_data = await self._maybe_await(check_res)

            result = result_data.get("result", "UNKNOWN")
            exit_price = result_data.get("closePrice") or 0.0
            conf = self.last_confidence or 0.0

            print(f"[TRADE] {tick_time.isoformat()} | {direction.upper()} | Result: {result} | Exit: {exit_price:.5f} | Confidence: {conf:.2f}")

        except Exception as e:
            print(f"[ERROR] Trade failed: {e}")