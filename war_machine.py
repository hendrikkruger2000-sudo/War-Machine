import asyncio
import time
from datetime import datetime, timedelta
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

# === CONFIG ===
SYMBOL = "EURUSD_otc"
SSID = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
STAKE_RATIO = 0.20
STAKE_MIN = 20.0
STAKE_MAX = 2000.0
EARLY_SECONDS = 2.0
EXECUTION_DELAY = 7.0
EARLY_MIN_TICKS = 3
EARLY_FORCE_CONF = 0.40
EARLY_FORCE_PURITY = 0.5
SAMPLE_SLEEP = 0.05

class CandleEngine:
    def __init__(self):
        self.current_candle = None
        self.last_minute = None

    def update(self, tick):
        ts_raw = tick.get("timestamp") or tick.get("time")
        if ts_raw is None:
            return None
        try:
            ts = float(ts_raw)
            if ts > 1e12:
                ts /= 1000.0
            dt = datetime.utcfromtimestamp(ts)
            minute = dt.replace(second=0, microsecond=0)
        except:
            return None

        price = tick.get("close")
        if price is None:
            return None

        if self.last_minute is None or minute > self.last_minute:
            closed = self.current_candle
            self.current_candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "timestamp": minute.timestamp(),
                "is_closed": False,
            }
            self.last_minute = minute
            if closed:
                closed["is_closed"] = True
                return closed
            return None

        self.current_candle["high"] = max(self.current_candle["high"], price)
        self.current_candle["low"] = min(self.current_candle["low"], price)
        self.current_candle["close"] = price
        return None

class WarMachine:
    def __init__(self):
        self.api = PocketOptionAsync(SSID)
        self.latest_price = None
        self.open_price = None
        self.stake = STAKE_MIN
        self.tick_buffer = []
        self.tick_history = []
        self.trend_memory = []

    async def tick_listener(self, stream):
        async for tick in stream:
            self.latest_price = tick.get("close")
            self.tick_buffer.append(tick)

    async def run(self):
        print("Connecting to Pocket Option...")
        time.sleep(5)
        try:
            balance = await self.api.balance()
        except:
            balance = 1000.0
        self.stake = max(STAKE_MIN, min(balance * STAKE_RATIO, STAKE_MAX))
        print(f"Connected. Account balance: {balance:.2f}")

        stream = await self.api.subscribe_symbol(SYMBOL)
        engine = CandleEngine()
        asyncio.create_task(self.tick_listener(stream))

        while True:
            await asyncio.sleep(0.1)
            tick = self.tick_buffer[-1] if self.tick_buffer else None
            if tick:
                candle = engine.update(tick)
                if candle and candle["is_closed"]:
                    self.open_price = candle["open"]
                    self.trend_memory.append("call" if candle["close"] > candle["open"] else "put")
                    if len(self.trend_memory) > 3:
                        self.trend_memory.pop(0)
                    candle_open_time = datetime.utcnow()
                    print(f"New Candle Open at {candle_open_time}")
                    await self.early_bias_window(candle_open_time)

    async def early_bias_window(self, candle_open_time):
        self.tick_history = []

        async def sample_loop():
            end_time = candle_open_time + timedelta(seconds=EARLY_SECONDS)
            while datetime.utcnow() < end_time:
                if self.latest_price is not None:
                    self.tick_history.append(self.latest_price)
                await asyncio.sleep(SAMPLE_SLEEP)

        async def execute_at_precise_time():
            target_time = candle_open_time + timedelta(seconds=EXECUTION_DELAY)
            while datetime.utcnow() < target_time:
                await asyncio.sleep(0.001)
            print(f"[EXECUTION TIME] {datetime.utcnow().isoformat()} — Target: {target_time.isoformat()}")
            await self.execute_bias_trade()

        await sample_loop()
        await execute_at_precise_time()

    async def execute_bias_trade(self):
        if len(self.tick_history) < EARLY_MIN_TICKS:
            print("[EARLY BIAS] Not enough ticks — skipping")
            return

        volatility = max(self.tick_history) - min(self.tick_history)
        if volatility < 0.00005:
            print("[PURITY] Volatility too low — skipping")
            return

        chop = volatility / abs(self.tick_history[-1] - self.tick_history[0] + 1e-8)
        if chop > 10:
            print("[CHOP] Too sideways — skipping")
            return

        score = sum(
            +1.0 if tick > self.open_price else -1.0 if tick < self.open_price else 0.0
            for tick in self.tick_history
        ) / len(self.tick_history)

        conf = min(max(abs(score), 0.0), 1.0)
        direction = "call" if score > 0 else "put" if score < 0 else None

        print(f"[EARLY BIAS] ticks:{len(self.tick_history)} score:{score:.4f} → {direction} ({conf:.2f})")

        if direction and self.trend_memory.count(direction) < 2:
            print("[TREND] No alignment — skipping")
            return

        entry_delta = abs(self.latest_price - self.open_price)
        if entry_delta > 0.0001:
            print("[ENTRY] Price drifted too far — skipping")
            return

        if direction and conf >= EARLY_FORCE_CONF:
            if await self._purity_check(direction):
                print(f"[TRADE] Forcing {direction.upper()} (confidence {conf:.2f})")
                await self.execute_trade(direction)
            else:
                print("[TRADE] Purity failed — no trade")
        else:
            print("[TRADE] Bias too weak — no trade")

    async def _purity_check(self, direction):
        start = datetime.utcnow()
        while (datetime.utcnow() - start).total_seconds() < EARLY_SECONDS * EARLY_FORCE_PURITY:
            price = self.latest_price
            if price is None:
                await asyncio.sleep(SAMPLE_SLEEP)
                continue
            if direction == "call" and price < self.open_price:
                return False
            if direction == "put" and price > self.open_price:
                return False
            await asyncio.sleep(SAMPLE_SLEEP)
        return True

    async def execute_trade(self, direction):
        try:
            print(f"[TRADE] Attempting {direction.upper()} at {datetime.utcnow().isoformat()}")
            if direction == "call":
                trade_id, *_ = await self.api.buy(SYMBOL, self.stake, 5)
            else:
                trade_id, *_ = await self.api.sell(SYMBOL, self.stake, 5)

            result = await self.api.check_win(trade_id)
            outcome = result.get("result", "UNKNOWN")
            exit_price = result.get("closePrice", 0.0)
            print(f"[RESULT] {direction.upper()} | {outcome} | Exit: {exit_price:.5f}")
        except Exception as e:
            print(f"[TRADE ERROR] {e}")

if __name__ == "__main__":
    bot = WarMachine()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Stopping...")