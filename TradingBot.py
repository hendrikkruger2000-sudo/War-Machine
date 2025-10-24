import asyncio
import inspect
from datetime import datetime
from candle_builder import CandleBuilder
from trade_engine import TradeEngine
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

async def _maybe_await(maybe):
    """
    Await the value only if it's awaitable; otherwise return it directly.
    Prevents 'object list can't be used in await expression' when an API returns a plain list/tuple.
    """
    if maybe is None:
        return None
    if inspect.isawaitable(maybe):
        return await maybe
    return maybe

class WarMachineBot:
    def __init__(self, symbol="EURUSD_otc", mode="demo"):
        self.symbol = symbol
        self.mode = mode
        self.api = None
        self.cb = CandleBuilder()
        self.engine = TradeEngine(self.cb, self.api)
        self.cb.engine = self.engine  # ✅ Injected here
        self.last_tick_time = datetime.utcnow()
        # ensure engine knows symbol and api after connect
        self.engine = TradeEngine(self.cb, self.api, symbol=self.symbol)

    async def connect(self):
        ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
        ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'
        ssid = ssid_demo if self.mode == "demo" else ssid_real
        self.api = PocketOptionAsync(ssid)
        print("[CONNECT] API initialized")
        self.engine.api = self.api
        await asyncio.sleep(3)

    async def stream_ticks(self):
        # subscribe_symbol may be sync (returning an iterable/list) or async (returning an async iterator/coroutine).
        sub_result = self.api.subscribe_symbol(self.symbol)
        stream = await _maybe_await(sub_result)
        print(f"[STREAM] Subscribed to {self.symbol}")

        try:
            # If stream is a plain list/iterable of ticks (sync), iterate normally.
            # If it's an async iterable (e.g., websocket async generator), use async for.
            if hasattr(stream, "__aiter__"):
                async_iter = True
            else:
                async_iter = False

            if async_iter:
                async for tick in stream:
                    await self._handle_tick(tick)
            else:
                # sync iterable (list or generator)
                for tick in stream:
                    # keep parity: run handler in event loop
                    await self._handle_tick(tick)

        except asyncio.CancelledError:
            print("[STREAM] Cancelled")
        except Exception as e:
            print(f"[STREAM ERROR] {type(e).__name__}: {e}")

    async def _handle_tick(self, tick):
        price = tick.get("close") or tick.get("price") or tick.get("open")
        if price is None:
            return

        ts_raw = tick.get("timestamp") or tick.get("time") or tick.get("ts")
        try:
            ts = float(ts_raw)
            if ts > 1e12:
                ts /= 1000.0
            tick_time = datetime.utcfromtimestamp(ts)
        except Exception:
            tick_time = datetime.utcnow()

        self.last_tick_time = tick_time
        self.cb.process_tick(tick_time, price)

        # Trade evaluation only on candle close
        if tick_time.second == 0:
            print(f"[TICK] Evaluating trade at {tick_time.isoformat()}")
            await self.engine.evaluate_trade(tick_time)

        # Live-only: no simulated trades to resolve
        await self.engine.check_live_trade_result_batch()

    async def start(self):
        await self.connect()
        await self.stream_ticks()