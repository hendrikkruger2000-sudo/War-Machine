import asyncio
from datetime import datetime
from candle_builder import CandleBuilder
from trade_engine import TradeEngine
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

class WarMachineBot:
    def __init__(self, symbol="EURUSD_otc", mode="demo"):
        self.symbol = symbol
        self.mode = mode
        self.api = None
        self.cb = CandleBuilder()
        self.engine = TradeEngine(self.cb, self.api)
        self.cb.engine = self.engine  # ✅ Injected here
        self.last_tick_time = datetime.utcnow()
        self.engine = TradeEngine(self.cb, self.api, symbol=self.symbol)

    async def connect(self):
        ssid_demo = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'
        ssid_real = '42["auth",{"session":"cee1286d4b06ad51b039409238b0a9aa","isDemo":0,"uid":95806403,"platform":2}]'
        ssid = ssid_demo if self.mode == "demo" else ssid_real
        self.api = PocketOptionAsync(ssid)
        self.engine.api = self.api
        await asyncio.sleep(3)

    async def stream_ticks(self):
        stream = await self.api.subscribe_symbol(self.symbol)
        print(f"[STREAM] Subscribed to {self.symbol}")

        try:
            async for tick in stream:
                price = tick.get("close") or tick.get("price") or tick.get("open")
                if price is None:
                    continue

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
                    await self.engine.evaluate_trade(tick_time)

                await self.engine.resolve_simulated_trades(tick_time)

        except asyncio.CancelledError:
            print("[STREAM] Cancelled")
        except Exception as e:
            print(f"[STREAM ERROR] {type(e).__name__}: {e}")

    async def start(self):
        await self.connect()
        await self.stream_ticks()