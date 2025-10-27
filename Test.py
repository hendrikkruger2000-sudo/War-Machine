import asyncio
import time
from datetime import datetime
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

SYMBOL = "EURUSD_otc"
STAKE_MIN = 20.0
STAKE_MAX = 2000.0
STAKE_RATIO = 0.20
OBSERVE_DURATION = 5.0
ssid = '42["auth",{"session":"pb7m1jl316va2k7gro84qs5t8a","isDemo":1,"uid":95806403,"platform":2}]'

class WarMachine:
    def __init__(self, ssid: str):
        self.api = PocketOptionAsync(ssid)
        self.latest_price = None
        self.open_price = None
        self.stake = STAKE_MIN

    async def connect_and_prepare(self):
        balance = await self.api.balance()
        self.stake = max(STAKE_MIN, min(balance * STAKE_RATIO, STAKE_MAX))
        print(f"[CONNECTED] Balance: {balance:.2f} | Stake: {self.stake:.2f}")

    async def run(self):
        await self.connect_and_prepare()
        time.sleep(5)
        stream = await self.api.subscribe_symbol(SYMBOL)
        print(stream)

if __name__ == "__main__":
    bot = WarMachine(ssid)
    asyncio.run(bot.run())