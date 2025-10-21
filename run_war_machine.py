# run_war_machine.py
import asyncio
from TradingBot import WarMachineBot

async def main():
    bot = WarMachineBot(symbol="EURUSD_otc", mode="demo")
    await bot.start()


asyncio.run(main())