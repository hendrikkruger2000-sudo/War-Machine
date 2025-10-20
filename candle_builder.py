import json
from datetime import datetime, timedelta
from core_indicators import ema, rsi, bollinger_bands, choppiness_index, kalman_smooth, detect_momentum_from_candles

class CandleBuilder:
    def __init__(self, storage_file="war_machine_candles.json", max_candles=200):
        self.candle_data = []
        self.current_candle = None
        self.candle_start = None
        self.candle_duration = 60
        self.storage_file = storage_file
        self.max_candles = max_candles
        self.indicators = {}

        self.ensure_storage_file()

    def ensure_storage_file(self):
        try:
            with open(self.storage_file, "r") as f:
                self.candle_data = json.load(f)
        except Exception:
            self.candle_data = []

    def save_candles(self):
        try:
            with open(self.storage_file, "w") as f:
                json.dump(self.candle_data, f, default=str)
        except Exception as e:
            print(f"[ERROR] Failed to save candles: {e}")

    def process_tick(self, tick_time, price):
        bucket = tick_time.replace(second=0, microsecond=0)

        if self.candle_start is None:
            self.candle_start = bucket
            self.current_candle = {
                "time": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "ticks": []
            }

        elif bucket > self.candle_start:
            self.finalize_current_candle()
            self.candle_start = bucket
            self.current_candle = {
                "time": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "ticks": []
            }

        self.current_candle["high"] = max(self.current_candle["high"], price)
        self.current_candle["low"] = min(self.current_candle["low"], price)
        self.current_candle["close"] = price
        self.current_candle["ticks"].append((tick_time.isoformat(), price))

    def finalize_current_candle(self):
        if self.current_candle:
            self.current_candle["volume"] = len(self.current_candle["ticks"])
            self.candle_data.append(self.current_candle)
            if len(self.candle_data) > self.max_candles:
                self.candle_data.pop(0)
            self.save_candles()
            self.update_indicators()

    def update_indicators(self):
        if len(self.candle_data) < 50:
            return

        closes = [c["close"] for c in self.candle_data[-50:]]
        smoothed = kalman_smooth(closes)

        self.indicators["ema8"] = ema(smoothed, period=8)
        self.indicators["ema21"] = ema(smoothed, period=21)
        self.indicators["rsi"] = rsi(smoothed, period=14)
        self.indicators["choppiness"] = choppiness_index(smoothed, period=14)
        low, mid, high = bollinger_bands(smoothed, period=20, k=2.0)
        self.indicators["boll_low"] = low
        self.indicators["boll_mid"] = mid
        self.indicators["boll_high"] = high

        momentum_dir, momentum_conf = detect_momentum_from_candles(self.candle_data)
        self.indicators["momentum"] = momentum_conf if momentum_dir else 0.0
        print(
            f"[INDICATORS] EMA8:{self.indicators['ema8']:.5f} | EMA21:{self.indicators['ema21']:.5f} | RSI:{self.indicators['rsi']:.1f} | Momentum:{self.indicators['momentum']:.2f} | Confidence:{self.indicators['confidence']:.2f}")