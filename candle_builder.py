import json
from datetime import datetime, timedelta
from core_indicators import ema, rsi, bollinger_bands, choppiness_index, kalman_smooth, detect_momentum_from_candles

class CandleBuilder:
    def __init__(self, storage_file="war_machine_candles.json", max_candles=200, engine=None):
        self.candle_data = []
        self.current_candle = None
        self.candle_start = None
        self.candle_duration = 60
        self.storage_file = storage_file
        self.max_candles = max_candles
        self.indicators = {}
        self.engine = engine  # ✅ Now properly defined

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
        from core_indicators import compute_volatility
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
        required_keys = ["ema8", "ema21", "rsi", "momentum", "choppiness", "boll_low", "boll_high"]
        i = self.indicators

        def safe_fmt(val, fmt):
            return format(val, fmt) if val is not None else "N/A"

        def color(val, passed, fmt=".2f"):
            return f"\033[92m{format(val, fmt)}\033[0m" if passed else f"\033[91m{format(val, fmt)}\033[0m"

        i = self.indicators
        price = self.candle_data[-1]["close"]
        volatility = compute_volatility([c["close"] for c in self.candle_data[-20:]])
        ema_gap = abs(i["ema8"] - i["ema21"])
        upper_proximity = abs(price - i["boll_high"])
        lower_proximity = abs(price - i["boll_low"])
        conf_raw = getattr(self.engine, "last_confidence", None)
        conf = format(conf_raw, ".2f") if conf_raw is not None else "N/A"

        # Indicator gates
        gap_ok = ema_gap >= 0.000005
        momentum_ok = i["momentum"] >= 0.35
        choppiness_ok = i["choppiness"] <= 85
        rsi_ok = i["rsi"] < 47 or i["rsi"] > 53
        upper_ok = upper_proximity < 0.00015
        lower_ok = lower_proximity < 0.00015
        vol_ok = 0.000008 <= volatility <= 0.00040

        print(
            f"[INDICATORS] EMA8:{format(i['ema8'], '.5f')} | EMA21:{format(i['ema21'], '.5f')} | "
            f"RSI:{color(i['rsi'], rsi_ok, '.1f')} | Momentum:{color(i['momentum'], momentum_ok)} | "
            f"Choppiness:{color(i['choppiness'], choppiness_ok, '.1f')} | BollLow:{format(i['boll_low'], '.5f')} | BollHigh:{format(i['boll_high'], '.5f')} | "
            f"Confidence:{conf}"
        )

        print(
            f"[METRICS] Volatility:{color(volatility, vol_ok, '.8f')} | EMA Gap:{color(ema_gap, gap_ok, '.8f')} | "
            f"Upper Proximity:{color(upper_proximity, upper_ok, '.8f')} | Lower Proximity:{color(lower_proximity, lower_ok, '.8f')}"
        )