# candle_builder_revised.py
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from core_indicators import (
    KalmanFilter1D,
    rsi_from_prices,
    bollinger_bands,
    choppiness_index,
    detect_momentum_from_candles,
    compute_volatility_best,
)


class CandleBuilder:
    """
    Builds fixed-duration candles from ticks, persists candles, and maintains indicators/metrics.
    - storage_file: path to JSON file storing candle list (oldest->newest). Times stored as ISO strings.
    - max_candles: max kept in memory/storage.
    - candle_duration: seconds (currently uses 60s buckets aligned to minute).
    """

    def __init__(self, storage_file: str = "war_machine_candles.json", max_candles: int = 200, engine=None):
        self.candle_data: List[Dict[str, Any]] = []
        self.current_candle: Optional[Dict[str, Any]] = None
        self.candle_start: Optional[datetime] = None
        self.candle_duration = 60
        self.storage_file = storage_file
        self.max_candles = int(max_candles)
        self.indicators: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.engine = engine  # TradeEngine instance (optional)

        # Kalman smoother used for batch smoothing of closes
        self._kalman = KalmanFilter1D(Q=1e-5, R=1e-4, adapt=False)

        self.ensure_storage_file()

    def ensure_storage_file(self) -> None:
        try:
            with open(self.storage_file, "r") as f:
                raw = json.load(f)
            # expect list of dicts; ensure times are left as ISO strings (we handle parsing if needed)
            self.candle_data = raw[-self.max_candles :]
        except Exception:
            self.candle_data = []

    def save_candles(self) -> None:
        try:
            # ensure datetimes are serializable: time stored as ISO string if it's a datetime
            out = []
            for c in self.candle_data:
                c_copy = dict(c)
                t = c_copy.get("time")
                if isinstance(t, datetime):
                    c_copy["time"] = t.isoformat()
                out.append(c_copy)
            with open(self.storage_file, "w") as f:
                json.dump(out, f, default=str)
        except Exception as e:
            print(f"[ERROR] Failed to save candles: {e}")

    def process_tick(self, tick_time: datetime, price: float) -> None:
        """
        Feed each incoming tick (tick_time is a datetime, price is float).
        This aligns ticks to minute buckets (second/microsecond zeroed).
        """
        bucket = tick_time.replace(second=0, microsecond=0)

        if self.candle_start is None:
            # start first candle
            self.candle_start = bucket
            self.current_candle = {
                "time": bucket,
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "ticks": [],
            }
        elif bucket > self.candle_start:
            # finalize previous and start a new candle (one or more minute(s) advanced)
            self.finalize_current_candle()
            self.candle_start = bucket
            self.current_candle = {
                "time": bucket,
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "ticks": [],
            }

        # update current candle
        if self.current_candle is None:
            return
        self.current_candle["high"] = max(self.current_candle["high"], float(price))
        self.current_candle["low"] = min(self.current_candle["low"], float(price))
        self.current_candle["close"] = float(price)
        # store tick as (iso, price)
        try:
            self.current_candle["ticks"].append((tick_time.isoformat(), float(price)))
        except Exception:
            # fall back to string time
            self.current_candle["ticks"].append((str(tick_time), float(price)))

    def finalize_current_candle(self) -> None:
        """
        Append current_candle to candle_data, persist, and update indicators.
        """
        if not self.current_candle:
            return

        # add volume and canonicalize time as ISO string for persistence
        self.current_candle["volume"] = len(self.current_candle.get("ticks", []))
        # ensure time is serializable in saved file; keep in-memory as datetime for convenience
        self.candle_data.append(self.current_candle)
        if len(self.candle_data) > self.max_candles:
            self.candle_data.pop(0)
        self.save_candles()
        # update indicators now that candle is closed
        self.update_indicators()

    def update_indicators(self) -> None:
        """
        Compute indicators and metrics from candle_data (oldest->newest).
        Uses Kalman smoothing on closes before EMA/RSI/Bollinger/Choppiness.
        """
        # require a minimum history to compute stable indicators
        MIN_HISTORY = 50
        if len(self.candle_data) < MIN_HISTORY:
            return

        # extract closes oldest->newest
        closes = [float(c["close"]) for c in self.candle_data]
        # smooth closes with Kalman filter (batch)
        try:
            smooth_series, _, _ = self._kalman.batch(closes)
            # KalmanFilter1D.batch returns (x, P, K) in our core; if different adjust accordingly
            # If batch returned tuple (x, P, K) keep x
            if isinstance(smooth_series, tuple) or isinstance(smooth_series, list):
                # accommodate different return shapes: try first element if nested
                pass
        except Exception:
            # fallback to raw closes if Kalman fails
            smooth_series = closes

        # For safety, ensure smooth_series is a list-like of same length
        if hasattr(smooth_series, "__len__") and len(smooth_series) == len(closes):
            smoothed = list(smooth_series)
        else:
            smoothed = closes

        # compute core indicators using smoothed series
        try:
            ema8 = None
            ema21 = None
            rsi_val = None
            chop = None
            boll_low = boll_mid = boll_high = None

            # Use last N of smoothed to compute RSI and Bollinger as batch functions expect prices oldest->newest
            max_n = max(60, MIN_HISTORY)
            smoothed_recent = smoothed[-max_n:] if len(smoothed) >= max_n else smoothed
            # rsi_from_prices expects list oldest->newest
            rsi_val = rsi_from_prices(smoothed, period=14)

            # Bollinger bands (stateless) returns (lower, middle, upper, bandwidth, percent_b, z_score)
            bb = bollinger_bands(smoothed, period=20, k=2.0, ddof=0)
            if bb[0] is not None:
                boll_low, boll_mid, boll_high, boll_band, boll_percent_b, boll_z = bb
            else:
                boll_low = boll_mid = boll_high = boll_band = boll_percent_b = boll_z = None

            # choppiness_index requires highs, lows, closes; we approximate by using candle OHLC history
            highs = [float(c.get("high", c["close"])) for c in self.candle_data]
            lows = [float(c.get("low", c["close"])) for c in self.candle_data]
            closes_for_chop = [float(c["close"]) for c in self.candle_data]
            chop = choppiness_index(highs, lows, closes_for_chop, period=14)

            # EMAs: use simple SMA-seeded EMA implemented elsewhere (if available). If not, compute approximate EMA via numpy ewm
            # Try to compute EMA using pandas-like formula without adding pandas dependency
            def compute_ema_from_series(series: List[float], period: int) -> Optional[float]:
                if series is None or len(series) < period:
                    return None
                k = 2.0 / (period + 1.0)
                # seed with SMA of first period
                seed = sum(series[:period]) / period
                ema_val = seed
                for x in series[period:]:
                    ema_val = x * k + ema_val * (1.0 - k)
                return ema_val

            ema8 = compute_ema_from_series(smoothed, period=8)
            ema21 = compute_ema_from_series(smoothed, period=21)

            # momentum detector: detect_momentum_from_candles expects candle_data oldest->newest
            momentum_dir, momentum_conf, momentum_score = detect_momentum_from_candles(self.candle_data,
                                                                                       lookback=12,
                                                                                       roc_period=6,
                                                                                       ema_period=6,
                                                                                       vol_period=20,
                                                                                       min_abs_momentum=0.01,
                                                                                       z_threshold=0.5)

            momentum_value = momentum_score if momentum_dir else 0.0
            momentum_conf_val = momentum_conf if momentum_dir else 0.0

            # price and metrics
            last_price = float(self.candle_data[-1]["close"])
            volatility = compute_volatility_best([c["close"] for c in self.candle_data[-60:]],
                                                 method="ewma", span=60, use_log=True) or 0.0
            ema_gap = abs(ema8 - ema21) if (ema8 is not None and ema21 is not None) else None
            upper_proximity = abs(last_price - boll_high) if boll_high is not None else None
            lower_proximity = abs(last_price - boll_low) if boll_low is not None else None

            # assign indicators & metrics safely
            self.indicators.update({
                "ema8": ema8,
                "ema21": ema21,
                "rsi": rsi_val,
                "choppiness": chop,
                "boll_low": boll_low,
                "boll_mid": boll_mid,
                "boll_high": boll_high,
                "momentum": float(momentum_value),
                "momentum_conf": float(momentum_conf_val),
                "boll_percent_b": boll_percent_b if 'boll_percent_b' in locals() else None,
                "boll_z": boll_z if 'boll_z' in locals() else None,
            })

            self.metrics.update({
                "volatility": float(volatility),
                "ema_gap": float(ema_gap) if ema_gap is not None else None,
                "upper_proximity": float(upper_proximity) if upper_proximity is not None else None,
                "lower_proximity": float(lower_proximity) if lower_proximity is not None else None,
                "last_price": last_price,
            })

        except Exception as e:
            print(f"[ERROR] update_indicators failed: {e}")

        # small helper printout for diagnostics (non-blocking)
        try:
            i = self.indicators
            m = self.metrics
            conf_raw = getattr(self.engine, "last_confidence", None)
            conf = format(conf_raw, ".2f") if conf_raw is not None else "N/A"
            def safe_fmt(v, fmt):
                return format(v, fmt) if v is not None else "N/A"

            print(
                f"[INDICATORS] EMA8:{safe_fmt(i.get('ema8'), '.5f')} | EMA21:{safe_fmt(i.get('ema21'), '.5f')} | "
                f"RSI:{safe_fmt(i.get('rsi'), '.1f')} | Momentum:{safe_fmt(i.get('momentum'), '.3f')} | "
                f"Choppiness:{safe_fmt(i.get('choppiness'), '.1f')} | BollLow:{safe_fmt(i.get('boll_low'), '.5f')} | "
                f"BollHigh:{safe_fmt(i.get('boll_high'), '.5f')} | Confidence:{conf}"
            )
            print(
                f"[METRICS] Vol:{safe_fmt(m.get('volatility'), '.8f')} | EMA Gap:{safe_fmt(m.get('ema_gap'), '.8f')} | "
                f"UpperProx:{safe_fmt(m.get('upper_proximity'), '.8f')} | LowerProx:{safe_fmt(m.get('lower_proximity'), '.8f')}"
            )
        except Exception:
            pass