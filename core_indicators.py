import numpy as np
import math
from collections import deque
import pickle
from typing import List, Optional, Tuple, Deque, Dict


# ------------------------------
# EMA State and Controller
# ------------------------------
class EMAState:
    """
    EMAState maintains incremental EMA with SMA seeding.
    Usage:
      - call seed_add(price) repeatedly until it returns True (period samples collected)
      - thereafter call update(price, ...) to update EMA incrementally
    Note: update() will fallback to a single-tick seed only if ema is None; prefer seeding first.
    """
    def __init__(self, period: int):
        self.period = int(period)
        self.k = 2.0 / (period + 1.0)
        self.ema: Optional[float] = None
        self.last_price: Optional[float] = None
        self.seed_count = 0
        self.seed_sum = 0.0

    def seed_add(self, price: float):
        # accumulate until we have 'period' samples to seed SMA
        self.seed_sum += float(price)
        self.seed_count += 1
        if self.seed_count >= self.period:
            self.ema = self.seed_sum / self.period
            self.last_price = float(price)
            return True
        return False

    def update(self, price: float, adaptive_k: Optional[float] = None, clamp_pct: Optional[float] = None) -> float:
        """
        Update EMA with new price. Prefer to have seeded with seed_add().
        If ema is None, update will initialize ema to the current price (single-tick seed).
        adaptive_k overrides the default smoothing factor for this update.
        clamp_pct limits extreme EMA jumps relative to last_price (e.g., 0.02 means 2%).
        """
        p = float(price)
        if self.ema is None:
            # Fallback single-tick seed — prefer using seed_add() in production
            self.ema = p
            self.last_price = p
            return self.ema

        k = adaptive_k if adaptive_k is not None else self.k
        new_ema = p * k + self.ema * (1.0 - k)

        if clamp_pct is not None and self.last_price is not None:
            # Prevent extreme jumps: clamp change size relative to last_price
            max_move = abs(self.last_price) * clamp_pct
            delta = new_ema - self.ema
            if abs(delta) > max_move:
                new_ema = self.ema + math.copysign(max_move, delta)

        self.last_price = p
        self.ema = new_ema
        return self.ema

    def to_dict(self):
        return {
            "period": self.period,
            "k": self.k,
            "ema": self.ema,
            "last_price": self.last_price,
            "seed_count": self.seed_count,
            "seed_sum": self.seed_sum
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(d["period"])
        s.k = d.get("k", s.k)
        s.ema = d.get("ema")
        s.last_price = d.get("last_price")
        s.seed_count = d.get("seed_count", 0)
        s.seed_sum = d.get("seed_sum", 0.0)
        return s


class EMAController:
    """
    EMAController exposes dual EMA (fast + slow) with adaptive smoothing based on EWMA volatility.
    - Call add_price(price) each new close; returns (fast_ema, slow_ema) or (None, None) until seeded.
    - Use trend_gap() to get fast - slow gap after both seeded.
    """
    def __init__(self,
                 fast_period=8,
                 slow_period=21,
                 vol_period=20,
                 min_k_scale=0.5,
                 max_k_scale=1.8,
                 clamp_pct=0.02,
                 ewma_span_for_vol: int = 20):
        self.fast = EMAState(fast_period)
        self.slow = EMAState(slow_period)
        self.vol_period = int(vol_period)
        self.recent_closes: List[float] = []  # sliding window of closes used for vol calc
        self.min_k_scale = float(min_k_scale)
        self.max_k_scale = float(max_k_scale)
        self.clamp_pct = float(clamp_pct)
        self.ewma_span_for_vol = int(ewma_span_for_vol)

    def _ewma_vol(self, closes: List[float], span: int) -> float:
        """
        Compute EWMA volatility of returns for a stability-steering signal.
        Returns the standard deviation (not annualized). If insufficient data, returns small epsilon.
        """
        if not closes or len(closes) < 2:
            return 1e-8
        # compute log returns
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        alpha = 2.0 / (span + 1.0)
        # initialize s2 with variance of the first min(span, len(rets)) returns if available
        init_n = min(span, len(rets))
        if init_n >= 2:
            s2 = float(np.var(rets[:init_n], ddof=1))
            start = init_n
        else:
            s2 = float(rets[0] ** 2)
            start = 1
        for r in rets[start:]:
            s2 = (1.0 - alpha) * s2 + alpha * (r * r)
        return math.sqrt(max(1e-12, s2))

    def add_price(self, price: float) -> Tuple[Optional[float], Optional[float]]:
        # update rolling closes window
        self.recent_closes.append(float(price))
        if len(self.recent_closes) > self.vol_period:
            # keep last vol_period closes
            self.recent_closes.pop(0)

        # seed EMAs if needed
        if self.fast.ema is None:
            self.fast.seed_add(price)
        if self.slow.ema is None:
            self.slow.seed_add(price)

        # If not seeded both yet, return current seeds/state
        if self.fast.ema is None or self.slow.ema is None:
            return self.fast.ema, self.slow.ema

        # use EWMA vol for stable mapping
        vol = self._ewma_vol(self.recent_closes, span=self.ewma_span_for_vol)
        # inverse volatility compressed via log1p then mapped smoothly to [min_k_scale, max_k_scale]
        inv = 1.0 / max(1e-12, vol)
        raw = math.log1p(inv)  # compress large inverse values
        # map raw to [0..1] using logistic-like scaling (tunable divisor)
        t = raw / 3.0
        logistic = 1.0 / (1.0 + math.exp(-t))
        scale = self.min_k_scale + (self.max_k_scale - self.min_k_scale) * logistic

        adaptive_k_fast = min(1.0, self.fast.k * scale)
        adaptive_k_slow = min(1.0, self.slow.k * (scale * 0.6))  # slow less reactive

        # update with clamps to prevent spikes
        fast_val = self.fast.update(price, adaptive_k=adaptive_k_fast, clamp_pct=self.clamp_pct)
        slow_val = self.slow.update(price, adaptive_k=adaptive_k_slow, clamp_pct=self.clamp_pct)
        return fast_val, slow_val

    def trend_gap(self) -> Optional[float]:
        if self.fast.ema is None or self.slow.ema is None:
            return None
        return self.fast.ema - self.slow.ema

    def save_state(self, path: str):
        d = {
            "fast": self.fast.to_dict(),
            "slow": self.slow.to_dict(),
            "recent_closes": self.recent_closes
        }
        with open(path, "wb") as f:
            pickle.dump(d, f)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.fast = EMAState.from_dict(d["fast"])
        self.slow = EMAState.from_dict(d["slow"])
        self.recent_closes = list(d.get("recent_closes", []))


# ------------------------------
# RSI (Wilder) implementations
# ------------------------------
def rsi_from_prices(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Return latest RSI using Wilder smoothing computed from scratch.
    prices: list of floats (oldest -> newest)
    period: int
    Returns RSI in [0,100] or None if not enough data
    """
    if prices is None or len(prices) < period + 1:
        return None

    # compute deltas
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    # seed using first 'period' deltas
    seed_gains = [max(deltas[i], 0.0) for i in range(period)]
    seed_losses = [max(-deltas[i], 0.0) for i in range(period)]
    avg_gain = sum(seed_gains) / period
    avg_loss = sum(seed_losses) / period

    # Wilder smoothing over remaining deltas
    for d in deltas[period:]:
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(prices: List[float], period: int = 14) -> List[Optional[float]]:
    """
    Compute full RSI series (aligned to prices index) using Wilder smoothing.
    Returns list of RSI values where the first (period) entries are None until seeded.
    """
    if prices is None or len(prices) < period + 1:
        return [None] * len(prices)

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    out: List[Optional[float]] = [None] * (period)
    # seed
    seed_gains = [max(deltas[i], 0.0) for i in range(period)]
    seed_losses = [max(-deltas[i], 0.0) for i in range(period)]
    avg_gain = sum(seed_gains) / period
    avg_loss = sum(seed_losses) / period
    if avg_loss == 0.0:
        first_rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        first_rsi = 100.0 - (100.0 / (1.0 + rs))
    out.append(first_rsi)

    for d in deltas[period:]:
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0.0:
            r = 100.0
        else:
            rs = avg_gain / avg_loss
            r = 100.0 - (100.0 / (1.0 + rs))
        out.append(r)
    return out


class RSIState:
    """
    Incremental RSI state for live updates using Wilder smoothing.
    Use:
      state = RSIState(period)
      for each new price: state.update(new_price) -> RSI or None until seeded
    """
    def __init__(self, period: int = 14):
        self.period = int(period)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.prev_price: Optional[float] = None
        self.seeds: List[Tuple[float, float]] = []
        self.rsi: Optional[float] = None

    def update(self, price: float) -> Optional[float]:
        p = float(price)
        if self.prev_price is None:
            self.prev_price = p
            return None

        delta = p - self.prev_price
        self.prev_price = p
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if self.avg_gain is None:
            # collecting seed deltas
            self.seeds.append((gain, loss))
            if len(self.seeds) < self.period:
                return None
            # seed SMA
            self.avg_gain = sum(g for g, _ in self.seeds) / self.period
            self.avg_loss = sum(l for _, l in self.seeds) / self.period
        else:
            # Wilder smoothing incremental update
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        if self.avg_loss == 0.0:
            self.rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.rsi = 100.0 - (100.0 / (1.0 + rs))
        return self.rsi

    def to_dict(self) -> Dict:
        return {
            "period": self.period,
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "prev_price": self.prev_price,
            "seeds": list(self.seeds),
            "rsi": self.rsi,
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(d["period"])
        s.avg_gain = d.get("avg_gain")
        s.avg_loss = d.get("avg_loss")
        s.prev_price = d.get("prev_price")
        s.seeds = list(d.get("seeds", []))
        s.rsi = d.get("rsi")
        return s


# ------------------------------
# Bollinger Bands
# ------------------------------
def bollinger_bands(prices: List[float],
                    period: int = 20,
                    k: float = 2.0,
                    ddof: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Stateless Bollinger Bands with derived metrics:
    Returns (lower, middle, upper, bandwidth, percent_b, z_score).
    Input 'prices' must be oldest->newest.
    """
    if prices is None or len(prices) < period:
        return None, None, None, None, None, None

    window = [float(x) for x in prices[-period:]]
    n = len(window)
    mean = sum(window) / n

    denom = n - ddof
    if denom <= 0:
        std = 0.0
    else:
        sum_x = sum(window)
        sum_x2 = sum(x * x for x in window)
        var = max(0.0, (sum_x2 - (sum_x * sum_x) / n) / denom)
        std = math.sqrt(var)

    upper = mean + k * std
    lower = mean - k * std

    bandwidth = (upper - lower) / mean if mean != 0 else 0.0
    last_price = window[-1]
    range_ = upper - lower
    percent_b = (last_price - lower) / range_ if range_ > 0 else 0.5
    z_score = (last_price - mean) / std if std > 0 else 0.0

    return lower, mean, upper, bandwidth, percent_b, z_score


class BollingerState:
    """
    Incremental Bollinger Bands state for O(1) updates per price.
    Use add_price(price) each new close; call current() to get metrics.
    """
    def __init__(self, period: int = 20, k: float = 2.0, ddof: int = 0):
        self.period = int(period)
        self.k = float(k)
        self.ddof = int(ddof)
        self.window: Deque[float] = deque(maxlen=self.period)
        self.sum = 0.0
        self.sum_sq = 0.0

    def add_price(self, price: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        p = float(price)
        # if window full, remove oldest using popleft to update sums safely
        if len(self.window) == self.window.maxlen:
            old = self.window.popleft()
            self.sum -= old
            self.sum_sq -= old * old

        self.window.append(p)
        self.sum += p
        self.sum_sq += p * p

        if len(self.window) < self.period:
            return None, None, None, None, None, None

        n = len(self.window)
        mean = self.sum / n
        denom = n - self.ddof
        if denom <= 0:
            std = 0.0
        else:
            var = max(0.0, (self.sum_sq - (self.sum * self.sum) / n) / denom)
            std = math.sqrt(var)

        upper = mean + self.k * std
        lower = mean - self.k * std
        bandwidth = (upper - lower) / mean if mean != 0 else 0.0
        last_price = self.window[-1]
        range_ = upper - lower
        percent_b = (last_price - lower) / range_ if range_ > 0 else 0.5
        z_score = (last_price - mean) / std if std > 0 else 0.0

        return lower, mean, upper, bandwidth, percent_b, z_score

    def current(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        if len(self.window) < self.period:
            return None, None, None, None, None, None
        n = len(self.window)
        mean = self.sum / n
        denom = n - self.ddof
        if denom <= 0:
            std = 0.0
        else:
            var = max(0.0, (self.sum_sq - (self.sum * self.sum) / n) / denom)
            std = math.sqrt(var)
        upper = mean + self.k * std
        lower = mean - self.k * std
        bandwidth = (upper - lower) / mean if mean != 0 else 0.0
        last_price = self.window[-1]
        range_ = upper - lower
        percent_b = (last_price - lower) / range_ if range_ > 0 else 0.5
        z_score = (last_price - mean) / std if std > 0 else 0.0
        return lower, mean, upper, bandwidth, percent_b, z_score

    def to_dict(self) -> Dict:
        return {
            "period": self.period,
            "k": self.k,
            "ddof": self.ddof,
            "window": list(self.window),
            "sum": self.sum,
            "sum_sq": self.sum_sq
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BollingerState":
        s = cls(d.get("period", 20), d.get("k", 2.0), d.get("ddof", 0))
        for p in d.get("window", []):
            s.window.append(float(p))
        s.sum = d.get("sum", 0.0)
        s.sum_sq = d.get("sum_sq", 0.0)
        return s

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = self.from_dict(d)
        self.period = obj.period
        self.k = obj.k
        self.ddof = obj.ddof
        self.window = deque(obj.window, maxlen=self.period)
        self.sum = obj.sum
        self.sum_sq = obj.sum_sq


# ------------------------------
# Choppiness Index
# ------------------------------
def choppiness_index(highs: List[float],
                     lows: List[float],
                     closes: List[float],
                     period: int = 14) -> Optional[float]:
    """
    Stateless batch Choppiness Index.
    Inputs: lists of highs, lows, closes (oldest -> newest). Must have len >= period+1.
    Returns CI in [0.0, 100.0] or None if not enough data.
    """
    if not (highs and lows and closes):
        return None
    n = len(closes)
    if len(highs) != n or len(lows) != n or n < period + 1:
        return None

    start = n - period
    tr_sum = 0.0
    for i in range(start, n):
        hi = highs[i]
        lo = lows[i]
        prev_close = closes[i - 1]
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        tr_sum += tr

    highest_high = max(highs[start:n])
    lowest_low = min(lows[start:n])
    range_hl = highest_high - lowest_low

    if range_hl <= 1e-12 or tr_sum <= 1e-12:
        return 100.0

    ci = 100.0 * math.log10(tr_sum / range_hl) / math.log10(period)
    return max(0.0, min(100.0, ci))


class ChoppinessState:
    """
    Stateful incremental Choppiness Index. Feed add_candle(high, low, close).
    Returns CI when seeded (period TRs available), else None.
    """
    def __init__(self, period: int = 14):
        self.period = int(period)
        self.highs: Deque[float] = deque(maxlen=period)
        self.lows: Deque[float] = deque(maxlen=period)
        # keep closes length period+1 so we always have prev_close for TR calc
        self.closes: Deque[float] = deque(maxlen=period + 1)
        self.tr_queue: Deque[float] = deque(maxlen=period)
        self.tr_sum = 0.0

    def add_candle(self, high: float, low: float, close: float) -> Optional[float]:
        # need prev close to compute true range for this candle
        if len(self.closes) == 0:
            self.closes.append(close)
            self.highs.append(high)
            self.lows.append(low)
            return None

        prev_close = self.closes[-1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        # maintain rolling TR sum
        if len(self.tr_queue) == self.tr_queue.maxlen:
            old_tr = self.tr_queue.popleft()
            self.tr_sum -= old_tr
        self.tr_queue.append(tr)
        self.tr_sum += tr

        # maintain OHLC windows
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        if len(self.tr_queue) < self.period:
            return None

        highest_high = max(self.highs)
        lowest_low = min(self.lows)
        range_hl = highest_high - lowest_low

        if range_hl <= 1e-12 or self.tr_sum <= 1e-12:
            return 100.0

        ci = 100.0 * math.log10(self.tr_sum / range_hl) / math.log10(self.period)
        return max(0.0, min(100.0, ci))

    def current(self) -> Optional[float]:
        if len(self.tr_queue) < self.period:
            return None
        highest_high = max(self.highs)
        lowest_low = min(self.lows)
        range_hl = highest_high - lowest_low
        if range_hl <= 1e-12 or self.tr_sum <= 1e-12:
            return 100.0
        ci = 100.0 * math.log10(self.tr_sum / range_hl) / math.log10(self.period)
        return max(0.0, min(100.0, ci))

    def to_dict(self) -> Dict:
        return {
            "period": self.period,
            "highs": list(self.highs),
            "lows": list(self.lows),
            "closes": list(self.closes),
            "tr_queue": list(self.tr_queue),
            "tr_sum": self.tr_sum
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ChoppinessState":
        s = cls(d.get("period", 14))
        for v in d.get("highs", []):
            s.highs.append(float(v))
        for v in d.get("lows", []):
            s.lows.append(float(v))
        for v in d.get("closes", []):
            s.closes.append(float(v))
        for v in d.get("tr_queue", []):
            s.tr_queue.append(float(v))
        s.tr_sum = float(d.get("tr_sum", 0.0))
        return s

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = self.from_dict(d)
        self.period = obj.period
        self.highs = deque(obj.highs, maxlen=self.period)
        self.lows = deque(obj.lows, maxlen=self.period)
        self.closes = deque(obj.closes, maxlen=self.period + 1)
        self.tr_queue = deque(obj.tr_queue, maxlen=self.period)
        self.tr_sum = obj.tr_sum


# ------------------------------
# Kalman filters
# ------------------------------
class KalmanFilter1D:
    """
    Simple level-only Kalman filter with optional adaptive Q tuning.
    """
    def __init__(self,
                 Q: float = 1e-5,
                 R: float = 1e-4,
                 x0: Optional[float] = None,
                 P0: float = 1.0,
                 adapt_window: int = 50,
                 adapt: bool = False):
        self.Q = float(Q)
        self.R = float(R)
        self.x = x0
        self.P = float(P0)
        self.adapt_window = max(1, int(adapt_window))
        self.adapt = bool(adapt)
        self.resid_buf = deque(maxlen=self.adapt_window)

    def batch(self, prices: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(prices)
        x = np.zeros(n, dtype=float)
        P = np.zeros(n, dtype=float)
        K = np.zeros(n, dtype=float)

        if n == 0:
            return x, P, K

        first_valid = None
        for v in prices:
            if v is not None and not np.isnan(v):
                first_valid = float(v)
                break
        if first_valid is None:
            return x, P, K
        if self.x is None:
            self.x = first_valid
        x_prev = float(self.x)
        P_prev = float(self.P)

        for t in range(n):
            z = prices[t]
            x_pred = x_prev
            P_pred = P_prev + self.Q

            if z is None or np.isnan(z):
                x_t = x_pred
                P_t = P_pred
                K_t = 0.0
            else:
                K_t = P_pred / (P_pred + self.R)
                x_t = x_pred + K_t * (float(z) - x_pred)
                P_t = (1.0 - K_t) * P_pred

                if self.adapt:
                    resid = float(z) - x_pred
                    self.resid_buf.append(resid)
                    if len(self.resid_buf) == self.resid_buf.maxlen:
                        sigma2 = np.var(np.array(self.resid_buf, dtype=float), ddof=1)
                        self.Q = max(1e-12, 0.1 * sigma2)

            x[t] = x_t
            P[t] = P_t
            K[t] = K_t

            x_prev = x_t
            P_prev = P_t

        self.x = x_prev
        self.P = P_prev
        return x, P, K

    def update(self, z: Optional[float]) -> Tuple[float, float]:
        if self.x is None:
            if z is None or np.isnan(z):
                self.x = 0.0
            else:
                self.x = float(z)
        x_pred = self.x
        P_pred = self.P + self.Q

        if z is None or np.isnan(z):
            self.x = x_pred
            self.P = P_pred
            return self.x, self.P

        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (float(z) - x_pred)
        self.P = (1.0 - K) * P_pred

        if self.adapt:
            resid = float(z) - x_pred
            self.resid_buf.append(resid)
            if len(self.resid_buf) == self.resid_buf.maxlen:
                sigma2 = np.var(np.array(self.resid_buf, dtype=float), ddof=1)
                self.Q = max(1e-12, 0.1 * sigma2)

        return self.x, self.P


class KalmanFilterLevelTrend:
    """
    Two-state Kalman filter with state [level, trend]. Use for trend extraction.
    """
    def __init__(self,
                 dt: float = 1.0,
                 Q: Optional[np.ndarray] = None,
                 R: float = 1e-4,
                 x0: Optional[np.ndarray] = None,
                 P0: Optional[np.ndarray] = None,
                 adapt: bool = False,
                 adapt_window: int = 50):
        self.dt = float(dt)
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]], dtype=float)
        self.H = np.array([[1.0, 0.0]], dtype=float)
        if Q is None:
            q_level = 1e-5
            q_trend = 1e-5
            self.Q = np.array([[q_level, 0.0],
                               [0.0, q_trend]], dtype=float)
        else:
            self.Q = np.array(Q, dtype=float)
        self.R = float(R)
        if x0 is None:
            self.x = np.zeros((2,), dtype=float)
        else:
            self.x = np.array(x0, dtype=float).reshape(2)
        if P0 is None:
            self.P = np.eye(2, dtype=float) * 1.0
        else:
            self.P = np.array(P0, dtype=float)
        self.adapt = bool(adapt)
        self.adapt_window = max(1, int(adapt_window))
        self.resid_buf = deque(maxlen=self.adapt_window)

    def batch(self, prices: List[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(prices)
        xs = np.zeros((n, 2), dtype=float)
        Ps = np.zeros((n, 2, 2), dtype=float)

        x_prev = self.x.copy()
        P_prev = self.P.copy()

        for t in range(n):
            x_pred = self.F.dot(x_prev)
            P_pred = self.F.dot(P_prev).dot(self.F.T) + self.Q

            z = prices[t]
            if z is None or (isinstance(z, float) and np.isnan(z)):
                x_upd = x_pred
                P_upd = P_pred
            else:
                zf = float(z)
                S = (self.H.dot(P_pred).dot(self.H.T) + self.R).item()
                K = P_pred.dot(self.H.T) / S
                y = zf - (self.H.dot(x_pred)).item()
                x_upd = x_pred + (K.flatten() * y)
                P_upd = (np.eye(2) - K.dot(self.H)).dot(P_pred)

                if self.adapt:
                    self.resid_buf.append(y)
                    if len(self.resid_buf) == self.resid_buf.maxlen:
                        sigma2 = np.var(np.array(self.resid_buf, dtype=float), ddof=1)
                        self.Q = np.diag([max(1e-12, 0.05 * sigma2), max(1e-12, 0.01 * sigma2)])

            xs[t, :] = x_upd
            Ps[t, :, :] = P_upd
            x_prev = x_upd
            P_prev = P_upd

        self.x = x_prev
        self.P = P_prev
        return xs[:, 0], xs[:, 1]

    def update(self, z: Optional[float]) -> Tuple[float, float]:
        x_pred = self.F.dot(self.x)
        P_pred = self.F.dot(self.P).dot(self.F.T) + self.Q

        if z is None or (isinstance(z, float) and np.isnan(z)):
            self.x = x_pred
            self.P = P_pred
            return float(self.x[0]), float(self.x[1])

        zf = float(z)
        S = (self.H.dot(P_pred).dot(self.H.T) + self.R).item()
        K = P_pred.dot(self.H.T) / S
        y = zf - (self.H.dot(x_pred)).item()
        self.x = x_pred + (K.flatten() * y)
        self.P = (np.eye(2) - K.dot(self.H)).dot(P_pred)

        if self.adapt:
            self.resid_buf.append(y)
            if len(self.resid_buf) == self.resid_buf.maxlen:
                sigma2 = np.var(np.array(self.resid_buf, dtype=float), ddof=1)
                self.Q = np.diag([max(1e-12, 0.05 * sigma2), max(1e-12, 0.01 * sigma2)])

        return float(self.x[0]), float(self.x[1])


# ------------------------------
# Volatility estimators
# ------------------------------
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0.0, None) else default

def log_returns(prices: List[float]) -> List[float]:
    return [math.log(float(prices[i]) / float(prices[i - 1])) for i in range(1, len(prices))]


def rolling_volatility(prices: List[float],
                       window: int = 60,
                       use_log: bool = True,
                       ddof: int = 1,
                       annualize: bool = False,
                       periods_per_year: int = 252) -> Optional[float]:
    if prices is None or len(prices) < window + 1:
        return None
    rets = log_returns(prices) if use_log else [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    window_rets = np.asarray(rets[-window:], dtype=float)
    if window_rets.size < 2:
        return None
    sigma = float(np.std(window_rets, ddof=ddof))
    return sigma * math.sqrt(periods_per_year) if annualize else sigma


def ewma_volatility(prices: List[float],
                    span: int = 60,
                    use_log: bool = True,
                    annualize: bool = False,
                    periods_per_year: int = 252) -> Optional[float]:
    if prices is None or len(prices) < 2:
        return None
    rets = np.array(log_returns(prices) if use_log else [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))], dtype=float)
    alpha = 2.0 / (span + 1.0)
    init_n = min(span, len(rets))
    if init_n >= 2:
        s2 = float(np.var(rets[:init_n], ddof=1))
        start = init_n
    else:
        s2 = float(rets[0] ** 2)
        start = 1
    for r in rets[start:]:
        s2 = (1.0 - alpha) * s2 + alpha * (r * r)
    vol = math.sqrt(max(0.0, s2))
    return vol * math.sqrt(periods_per_year) if annualize else vol


def realized_volatility(returns: List[float],
                        interval_aggregate: int = 1,
                        annualize: bool = False,
                        periods_per_year: int = 252) -> Optional[float]:
    if not returns:
        return None
    n_blocks = len(returns) // interval_aggregate
    if n_blocks < 1:
        return None
    realized_var = 0.0
    for i in range(n_blocks):
        block = returns[i * interval_aggregate:(i + 1) * interval_aggregate]
        realized_var += sum(r * r for r in block)
    realized_var /= n_blocks
    vol = math.sqrt(max(0.0, realized_var))
    return vol * math.sqrt(periods_per_year) if annualize else vol


def robust_volatility(prices: List[float],
                      window: int = 60,
                      use_log: bool = True,
                      annualize: bool = False,
                      periods_per_year: int = 252) -> Optional[float]:
    if prices is None or len(prices) < window + 1:
        return None
    rets = log_returns(prices) if use_log else [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    block = rets[-window:]
    med = np.median(block)
    mad = np.median([abs(r - med) for r in block])
    sigma = 1.4826 * mad
    return float(sigma * math.sqrt(periods_per_year)) if annualize else float(sigma)


class EWMAVolState:
    def __init__(self, span: int = 60, use_log: bool = True, periods_per_year: int = 252, annualize: bool = False):
        self.alpha = 2.0 / (span + 1.0)
        self.s2: Optional[float] = None
        self.prev_price: Optional[float] = None
        self.use_log = bool(use_log)
        self.periods_per_year = periods_per_year
        self.annualize = bool(annualize)

    def add_price(self, price: float) -> Optional[float]:
        p = float(price)
        if self.prev_price is None:
            self.prev_price = p
            return None
        r = math.log(p / self.prev_price) if self.use_log else (p - self.prev_price) / self.prev_price
        self.prev_price = p
        if self.s2 is None:
            self.s2 = r * r
        else:
            self.s2 = (1.0 - self.alpha) * self.s2 + self.alpha * (r * r)
        vol = math.sqrt(max(0.0, self.s2))
        return vol * math.sqrt(self.periods_per_year) if self.annualize else vol

    def state(self) -> Dict:
        return {"alpha": self.alpha, "s2": self.s2, "prev_price": self.prev_price,
                "use_log": self.use_log, "periods_per_year": self.periods_per_year, "annualize": self.annualize}

    def load_state(self, d: Dict):
        self.alpha = d.get("alpha", self.alpha)
        self.s2 = d.get("s2", self.s2)
        self.prev_price = d.get("prev_price", self.prev_price)
        self.use_log = d.get("use_log", self.use_log)
        self.periods_per_year = d.get("periods_per_year", self.periods_per_year)
        self.annualize = d.get("annualize", self.annualize)


def compute_volatility_best(prices: List[float],
                            method: str = "ewma",
                            **kwargs) -> Optional[float]:
    if method == "ewma":
        return ewma_volatility(prices, **kwargs)
    if method == "rolling":
        return rolling_volatility(prices, **kwargs)
    if method == "realized":
        returns = kwargs.get("returns")
        if returns is not None:
            return realized_volatility(returns, kwargs.get("interval_aggregate", 1), kwargs.get("annualize", False), kwargs.get("periods_per_year", 252))
        return ewma_volatility(prices, **kwargs)
    if method == "robust":
        return robust_volatility(prices, **kwargs)
    return None


# ------------------------------
# Momentum detector (improved)
# ------------------------------
def detect_momentum_from_candles(candle_data,
                                 lookback: int = 12,
                                 roc_period: int = 6,
                                 ema_period: int = 6,
                                 vol_period: int = 20,
                                 min_abs_momentum: float = 0.02,
                                 z_threshold: float = 0.5):
    """
    High-precision momentum detector that returns (direction, confidence, momentum_score).
    Inputs:
      - candle_data: list of dicts with "close" (oldest->newest)
    """
    if not candle_data:
        return None, 0.0, 0.0

    closes = [c["close"] for c in candle_data]
    N = len(closes)
    required = max(lookback, roc_period, vol_period) + 1
    if N < required:
        return None, 0.0, 0.0

    # safety: ensure roc_period index valid
    if roc_period >= N:
        return None, 0.0, 0.0

    p_t = closes[-1]
    p_t_roc = closes[-1 - roc_period]
    if p_t_roc == 0:
        return None, 0.0, 0.0
    raw_roc = (p_t - p_t_roc) / p_t_roc

    # deltas over lookback
    start_idx = max(1, N - lookback)
    deltas = [closes[i] - closes[i - 1] for i in range(start_idx, N)]
    # EMA-smooth deltas
    alpha = 2.0 / (ema_period + 1.0)
    ema = deltas[0]
    for d in deltas[1:]:
        ema = alpha * d + (1 - alpha) * ema
    smooth_momentum = ema / (closes[-1] if closes[-1] != 0 else 1.0)

    # volatility normalization using EWMA of returns for stability
    vol_window = closes[-(vol_period + 1):] if len(closes) >= vol_period + 1 else closes
    vol = 0.0
    if len(vol_window) >= 2:
        vol = ewma_volatility(vol_window, span=min(vol_period, max(3, vol_period)))
        if vol is None:
            vol = 0.0

    # denom: robust floor using asset-scale heuristic
    min_vol = max(1e-8, abs(raw_roc) * 0.5, vol * 0.3)
    denom = max(min_vol, vol, abs(raw_roc), 1e-12)

    z_raw = raw_roc / denom
    z_smooth = smooth_momentum / denom

    momentum_score = 0.65 * z_smooth + 0.35 * z_raw

    # consensus guard
    ups = sum(1 for i in range(N - lookback + 1, N) if closes[i] > closes[i - 1])
    downs = lookback - ups
    consensus = (ups - downs) / float(lookback) if lookback > 0 else 0.0

    clipped = max(-5.0, min(5.0, momentum_score))
    signed_norm = clipped / 5.0

    abs_score = abs(signed_norm)
    conf_from_score = max(0.0, (abs_score - (z_threshold / 5.0)) / (1.0 - (z_threshold / 5.0)))
    conf = 0.75 * conf_from_score + 0.25 * abs(consensus)
    conf = max(0.0, min(1.0, conf))

    if abs(signed_norm) < min_abs_momentum:
        return None, 0.0, round(signed_norm, 4)

    direction = "call" if signed_norm > 0 else "put"
    return direction, round(conf, 3), round(signed_norm, 4)