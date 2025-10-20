import numpy as np
import math

def ema(prices, period=8):
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema_val = prices[0]
    for p in prices[1:]:
        ema_val = p * k + ema_val * (1 - k)
    return ema_val

def rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = prices[-i] - prices[-i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def bollinger_bands(prices, period=20, k=2.0):
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    ma = float(np.mean(window))
    std = float(np.std(window))
    return ma - k * std, ma, ma + k * std

def choppiness_index(prices, period=14):
    if len(prices) < period + 1:
        return None
    window = prices[-period:]
    high, low = max(window), min(window)
    sum_tr = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    if high == low or sum_tr <= 1e-12:
        return 100.0
    return 100.0 * math.log10(sum_tr / (high - low)) / math.log10(period)

def kalman_smooth(prices):
    n = len(prices)
    x = np.zeros(n)
    P = np.zeros(n)
    x[0] = prices[0]
    P[0] = 1.0
    Q = 0.00001
    R = 0.0001

    for t in range(1, n):
        x_pred = x[t - 1]
        P_pred = P[t - 1] + Q
        K = P_pred / (P_pred + R)
        x[t] = x_pred + K * (prices[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return x.tolist()

def compute_volatility(prices):
    if len(prices) < 2:
        return 0.0
    diffs = np.diff(prices)
    return np.std(diffs)

def detect_momentum_from_candles(candle_data, window=6):
    if len(candle_data) < window + 1:
        return None, 0.0
    closes = [c["close"] for c in candle_data[-(window + 1):]]
    up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    down_moves = window - up_moves
    if up_moves > down_moves:
        confidence = up_moves / window
        return "call", confidence
    elif down_moves > up_moves:
        confidence = down_moves / window
        return "put", confidence
    return None, 0.0