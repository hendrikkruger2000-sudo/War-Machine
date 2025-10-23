import math
from datetime import datetime, timezone, timedelta
import pickle
from typing import Dict, Any, Optional


class GateInversionEngine:
    """
    Improved gate/inversion engine.

    Usage:
      engine = GateInversionEngine(config=dict(...))
      out = engine.evaluate(indicators, metrics)
      out -> {"flags": {"misalignment": {"on": True, "conf": 0.6}, ...}, "score": 1.85}
    """

    DEFAULTS = {
        # misalignment
        "misalignment_min_rsi_div": 3.0,      # RSI distance from 50 required (absolute)
        "misalignment_min_ema_gap_pct": 0.00005,  # ema_gap normalized to price (fraction)
        # breakout
        "breakout_min_momentum": 0.6,
        "breakout_min_normalized_gap": 0.0003,
        "breakout_percent_b_edge": 0.85,     # prefer price outside/near edge
        # compression
        "compression_max_vol": 2e-5,         # very low vol
        "compression_min_momentum": 0.45,
        "compression_percent_b_center": 0.15,  # percent_b distance from center allowed
        # volatility surge
        "volatility_surge_min": 0.00035,
        "volatility_surge_high": 0.0010,
        # choppy
        "choppy_threshold": 80.0,
        "choppy_min_trend_strength": 0.00015,  # normalized EMA gap to call it trend
        # logging/pruning
        "log_prune_days": 14
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(self.DEFAULTS)
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.inversion_log = []

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _safe_get(d: Dict, k: str, default: float = 0.0) -> float:
        v = d.get(k) if isinstance(d, dict) else None
        return float(v) if v is not None else default

    @staticmethod
    def _sigmoid_scale(x: float, x0: float = 1.0, steep: float = 6.0) -> float:
        # maps positive x to (0,1), with midpoint at x0
        try:
            return 1.0 / (1.0 + math.exp(-steep * (x / x0 - 1.0)))
        except OverflowError:
            return 1.0 if x > x0 else 0.0

    @staticmethod
    def _now_iso():
        return datetime.now(timezone.utc).isoformat()

    # -- public API -------------------------------------------------------------
    def evaluate(self, indicators: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate inversion triggers.

        indicators: dict with keys like 'ema8','ema21','rsi','choppiness', etc.
        metrics: dict with keys like 'volatility','ema_gap','boll_percent_b','lower_proximity','upper_proximity'
        """
        flags: Dict[str, Dict[str, Any]] = {}
        total_score = 0.0

        # safe fetches
        ema8 = self._safe_get(indicators, "ema8", None)
        ema21 = self._safe_get(indicators, "ema21", None)
        rsi = self._safe_get(indicators, "rsi", 50.0)
        choppiness = self._safe_get(indicators, "choppiness", 100.0)
        boll_percent_b = self._safe_get(indicators, "boll_percent_b", 0.5)

        vol = self._safe_get(metrics, "volatility", 0.0)
        ema_gap = self._safe_get(metrics, "ema_gap", 0.0)
        # normalize ema_gap relative to price if price available
        last_price = self._safe_get(metrics, "last_price", None)
        if last_price and last_price != 0:
            ema_gap_pct = abs(ema_gap) / abs(last_price)
        else:
            ema_gap_pct = abs(ema_gap)  # fallback absolute

        lower_prox = self._safe_get(metrics, "lower_proximity", None)
        upper_prox = self._safe_get(metrics, "upper_proximity", None)
        momentum = self._safe_get(indicators, "momentum", 0.0)
        momentum_conf = self._safe_get(indicators, "momentum_conf", 0.0)

        # --- misalignment: EMA direction opposite RSI (RSI away from 50)
        mis_rsi_div = abs(rsi - 50.0)
        ema_trend = 0
        if ema8 is not None and ema21 is not None:
            if ema8 > ema21:
                ema_trend = 1
            elif ema8 < ema21:
                ema_trend = -1

        mis_conf = 0.0
        if ema_trend != 0:
            # sign of RSI relative to 50 and EMA trend
            rsi_side = 1 if rsi > 50 else -1 if rsi < 50 else 0
            if rsi_side != 0 and rsi_side != ema_trend:
                # stronger if RSI is further from 50 and ema gap is material
                gap_scale = min(1.0, ema_gap_pct / max(1e-8, self.cfg["misalignment_min_ema_gap_pct"]))
                rsi_scale = min(1.0, mis_rsi_div / max(1e-8, self.cfg["misalignment_min_rsi_div"]))
                mis_conf = 0.25 + 0.75 * (0.5 * gap_scale + 0.5 * rsi_scale)  # baseline 0.25 -> up to 1.0
                flags["misalignment"] = {"on": True, "conf": round(min(1.0, mis_conf), 3)}
                total_score += flags["misalignment"]["conf"]

        # --- breakout trigger: price near band edge + momentum + normalized gap
        # use boll_percent_b and proximity metrics
        bp = boll_percent_b
        near_edge = bp > self.cfg["breakout_percent_b_edge"] or bp < (1.0 - self.cfg["breakout_percent_b_edge"])
        normalized_gap_ok = ema_gap_pct >= self.cfg["breakout_min_normalized_gap"]
        if near_edge and momentum >= self.cfg["breakout_min_momentum"] and normalized_gap_ok:
            # conf grows with momentum and gap
            conf = 0.3 + 0.7 * min(1.0, (momentum - self.cfg["breakout_min_momentum"]) / 0.5 + (ema_gap_pct / (self.cfg["breakout_min_normalized_gap"] * 5)))
            flags["breakout_trigger"] = {"on": True, "conf": round(min(1.0, conf), 3)}
            total_score += flags["breakout_trigger"]["conf"]

        # --- compression trigger: very low vol, momentum present, price centered in band
        centered = abs(bp - 0.5) <= 0.25  # within center half of band
        if vol <= self.cfg["compression_max_vol"] and momentum >= self.cfg["compression_min_momentum"] and centered:
            # stronger if vol is extremely low and momentum_conf adequate
            vol_scale = 1.0 - min(1.0, vol / max(1e-12, self.cfg["compression_max_vol"]))
            conf = 0.2 + 0.8 * (0.5 * vol_scale + 0.5 * min(1.0, momentum_conf))
            flags["compression_trigger"] = {"on": True, "conf": round(min(1.0, conf), 3)}
            total_score += flags["compression_trigger"]["conf"]

        # --- volatility surge: tiered by how high vol is
        if vol >= self.cfg["volatility_surge_min"]:
            # base conf proportional to distance above min
            ratio = (vol - self.cfg["volatility_surge_min"]) / max(1e-12, (self.cfg["volatility_surge_high"] - self.cfg["volatility_surge_min"]))
            conf = min(1.0, 0.25 + 0.75 * ratio)
            # require RSI saturation or strong momentum to mark as surge
            if rsi >= 80 or rsi <= 20 or momentum >= 0.8:
                flags["volatility_surge"] = {"on": True, "conf": round(conf, 3)}
                total_score += conf

        # --- choppy zone: high choppiness and weak trend strength
        if choppiness >= self.cfg["choppy_threshold"]:
            trend_strength = ema_gap_pct
            if trend_strength < self.cfg["choppy_min_trend_strength"]:
                conf = min(1.0, (choppiness - self.cfg["choppy_threshold"]) / 20.0 + 0.25)
                flags["choppy_zone"] = {"on": True, "conf": round(conf, 3)}
                total_score += conf

        # record event in inversion_log for diagnostics
        entry = {
            "time": self._now_iso(),
            "flags": flags,
            "indicators_snapshot": {
                "rsi": rsi, "ema_gap_pct": round(ema_gap_pct, 8), "vol": round(vol, 8), "boll_percent_b": round(bp, 3),
                "momentum": round(momentum, 3), "momentum_conf": round(momentum_conf, 3)
            },
            "score": round(float(total_score), 3)
        }
        self.inversion_log.append(entry)
        # prune old log entries
        self._prune_log()

        return {"flags": flags, "score": round(float(total_score), 3)}

    # -- diagnostics / persistence ----------------------------------------------
    def _prune_log(self) -> None:
        """Remove entries older than log_prune_days to keep memory bounded."""
        max_age_days = int(self.cfg.get("log_prune_days", 14))
        if max_age_days <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        # keep only entries newer than cutoff
        self.inversion_log = [e for e in self.inversion_log if datetime.fromisoformat(e["time"]) >= cutoff]

    def reset(self) -> None:
        self.inversion_log = []

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg, "log": self.inversion_log}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.cfg = d.get("cfg", self.cfg)
        self.inversion_log = d.get("log", [])