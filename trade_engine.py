# trade_engine_revised.py
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List

from core_indicators import (
    compute_volatility_best,
    detect_momentum_from_candles,
    EMAController,
    RSIState,
    BollingerState,
    ChoppinessState,
    EWMAVolState,
)
from SimAudit import SimAudit
from GateInversionEngine import GateInversionEngine


class TradeEngine:
    """
    Revised TradeEngine that uses core_indicators' stateful APIs and safer decision logic.
    Contracts:
      - candle_builder.cb.candle_data: list of candles oldest->newest, each with open/high/low/close
      - candle_builder.cb.indicators and .metrics will be maintained by caller; missing keys are computed here when necessary
    """
    COOLDOWN_S = 60

    def __init__(self, candle_builder, api=None, symbol: str = "EURUSD_otc"):
        self.cb = candle_builder
        self.api = api
        self.symbol = symbol

        self.last_trade_time: Optional[datetime] = None
        self.simulated_trades: List[Dict[str, Any]] = []
        self.live_trades_history: Dict[str, Dict[str, Any]] = {}
        self.sim_backtest_window = 20
        self.sim_required_winrate = 0.60
        self.live_mode = False  # start in simulation unless explicitly enabled
        self.loss_cluster_limit = 3

        self.trade_amount = 20.0
        self.base_stake_ratio = 0.02
        self.max_stake = 2000.0
        self.last_confidence: Optional[float] = None

        # indicator state engines (persist these across restarts in your app)
        self.ema_controller = EMAController(fast_period=8, slow_period=21, vol_period=20,
                                            min_k_scale=0.5, max_k_scale=1.8, clamp_pct=0.02,
                                            ewma_span_for_vol=20)
        self.rsi_state = RSIState(period=14)
        self.bollinger_state = BollingerState(period=20, k=2.0, ddof=0)
        self.chop_state = ChoppinessState(period=14)
        self.ewma_vol_state = EWMAVolState(span=60, use_log=True, annualize=False)
        self.inversion_engine = GateInversionEngine()
        self.audit = SimAudit()

    # -------------------------
    # Helpers
    # -------------------------
    def _cooldown_ok(self, now: datetime) -> bool:
        if self.last_trade_time is None:
            return True
        return (now - self.last_trade_time).total_seconds() >= self.COOLDOWN_S

    def _update_states_from_latest_candle(self) -> None:
        """
        Ensure incremental states are updated from the latest candle.
        Call before making decisions.
        """
        if not hasattr(self.cb, "candle_data") or len(self.cb.candle_data) == 0:
            return

        last = self.cb.candle_data[-1]
        close = float(last["close"])
        high = float(last.get("high", close))
        low = float(last.get("low", close))

        # Update EMA controller (it seeds internally)
        self.ema_controller.add_price(close)

        # Update RSIState
        self.rsi_state.update(close)

        # BollingerState
        self.bollinger_state.add_price(close)

        # ChoppinessState needs OHLC and close history; add_candle returns None until seeded
        self.chop_state.add_candle(high, low, close)

        # EWMA vol state (live normalization)
        self.ewma_vol_state.add_price(close)

        # update cb.indicators / metrics placeholders if not provided by candle_builder
        ind = getattr(self.cb, "indicators", {})
        m = getattr(self.cb, "metrics", {})

        ind.setdefault("ema8", self.ema_controller.fast.ema)
        ind.setdefault("ema21", self.ema_controller.slow.ema)
        ind.setdefault("rsi", self.rsi_state.rsi)
        # momentum will be computed via detect_momentum_from_candles below
        ind.setdefault("boll_low", None)
        ind.setdefault("boll_high", None)
        ind.setdefault("choppiness", self.chop_state.current())

        m.setdefault("volatility", self.ewma_vol_state.s2 and (self.ewma_vol_state.s2 ** 0.5) or None)
        # ema_gap as fast - slow if available
        m.setdefault("ema_gap", (self.ema_controller.fast.ema - self.ema_controller.slow.ema)
                     if (self.ema_controller.fast.ema is not None and self.ema_controller.slow.ema is not None) else None)

        # write back
        self.cb.indicators = ind
        self.cb.metrics = m

    # -------------------------
    # Decision logic (reworked)
    # -------------------------
    def should_enter_trade(self) -> Optional[Tuple[str, float, bool, Dict[str, Any]]]:
        """
        Returns (direction, confidence, override_allowed, inversion_flags) or None.
        Uses:
          - detect_momentum_from_candles for robust momentum + confidence
          - EMAController for trend gap
          - choppiness for regime detection
          - bollinger percent_b and proximity metrics
          - strict suppression gates before scoring
        """
        if not hasattr(self.cb, "candle_data") or len(self.cb.candle_data) < 5:
            return None

        # ensure internal states reflect latest candle
        self._update_states_from_latest_candle()

        candle_data = self.cb.candle_data
        ind = self.cb.indicators
        m = self.cb.metrics

        # compute momentum + confidence using the batch detector
        direction_m, conf_m, momentum_score = detect_momentum_from_candles(
            candle_data,
            lookback=12,
            roc_period=6,
            ema_period=6,
            vol_period=20,
            min_abs_momentum=0.01,
            z_threshold=0.5,
        )

        # If detector returned None, no momentum
        if direction_m is None:
            ind["momentum"] = 0.0
            ind["momentum_conf"] = 0.0
        else:
            ind["momentum"] = momentum_score
            ind["momentum_conf"] = conf_m

        # fill in Bollinger metrics from state if available
        bb = self.bollinger_state.current()
        if bb[0] is not None:
            lower, middle, upper, bandwidth, percent_b, z_score = bb
            ind["boll_low"] = lower
            ind["boll_mid"] = middle
            ind["boll_high"] = upper
            ind["boll_bandwidth"] = bandwidth
            ind["boll_percent_b"] = percent_b
            ind["boll_z"] = z_score
            # proximity metrics (distance to bands normalized by price)
            last_price = float(candle_data[-1]["close"])
            m["lower_proximity"] = abs(last_price - lower) / (last_price if last_price != 0 else 1.0)
            m["upper_proximity"] = abs(upper - last_price) / (last_price if last_price != 0 else 1.0)
        else:
            m.setdefault("lower_proximity", None)
            m.setdefault("upper_proximity", None)

        # choppiness
        ch = self.chop_state.current()
        ind["choppiness"] = ch

        # ema gap
        ema_gap = m.get("ema_gap")
        # volatility (use compute_volatility_best if metric missing)
        vol = m.get("volatility") or compute_volatility_best([c["close"] for c in candle_data], method="ewma", span=60)

        m["volatility"] = vol
        m["ema_gap"] = ema_gap

        # inversion flags
        inversions = self.inversion_engine.evaluate(ind, m) if self.inversion_engine else {}

        # --- Hard suppression gates ---
        # require momentum confidence > 0.45 and momentum score nonzero
        if ind.get("momentum_conf", 0.0) < 0.45 or abs(ind.get("momentum", 0.0)) < 0.01:
            return None

        # choppiness threshold: only trade when trending (low chop) or when compression trigger present
        if ch is not None and ch > 65 and not inversions.get("compression_trigger"):
            return None

        # avoid extreme RSI saturation
        rsi = ind.get("rsi")
        if rsi is None:
            return None
        if rsi >= 95 or rsi <= 5:
            return None

        # candle confirmation: require last candle direction to align with momentum direction
        last_c = candle_data[-1]
        last_price_up = last_c["close"] > last_c["open"]
        if ind["momentum"] > 0 and not last_price_up:
            return None
        if ind["momentum"] < 0 and last_price_up:
            return None

        # volatility filter: if market too wild, require stronger momentum_conf
        if vol is not None and vol > 0.0015 and ind.get("momentum_conf", 0.0) < 0.7:
            return None

        # multi-indicator alignment: trend direction from EMAs
        ema_trend = None
        if self.ema_controller.fast.ema is not None and self.ema_controller.slow.ema is not None:
            if self.ema_controller.fast.ema > self.ema_controller.slow.ema:
                ema_trend = 1
            elif self.ema_controller.fast.ema < self.ema_controller.slow.ema:
                ema_trend = -1

        # map momentum sign to direction
        direction = "call" if ind["momentum"] > 0 else "put"

        # if EMA trend exists, require alignment or require inversion flag to override
        if ema_trend is not None:
            if (ind["momentum"] > 0 and ema_trend < 0) or (ind["momentum"] < 0 and ema_trend > 0):
                # allow if inversion indicates misalignment trade (rare)
                if not inversions.get("misalignment"):
                    return None

        # --- Scoring (conservative) ---
        score = 0.0
        # momentum strength (scaled)
        score += min(2.0, abs(ind["momentum"]) * 2.0)
        # momentum confidence
        score += ind.get("momentum_conf", 0.0) * 2.0
        # EMA gap
        if ema_gap is not None:
            score += min(1.0, max(-0.5, ema_gap * 10000.0))  # scaled to sensible units
        # Bollinger proximity reward (close to lower band for calls, upper for puts)
        if ind.get("boll_percent_b") is not None:
            if direction == "call":
                score += max(0.0, 0.5 - ind["boll_percent_b"])  # prefer near lower band
            else:
                score += max(0.0, ind["boll_percent_b"] - 0.5)  # prefer near upper band
        # choppiness penalty
        if ch is not None:
            score += max(0.0, (50.0 - ch) / 50.0)  # higher score when ch < 50

        # inversion bonuses
        if inversions.get("compression_trigger"):
            score += 1.0
        if inversions.get("breakout_trigger"):
            score += 0.8
        if inversions.get("misalignment"):
            score += 0.6

        # require a minimum score to proceed
        if score < 2.0:
            return None

        # compute confidence as a smooth mapping of score (clamped)
        confidence = min(0.975, 0.5 + (score / 8.0))
        self.last_confidence = confidence

        # override allowed only when proximity to band is safe
        override = not ((m.get("upper_proximity") is not None and m["upper_proximity"] < 0.0005) or
                        (m.get("lower_proximity") is not None and m["lower_proximity"] < 0.0005))

        return direction, confidence, override, inversions

    # -------------------------
    # Simulation / execution
    # -------------------------
    def simulation_winrate(self) -> float:
        recent = [s for s in self.simulated_trades if s.get("result") is not None][-self.sim_backtest_window:]
        if not recent:
            return 0.0
        wins = sum(1 for s in recent if s["result"] == "win")
        return wins / len(recent)

    def simulate_trade(self, direction: str, price: float, tick_time: datetime, duration: int = 60,
                       override: bool = False, inversion: Optional[Dict] = None) -> None:
        expiry = tick_time + timedelta(seconds=duration)
        self.simulated_trades.append({
            "direction": direction,
            "entry_price": price,
            "time": tick_time,
            "exit_time": expiry,
            "result": None,
            "override": override,
            "inversion": inversion or {}
        })

    async def evaluate_trade(self, tick_time: datetime) -> None:
        # cooldown guard
        if not self._cooldown_ok(tick_time):
            return
        self.last_trade_time = tick_time

        decision = self.should_enter_trade()
        if not decision:
            self.last_confidence = None
            return

        direction, confidence, override, inversion_flags = decision
        self.last_confidence = confidence

        # compute trade amount (non-blocking; safe fallback if no API)
        await self.update_trade_amount(confidence)

        price = float(self.cb.candle_data[-1]["close"])
        duration = 60

        if self.live_mode and self.api:
            await self.place_live_trade(direction, price, tick_time, duration=duration)
        else:
            self.simulate_trade(direction, price, tick_time, duration=duration, override=override,
                                inversion=inversion_flags)

    async def update_trade_amount(self, confidence: float = 0.0) -> None:
        try:
            if self.api:
                balance = await self.api.balance()
            else:
                balance = 1000.0
            base = balance * self.base_stake_ratio
            scaled = base * (1.0 + (confidence - 0.5) * 1.5)
            self.trade_amount = max(20.0, min(scaled, self.max_stake))
        except Exception:
            self.trade_amount = 20.0

    async def place_live_trade(self, direction: str, price: float, tick_time: datetime, duration: int = 60) -> None:
        amount = self.trade_amount
        try:
            if direction == "call":
                trade_id, _ = await self.api.buy(self.symbol, amount, duration, check_win=False)
            else:
                trade_id, _ = await self.api.sell(self.symbol, amount, duration, check_win=False)

            self.live_trades_history[trade_id] = {
                "direction": direction,
                "entry_price": price,
                "entry_time": tick_time,
                "duration": duration,
                "confidence": self.last_confidence
            }

            await self.check_live_trade_result(trade_id)

        except Exception as e:
            print(f"[ERROR] Live trade failed: {e}")

    async def check_live_trade_result(self, trade_id: str) -> None:
        try:
            result_data = await self.api.check_win(trade_id)
            result = result_data.get("result", "UNKNOWN")
            exit_price = result_data.get("closePrice") or 0.0
            conf = self.last_confidence or 0.0

            # simple logging
            print(f"[LIVE RESULT] ID:{trade_id} | Result:{result} | Exit:{exit_price:.5f} | Confidence:{conf:.2f}")
            self.audit.log_trade({
                "direction": self.live_trades_history.get(trade_id, {}).get("direction", "unknown"),
                "entry_price": self.live_trades_history.get(trade_id, {}).get("entry_price", 0.0),
                "exit_price": exit_price,
                "result": result,
                "confidence": conf,
                "override": False,
                "time": datetime.utcnow().isoformat()
            })

        except Exception as e:
            print(f"[ERROR] Failed to check result: {e}")

    async def resolve_simulated_trades(self, tick_time: datetime) -> None:
        for sim in list(self.simulated_trades):
            if sim["result"] is None and tick_time >= sim["exit_time"]:
                exit_price = float(self.cb.candle_data[-1]["close"])
                sim["exit_price"] = exit_price
                if sim["direction"] == "call":
                    sim["result"] = "win" if exit_price >= sim["entry_price"] else "loss"
                else:
                    sim["result"] = "win" if exit_price <= sim["entry_price"] else "loss"
                sim["confidence"] = self.last_confidence or 0.0
                sim["override"] = sim.get("override", False)
                self.audit.log_trade(sim)

                i = self.cb.indicators
                required = ["ema8", "ema21", "momentum", "rsi", "choppiness", "boll_low", "boll_high"]
                if all(k in i and i[k] is not None for k in required):
                    conf = sim.get("confidence", 0.0)
                    print(f"[INDICATORS] EMA8:{i['ema8']:.5f} | EMA21:{i['ema21']:.5f} | "
                          f"RSI:{i['rsi']:.1f} | Momentum:{i['momentum']:.3f} | "
                          f"Choppiness:{i['choppiness']:.1f} | BollLow:{i['boll_low']:.5f} | BollHigh:{i['boll_high']:.5f} | "
                          f"Confidence:{conf:.2f}")
                else:
                    print("[INDICATORS] Not ready")