# trade_engine_fixed.py
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List, Union
import asyncio
import inspect

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
    TradeEngine robust to both sync and async API implementations.
    Key change: use _maybe_await to avoid awaiting non-awaitable results.
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
        self.live_mode = False
        self.loss_cluster_limit = 3

        self.trade_amount = 20.0
        self.base_stake_ratio = 0.02
        self.max_stake = 2000.0
        self.last_confidence: Optional[float] = None

        # indicator state engines
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
    # Utility to await only when needed
    # -------------------------
    async def _maybe_await(self, maybe_awaitable: Any) -> Any:
        """
        Await the object if it's awaitable/coroutine; otherwise return it.
        Accepts coroutine objects, awaitable objects, or plain sync returns.
        """
        if maybe_awaitable is None:
            return None
        # coroutine object or awaitable value
        if inspect.isawaitable(maybe_awaitable):
            return await maybe_awaitable
        # not awaitable: return directly
        return maybe_awaitable

    # -------------------------
    # Helpers
    # -------------------------
    def _cooldown_ok(self, now: datetime) -> bool:
        if self.last_trade_time is None:
            return True
        return (now - self.last_trade_time).total_seconds() >= self.COOLDOWN_S

    def _update_states_from_latest_candle(self) -> None:
        if not hasattr(self.cb, "candle_data") or len(self.cb.candle_data) == 0:
            return

        last = self.cb.candle_data[-1]
        close = float(last["close"])
        high = float(last.get("high", close))
        low = float(last.get("low", close))

        self.ema_controller.add_price(close)
        self.rsi_state.update(close)
        self.bollinger_state.add_price(close)
        self.chop_state.add_candle(high, low, close)
        self.ewma_vol_state.add_price(close)

        ind = getattr(self.cb, "indicators", {})
        m = getattr(self.cb, "metrics", {})

        ind.setdefault("ema8", self.ema_controller.fast.ema)
        ind.setdefault("ema21", self.ema_controller.slow.ema)
        ind.setdefault("rsi", self.rsi_state.rsi)
        ind.setdefault("boll_low", None)
        ind.setdefault("boll_high", None)
        ind.setdefault("choppiness", self.chop_state.current())

        m.setdefault("volatility", self.ewma_vol_state.s2 and (self.ewma_vol_state.s2 ** 0.5) or None)
        m.setdefault("ema_gap", (self.ema_controller.fast.ema - self.ema_controller.slow.ema)
                     if (self.ema_controller.fast.ema is not None and self.ema_controller.slow.ema is not None) else None)

        self.cb.indicators = ind
        self.cb.metrics = m

    # -------------------------
    # Decision logic (unchanged semantics)
    # -------------------------
    def should_enter_trade(self) -> Optional[Tuple[str, float, bool, Dict[str, Any]]]:
        """
        Minimal gating: trade every candle with direction based on momentum sign.
        No suppression. No scoring threshold. No RSI/volatility/choppiness filters.
        """
        if not hasattr(self.cb, "indicators") or not hasattr(self.cb, "metrics"):
            return None

        i = self.cb.indicators
        m = self.cb.metrics
        inversions = self.inversion_engine.evaluate(i, m)

        # Default to 'call' if momentum is positive, else 'put'
        momentum = i.get("momentum", 0.0)
        direction = "call" if momentum >= 0 else "put"

        # Confidence based on momentum magnitude, scaled
        confidence = min(0.99, 0.5 + abs(momentum) * 1.5)
        confidence = max(confidence, 0.55)  # enforce minimum confidence

        override = True  # always allow trade
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

        # compute trade amount (works with sync or async API)
        await self.update_trade_amount(confidence)

        price = float(self.cb.candle_data[-1]["close"])
        duration = 60

        if self.live_mode and self.api:
            # place_live_trade will internally call the API safely (sync or async)
            await self.place_live_trade(direction, price, tick_time, duration=duration)
        else:
            self.simulate_trade(direction, price, tick_time, duration=duration, override=override,
                                inversion=inversion_flags)

    async def update_trade_amount(self, confidence: float = 0.0) -> None:
        try:
            if self.api:
                bal = self.api.balance()
                balance = await self._maybe_await(bal)
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
                call_res = self.api.buy(self.symbol, amount, duration, check_win=False)
                call_ret = await self._maybe_await(call_res)
                # supports both: (trade_id, something) or a single trade_id returned
                if isinstance(call_ret, (tuple, list)):
                    trade_id = call_ret[0]
                else:
                    trade_id = call_ret
            else:
                put_res = self.api.sell(self.symbol, amount, duration, check_win=False)
                put_ret = await self._maybe_await(put_res)
                if isinstance(put_ret, (tuple, list)):
                    trade_id = put_ret[0]
                else:
                    trade_id = put_ret

            self.live_trades_history[trade_id] = {
                "direction": direction,
                "entry_price": price,
                "entry_time": tick_time,
                "duration": duration,
                "confidence": self.last_confidence
            }

            # check result (api.check_win may be sync or async)
            check_res = self.api.check_win(trade_id)
            result_data = await self._maybe_await(check_res)

            result = result_data.get("result", "UNKNOWN")
            exit_price = result_data.get("closePrice") or 0.0
            conf = self.last_confidence or 0.0

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
            print(f"[ERROR] Live trade failed: {e}")

    async def check_live_trade_result(self, trade_id: str) -> None:
        try:
            result_data = await self._maybe_await(self.api.check_win(trade_id))
            result = result_data.get("result", "UNKNOWN")
            exit_price = result_data.get("closePrice") or 0.0
            conf = self.last_confidence or 0.0

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

    async def check_live_trade_result_batch(self):
        for trade_id in list(self.live_trades_history.keys()):
            await self.check_live_trade_result(trade_id)

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