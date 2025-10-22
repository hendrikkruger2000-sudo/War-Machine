class GateInversionEngine:
    def __init__(self):
        self.inversion_log = []

    def evaluate(self, indicators, metrics):
        inversion_flags = {}

        # RSI–EMA misalignment → reversal setup
        if (indicators["ema8"] < indicators["ema21"] and indicators["rsi"] > 50) or \
           (indicators["ema8"] > indicators["ema21"] and indicators["rsi"] < 50):
            inversion_flags["misalignment"] = True

        # Proximity failure → breakout setup
        if metrics["upper_proximity"] > 0.0005 and metrics["lower_proximity"] > 0.0005 and \
           metrics["momentum"] > 0.8 and metrics["ema_gap"] > 0.0004:
            inversion_flags["breakout_trigger"] = True

        # Volatility too low → compression breakout
        if metrics["volatility"] < 0.000008 and metrics["momentum"] > 0.7 and indicators["rsi"] > 50:
            inversion_flags["compression_trigger"] = True

        # Volatility too high → momentum surge
        if metrics["volatility"] > 0.00040 and (indicators["rsi"] > 85 or indicators["rsi"] < 15):
            inversion_flags["volatility_surge"] = True

        # Choppiness too high → trend continuation
        if indicators["choppiness"] > 85 and indicators["rsi"] > 50 and indicators["ema8"] > indicators["ema21"]:
            inversion_flags["choppy_zone"] = True

        return inversion_flags