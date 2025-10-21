from datetime import datetime

class SimAudit:
    def __init__(self):
        self.logs = []

    def log_trade(self, trade):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "direction": trade["direction"],
            "entry_price": trade["entry_price"],
            "exit_price": trade.get("exit_price"),
            "result": trade.get("result"),
            "confidence": trade.get("confidence"),
            "override": trade.get("override", False),
            "mode": "SIM"
        }
        self.logs.append(entry)
        self.print_entry(entry)

    def print_entry(self, entry):
        print(
            f"[SIM AUDIT] {entry['direction'].upper()} | Entry:{entry['entry_price']:.5f} | "
            f"Exit:{entry['exit_price']:.5f} | Result:{entry['result'].upper()} | "
            f"Confidence:{entry['confidence']:.2f} | Override:{entry['override']} | Time:{entry['timestamp']}"
        )