import json
from collections import defaultdict

def load_trades(path):
    trades = []
    try:
        with open(path, "r") as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    trades.append(trade)
                except Exception:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to load trades: {e}")
    return trades

def normalize_conflicts(trade):
    conflicts = trade.get("conflicts", {})
    for key in conflicts:
        val = conflicts[key]
        if isinstance(val, str):
            conflicts[key] = val.lower() == "true"
    trade["conflicts"] = conflicts
    return trade

def debug_counts(trades):
    true_matches = sum(1 for t in trades if t.get("strategy_match") is True)
    false_matches = sum(1 for t in trades if t.get("strategy_match") is not True)
    print(f"\n[DEBUG] Trades with strategy_match:true → {true_matches}")
    print(f"[DEBUG] Trades with strategy_match:false → {false_matches}")
    print(f"[DEBUG] Total trades loaded → {len(trades)}")

def strategy_summary(trades):
    print("\n=== Strategy Profiler Summary ===\n")
    buckets = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0, "total": 0})

    for t in trades:
        conf = round(t.get("confidence", 0.0), 2)
        match = t.get("strategy_match") is True
        key = f"Match:{'True' if match else 'False'} | Conf:{conf}"
        buckets[key]["total"] += 1
        result = t.get("result")
        if result == "win":
            buckets[key]["win"] += 1
        elif result == "loss":
            buckets[key]["loss"] += 1
        elif result == "draw":
            buckets[key]["draw"] += 1

    true_keys = [k for k in buckets if k.startswith("Match:True")]
    false_keys = [k for k in buckets if k.startswith("Match:False")]

    print("\n--- Match:TRUE ---")
    for k in sorted(true_keys):
        stats = buckets[k]
        win_rate = (stats["win"] / stats["total"]) * 100 if stats["total"] else 0
        draw_rate = (stats["draw"] / stats["total"]) * 100 if stats["total"] else 0
        print(f"{k} → Total:{stats['total']} | Win:{stats['win']} | Loss:{stats['loss']} | Draw:{stats['draw']} | WinRate:{win_rate:.1f}% | DrawRate:{draw_rate:.1f}%")

    print("\n--- Match:FALSE ---")
    for k in sorted(false_keys):
        stats = buckets[k]
        win_rate = (stats["win"] / stats["total"]) * 100 if stats["total"] else 0
        draw_rate = (stats["draw"] / stats["total"]) * 100 if stats["total"] else 0
        print(f"{k} → Total:{stats['total']} | Win:{stats['win']} | Loss:{stats['loss']} | Draw:{stats['draw']} | WinRate:{win_rate:.1f}% | DrawRate:{draw_rate:.1f}%")

def conflict_analysis(trades):
    print("\n=== Conflict Analysis ===\n")
    conflict_stats = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0, "total": 0})

    for t in trades:
        conflicts = t.get("conflicts", {})
        result = t.get("result")
        for key, val in conflicts.items():
            if val is True:
                conflict_stats[key]["total"] += 1
                if result == "win":
                    conflict_stats[key]["win"] += 1
                elif result == "loss":
                    conflict_stats[key]["loss"] += 1
                elif result == "draw":
                    conflict_stats[key]["draw"] += 1

    for name, stats in conflict_stats.items():
        win_rate = (stats["win"] / stats["total"]) * 100 if stats["total"] else 0
        print(f"{name} → Total:{stats['total']} | Win:{stats['win']} | Loss:{stats['loss']} | Draw:{stats['draw']} | WinRate:{win_rate:.1f}%")

def indicator_analysis(trades):
    print("\n=== Indicator Impact ===\n")
    buckets = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0, "total": 0})

    for t in trades:
        indicators = t.get("indicators", {})
        result = t.get("result")

        if "rsi" in indicators:
            rsi_bucket = int(indicators["rsi"] // 10) * 10
            key = f"RSI:{rsi_bucket}"
            buckets[key]["total"] += 1
            if result == "win":
                buckets[key]["win"] += 1
            elif result == "loss":
                buckets[key]["loss"] += 1
            elif result == "draw":
                buckets[key]["draw"] += 1

        if "ema8" in indicators and "ema21" in indicators:
            ema_gap = abs(indicators["ema8"] - indicators["ema21"])
            ema_bucket = int(ema_gap * 1000)  # e.g. 0.002 → 2
            key = f"EMA_Gap:{ema_bucket}"
            buckets[key]["total"] += 1
            if result == "win":
                buckets[key]["win"] += 1
            elif result == "loss":
                buckets[key]["loss"] += 1
            elif result == "draw":
                buckets[key]["draw"] += 1

        if "momentum" in indicators:
            momentum_bucket = round(indicators["momentum"], 1)
            key = f"Momentum:{momentum_bucket}"
            buckets[key]["total"] += 1
            if result == "win":
                buckets[key]["win"] += 1
            elif result == "loss":
                buckets[key]["loss"] += 1
            elif result == "draw":
                buckets[key]["draw"] += 1

        if "choppiness" in indicators:
            choppy_bucket = int(indicators["choppiness"] // 10) * 10
            key = f"Choppy:{choppy_bucket}"
            buckets[key]["total"] += 1
            if result == "win":
                buckets[key]["win"] += 1
            elif result == "loss":
                buckets[key]["loss"] += 1
            elif result == "draw":
                buckets[key]["draw"] += 1

    for key, stats in sorted(buckets.items()):
        win_rate = (stats["win"] / stats["total"]) * 100 if stats["total"] else 0
        draw_rate = (stats["draw"] / stats["total"]) * 100 if stats["total"] else 0
        print(f"{key} → Total:{stats['total']} | Win:{stats['win']} | Loss:{stats['loss']} | Draw:{stats['draw']} | WinRate:{win_rate:.1f}% | DrawRate:{draw_rate:.1f}%")

def strategy_reason_analysis(trades):
    print("\n=== Strategy Filter Breakdown ===\n")
    reason_stats = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0, "total": 0})

    for t in trades:
        result = t.get("result")
        reasons = t.get("strategy_reason", {})
        for key, val in reasons.items():
            if val is True:
                reason_stats[key]["total"] += 1
                if result == "win":
                    reason_stats[key]["win"] += 1
                elif result == "loss":
                    reason_stats[key]["loss"] += 1
                elif result == "draw":
                    reason_stats[key]["draw"] += 1

    for key, stats in sorted(reason_stats.items()):
        win_rate = (stats["win"] / stats["total"]) * 100 if stats["total"] else 0
        draw_rate = (stats["draw"] / stats["total"]) * 100 if stats["total"] else 0
        print(f"{key} → Total:{stats['total']} | Win:{stats['win']} | Loss:{stats['loss']} | Draw:{stats['draw']} | WinRate:{win_rate:.1f}% | DrawRate:{draw_rate:.1f}%")

if __name__ == "__main__":
    trades = load_trades("C:/War-Machine/war_machine_exhaustive.ndjson")
    trades = [normalize_conflicts(t) for t in trades]
    strategy_reason_analysis(trades)
    debug_counts(trades)
    strategy_summary(trades)
    print("\n=== Strategy-Aligned Trades (Match:true) ===\n")
    conflict_analysis(trades)
    indicator_analysis(trades)