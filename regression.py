#Regression - compare a new run against a baseline run
from config import *
from database import SessionLocal, Run

# Importing Libraries
import sys
from rich import print
from rich.table import Table

QUALITY = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

def compare_runs(baseline_id: int, candidate_id: int, tolerance: float = 0.05) -> dict:
    with SessionLocal() as db:
        base = db.get(Run, baseline_id)
        cand = db.get(Run, candidate_id)

    if base is None or cand is None:
        raise ValueError("Baseline or candidate run not found")

    metrics = {}
    passed = True
    for name in QUALITY:
        b, c = getattr(base, name), getattr(cand, name)
        if b is None or c is None:
            metrics[name] = {"baseline": b, "candidate": c, "delta": None, "regressed": False}
            continue
        delta = c - b
        regressed = delta < -tolerance
        passed = passed and not regressed
        metrics[name] = {"baseline": round(b, 4), "candidate": round(c, 4), "delta": round(delta, 4), "regressed": regressed}

    def pct(b, c):
        return round((c - b) / b * 100, 2) if b else None

    return {
        "baseline_id": base.id,
        "candidate_id": cand.id,
        "tolerance": tolerance,
        "passed": passed,
        "quality": metrics,
        "latency_change_pct": pct(base.avg_latency, cand.avg_latency),
        "cost_change_pct": pct(base.total_cost, cand.total_cost),
        "hallucinations": {"baseline": base.hallucination_count, "candidate": cand.hallucination_count},
    }

# 💻CLI: python regression.py <baseline_id> <candidate_id> [tolerance]
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python regression.py <baseline_id> <candidate_id> [tolerance]")
        sys.exit(1)

    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    result = compare_runs(int(sys.argv[1]), int(sys.argv[2]), tol)

    table = Table(title=f"Run {result['baseline_id']} (baseline) vs Run {result['candidate_id']} (candidate)")
    for col in ["Metric", "Baseline", "Candidate", "Delta", "Status"]:
        table.add_column(col)
    for name, m in result["quality"].items():
        table.add_row(name, str(m["baseline"]), str(m["candidate"]), str(m["delta"]), "❌ REGRESSED" if m["regressed"] else "✅ ok")
    print(table)
    print(f"latency change: {result['latency_change_pct']}%  |  cost change: {result['cost_change_pct']}%  |  hallucinations: {result['hallucinations']}")

    if result["passed"]:
        print("[bold green]✅ Regression check PASSED[/bold green]")
    else:
        print("[bold red]❌ Regression check FAILED[/bold red]")
        sys.exit(1)
