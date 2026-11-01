#Runner - run one experiment: dataset -> RAG app -> RAGAS -> PostgreSQL + Langfuse
from config import *
from database import SessionLocal, Run, RunItem, TestCase, init_db
from rag_app import answer_question
from evaluator import score_rows

# Importing Libraries
import sys
from rich import print
from langfuse import get_client, propagate_attributes

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None

# 🆕Step 1 - create the run row (status = running)
def create_run(run_name: str, dataset_name: str, model: str = DEFAULT_MODEL, prompt_version: str = "v1", chunk_size: int = 800, k: int = 4) -> int:
    init_db()
    with SessionLocal() as db:
        run = Run(run_name=run_name, dataset_name=dataset_name, model=model, prompt_version=prompt_version, chunk_size=chunk_size, k=k, status="running")
        db.add(run)
        db.commit()
        return run.id

# ▶️Step 2 - execute it
def execute_run(run_id: int):
    langfuse = get_client()

    with SessionLocal() as db:
        run = db.get(Run, run_id)
        cases = db.query(TestCase).filter(TestCase.dataset_name == run.dataset_name).all()

        if not cases:
            run.status = "failed"
            db.commit()
            print(f"[red]No test cases found in dataset '{run.dataset_name}'. Run seed_dataset.py first.[/red]")
            return

        print(f"[bold cyan]🧪 Run {run.id}: {run.run_name}[/bold cyan] | model={run.model} prompt={run.prompt_version} chunk={run.chunk_size} k={run.k}")

        # 🔭Every trace of this run is grouped by session_id = run_name
        tags = [f"model:{run.model}", f"prompt:{run.prompt_version}", f"k:{run.k}", f"chunk:{run.chunk_size}"]
        items = []

        with propagate_attributes(session_id=run.run_name, tags=tags):
            for i, case in enumerate(cases, 1):
                item = RunItem(run_id=run.id, question=case.question, ground_truth=case.ground_truth)
                try:
                    out = answer_question(case.question, model=run.model, prompt_version=run.prompt_version, chunk_size=run.chunk_size, k=run.k)
                    item.answer = out["answer"]
                    item.contexts = out["contexts"]
                    item.latency = out["latency"]
                    item.input_tokens = out["input_tokens"]
                    item.output_tokens = out["output_tokens"]
                    item.cost = out["cost"]
                    item.trace_id = out["trace_id"]
                    item.trace_url = langfuse.get_trace_url(trace_id=out["trace_id"])
                    print(f"[{i}/{len(cases)}] ✅ {case.question[:60]} ({out['latency']:.1f}s)")
                except Exception as e:
                    item.error = str(e)
                    print(f"[{i}/{len(cases)}] ❌ {case.question[:60]} -> {e}")
                items.append(item)

        # 📏Step 3 - RAGAS scoring on the items that worked
        good = [it for it in items if it.error is None]
        if good:
            print("[bold yellow]📏 Scoring with RAGAS...[/bold yellow]")
            scores = score_rows([
                {"question": it.question, "answer": it.answer, "contexts": it.contexts, "ground_truth": it.ground_truth}
                for it in good
            ])
            for it, sc in zip(good, scores):
                for name in METRIC_NAMES:
                    setattr(it, name, sc[name])
                it.is_hallucination = int(it.faithfulness is not None and it.faithfulness < HALLUCINATION_THRESHOLD)

                # Push the scores to the Langfuse trace too
                for name in METRIC_NAMES:
                    if sc[name] is not None and it.trace_id:
                        langfuse.create_score(trace_id=it.trace_id, name=name, value=sc[name])

        # 📊Step 4 - run level numbers
        for name in METRIC_NAMES:
            setattr(run, name, avg([getattr(it, name) for it in good]))
        run.avg_latency = avg([it.latency for it in good])
        run.total_input_tokens = sum(it.input_tokens or 0 for it in items)
        run.total_output_tokens = sum(it.output_tokens or 0 for it in items)
        run.total_cost = sum(it.cost or 0 for it in items)
        run.hallucination_count = sum(it.is_hallucination or 0 for it in items)
        run.failed_count = len(items) - len(good)
        run.status = "completed"

        db.add_all(items)
        db.commit()
        langfuse.flush()

        print(f"[bold green]✅ Run {run.id} completed[/bold green]")
        print({name: (round(getattr(run, name), 3) if getattr(run, name) is not None else None) for name in METRIC_NAMES})
        print(f"latency={run.avg_latency or 0:.2f}s  tokens={run.total_input_tokens + run.total_output_tokens}  cost=${run.total_cost:.5f}  hallucinations={run.hallucination_count}")

# 💻CLI: python runner.py <run_name> [dataset] [model] [prompt_version] [chunk_size] [k]
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python runner.py <run_name> [dataset] [model] [prompt_version] [chunk_size] [k]")
        sys.exit(1)

    run_name = args[0]
    dataset_name = args[1] if len(args) > 1 else "dl-book-qa"
    model = args[2] if len(args) > 2 else DEFAULT_MODEL
    prompt_version = args[3] if len(args) > 3 else "v1"
    chunk_size = int(args[4]) if len(args) > 4 else 800
    k = int(args[5]) if len(args) > 5 else 4

    run_id = create_run(run_name, dataset_name, model, prompt_version, chunk_size, k)
    execute_run(run_id)
