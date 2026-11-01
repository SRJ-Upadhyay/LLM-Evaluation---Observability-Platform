#API - FastAPI layer for the LLM Evaluation & Observability Platform
from config import *
from database import SessionLocal, Run, RunItem, TestCase, init_db
from runner import create_run, execute_run
from regression import compare_runs

# Importing Libraries
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_

app = FastAPI(title="LLM Evaluation & Observability Platform")

@app.on_event("startup")
def startup():
    init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 📦Request models
class TestCaseIn(BaseModel):
    question : str
    ground_truth : str

class RunIn(BaseModel):
    run_name : str
    dataset_name : str = "dl-book-qa"
    model : str = DEFAULT_MODEL
    prompt_version : str = "v1"
    chunk_size : int = 800
    k : int = 4

# 🧩Helpers
def run_to_dict(r: Run) -> dict:
    return {
        "id": r.id, "run_name": r.run_name, "dataset_name": r.dataset_name, "status": r.status,
        "model": r.model, "prompt_version": r.prompt_version, "chunk_size": r.chunk_size, "k": r.k,
        "faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
        "context_precision": r.context_precision, "context_recall": r.context_recall,
        "avg_latency": r.avg_latency, "total_input_tokens": r.total_input_tokens,
        "total_output_tokens": r.total_output_tokens, "total_cost": r.total_cost,
        "hallucination_count": r.hallucination_count, "failed_count": r.failed_count,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }

def item_to_dict(i: RunItem) -> dict:
    return {
        "id": i.id, "question": i.question, "ground_truth": i.ground_truth, "answer": i.answer,
        "contexts": i.contexts, "latency": i.latency, "input_tokens": i.input_tokens,
        "output_tokens": i.output_tokens, "cost": i.cost, "faithfulness": i.faithfulness,
        "answer_relevancy": i.answer_relevancy, "context_precision": i.context_precision,
        "context_recall": i.context_recall, "is_hallucination": bool(i.is_hallucination),
        "error": i.error, "trace_id": i.trace_id, "trace_url": i.trace_url,
    }

# ❤️Health
@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL}

# 📚Datasets
@app.get("/datasets")
def list_datasets(db=Depends(get_db)):
    rows = db.query(TestCase.dataset_name).distinct().all()
    return [r[0] for r in rows]

@app.get("/datasets/{name}/cases")
def list_cases(name: str, db=Depends(get_db)):
    cases = db.query(TestCase).filter(TestCase.dataset_name == name).all()
    return [{"id": c.id, "question": c.question, "ground_truth": c.ground_truth} for c in cases]

@app.post("/datasets/{name}/cases")
def add_case(name: str, body: TestCaseIn, db=Depends(get_db)):
    case = TestCase(dataset_name=name, question=body.question, ground_truth=body.ground_truth)
    db.add(case)
    db.commit()
    return {"id": case.id}

# 🧪Runs
@app.post("/runs")
def start_run(body: RunIn, background: BackgroundTasks):
    run_id = create_run(body.run_name, body.dataset_name, body.model, body.prompt_version, body.chunk_size, body.k)
    background.add_task(execute_run, run_id)
    return {"run_id": run_id, "status": "running"}

@app.get("/runs")
def list_runs(dataset_name: str | None = None, db=Depends(get_db)):
    q = db.query(Run)
    if dataset_name:
        q = q.filter(Run.dataset_name == dataset_name)
    return [run_to_dict(r) for r in q.order_by(Run.id.desc()).all()]

@app.get("/runs/{run_id}")
def get_run(run_id: int, db=Depends(get_db)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run_to_dict(run)

@app.get("/runs/{run_id}/items")
def get_items(run_id: int, db=Depends(get_db)):
    items = db.query(RunItem).filter(RunItem.run_id == run_id).all()
    return [item_to_dict(i) for i in items]

# 🚨Failures + hallucinations (each item has a Langfuse trace_url to investigate)
@app.get("/runs/{run_id}/failures")
def get_failures(run_id: int, db=Depends(get_db)):
    items = db.query(RunItem).filter(
        RunItem.run_id == run_id,
        or_(RunItem.error.isnot(None), RunItem.is_hallucination == 1, RunItem.context_recall < 0.5),
    ).all()
    return [item_to_dict(i) for i in items]

# ⚖️Compare two runs (regression check)
@app.get("/compare")
def compare(baseline_id: int, candidate_id: int, tolerance: float = 0.05):
    try:
        return compare_runs(baseline_id, candidate_id, tolerance)
    except ValueError as e:
        raise HTTPException(404, str(e))
