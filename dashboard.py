#Dashboard - Streamlit UI for the LLM Evaluation & Observability Platform
from config import *
from database import SessionLocal, Run, RunItem, TestCase, init_db
from regression import compare_runs

# Importing Libraries
import threading
import pandas as pd
import streamlit as st
from runner import create_run, execute_run

st.set_page_config(page_title="LLM Eval Platform", page_icon="📊", layout="wide")
init_db()

QUALITY = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RUN_COLS = ["id", "run_name", "dataset_name", "status", "model", "prompt_version", "chunk_size", "k"] + QUALITY + [
    "avg_latency", "total_input_tokens", "total_output_tokens", "total_cost", "hallucination_count", "failed_count",
]

# 🧩Helpers
def fmt(v):
    return "-" if v is None or pd.isna(v) else f"{v:.3f}"

# 🗄️Data loaders (cached for 30s; the Refresh button clears the cache)
@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    with SessionLocal() as db:
        rows = db.query(Run).order_by(Run.id.desc()).all()
        data = [{col: getattr(r, col) for col in RUN_COLS} for r in rows]
    return pd.DataFrame(data, columns=RUN_COLS)

@st.cache_data(ttl=30)
def load_items(run_id: int) -> pd.DataFrame:
    with SessionLocal() as db:
        rows = db.query(RunItem).filter(RunItem.run_id == run_id).all()
        data = [{
            "id": i.id, "question": i.question, "ground_truth": i.ground_truth, "answer": i.answer,
            "contexts": i.contexts, "latency": i.latency, "cost": i.cost,
            "faithfulness": i.faithfulness, "answer_relevancy": i.answer_relevancy,
            "context_precision": i.context_precision, "context_recall": i.context_recall,
            "is_hallucination": bool(i.is_hallucination), "error": i.error, "trace_url": i.trace_url,
        } for i in rows]
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def load_datasets() -> list:
    with SessionLocal() as db:
        return [r[0] for r in db.query(TestCase.dataset_name).distinct().all()]

# 🧪Page 1 - all runs and the latest scores
def page_runs():
    st.header("🧪 Experiment runs")
    df = load_runs()
    if df.empty:
        st.info("No runs yet. Start one on the New Run page or with runner.py.")
        return

    completed = df[df["status"] == "completed"]
    if not completed.empty:
        latest = completed.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness", fmt(latest["faithfulness"]))
        c2.metric("Answer relevancy", fmt(latest["answer_relevancy"]))
        c3.metric("Context precision", fmt(latest["context_precision"]))
        c4.metric("Context recall", fmt(latest["context_recall"]))
        st.caption(f"Latest completed run: {latest['run_name']} (ID {latest['id']})")

    st.dataframe(df, use_container_width=True, hide_index=True)

    if not completed.empty:
        st.subheader("Quality by run")
        st.bar_chart(completed.set_index("run_name")[QUALITY])

        st.subheader("Cost and latency by run")
        st.bar_chart(completed.set_index("run_name")[["total_cost"]])
        st.bar_chart(completed.set_index("run_name")[["avg_latency"]])

# ⚖️Page 2 - regression check between two runs
def page_compare():
    st.header("⚖️ Compare two runs")
    df = load_runs()
    completed = df[df["status"] == "completed"]
    if len(completed) < 2:
        st.info("You need at least two completed runs to compare.")
        return

    labels = {int(r.id): f"{r.id} - {r.run_name}" for r in completed.itertuples()}
    ids = list(labels.keys())

    c1, c2, c3 = st.columns([2, 2, 1])
    baseline_id = c1.selectbox("Baseline", ids, format_func=lambda i: labels[i], index=min(1, len(ids) - 1))
    candidate_id = c2.selectbox("Candidate", ids, format_func=lambda i: labels[i], index=0)
    tolerance = c3.number_input("Tolerance", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    if baseline_id == candidate_id:
        st.warning("Pick two different runs.")
        return

    result = compare_runs(int(baseline_id), int(candidate_id), float(tolerance))

    if result["passed"]:
        st.success("✅ Regression check PASSED")
    else:
        st.error("❌ Regression check FAILED")

    table = pd.DataFrame([{
        "metric": name,
        "baseline": m["baseline"],
        "candidate": m["candidate"],
        "delta": m["delta"],
        "status": "❌ regressed" if m["regressed"] else "✅ ok",
    } for name, m in result["quality"].items()])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.bar_chart(table.set_index("metric")[["baseline", "candidate"]])

    c1, c2, c3 = st.columns(3)
    c1.metric("Latency change", f"{result['latency_change_pct']}%")
    c2.metric("Cost change", f"{result['cost_change_pct']}%")
    c3.metric("Hallucinations (baseline → candidate)",
              f"{result['hallucinations']['baseline']} → {result['hallucinations']['candidate']}")

# 🚨Page 3 - failed, hallucinated or low-recall questions
def page_failures():
    st.header("🚨 Failures and hallucinations")
    df = load_runs()
    if df.empty:
        st.info("No runs yet.")
        return

    run_names = dict(zip(df["id"], df["run_name"]))
    run_id = st.selectbox("Run", list(run_names.keys()), format_func=lambda i: f"{i} - {run_names[i]}")

    items = load_items(int(run_id))
    if items.empty:
        st.info("This run has no items.")
        return

    flagged = items[
        items["error"].notna()
        | items["is_hallucination"]
        | (items["context_recall"].fillna(1) < 0.5)
    ]
    st.write(f"{len(flagged)} of {len(items)} questions flagged")

    for _, row in flagged.iterrows():
        with st.expander(row["question"]):
            if isinstance(row["error"], str):
                st.error(row["error"])
            st.markdown(f"**Answer:** {row['answer']}")
            st.markdown(f"**Ground truth:** {row['ground_truth']}")
            st.write({m: fmt(row[m]) for m in QUALITY})
            if isinstance(row["trace_url"], str):
                st.markdown(f"[Open trace in Langfuse]({row['trace_url']})")
            if isinstance(row["contexts"], list):
                st.caption("Retrieved contexts")
                for ctx in row["contexts"]:
                    st.text(ctx[:300] + ("..." if len(ctx) > 300 else ""))

# ▶️Page 4 - start a run from the browser
def page_new_run():
    st.header("▶️ Start a new run")
    datasets = load_datasets()
    if not datasets:
        st.warning("No datasets yet. Run seed_dataset.py first.")
        return

    with st.form("new_run"):
        run_name = st.text_input("Run name", value="experiment-1")
        dataset_name = st.selectbox("Dataset", datasets)
        model = st.text_input("Model", value=DEFAULT_MODEL)
        prompt_version = st.selectbox("Prompt version", ["v1", "v2"])
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=3000, value=800, step=100)
        k = st.number_input("k (chunks retrieved)", min_value=1, max_value=20, value=4)
        submitted = st.form_submit_button("Start run")

    if submitted:
        run_id = create_run(run_name, dataset_name, model, prompt_version, int(chunk_size), int(k))
        # Run in a background thread so the page doesn't freeze while Gemini and RAGAS work
        threading.Thread(target=execute_run, args=(run_id,), daemon=True).start()
        st.cache_data.clear()
        st.success(f"Run {run_id} started. Go to Runs and click Refresh data to see progress.")

# 🧭Navigation
pages = {"Runs": page_runs, "Compare": page_compare, "Failures": page_failures, "New Run": page_new_run}
choice = st.sidebar.radio("Page", list(pages.keys()))
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

pages[choice]()