#Evaluator - RAGAS quality scoring
from config import *
import math

# Importing Libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# 🧠Judge LLM (embeddings are loaded on first use so the API starts fast)
judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL))
judge_embeddings = None

# strictness=1 -> the judge asks Gemini for 1 candidate (Gemini rejects multiple candidates)
METRICS = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]

# RAGAS column name -> our column name
NAME_MAP = {
    "faithfulness" : "faithfulness",
    "answer_relevancy" : "answer_relevancy",
    "llm_context_precision_with_reference" : "context_precision",
    "context_recall" : "context_recall",
}

def clean(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except Exception:
        return None

def score_rows(rows: list[dict]) -> list[dict]:
    """rows: [{question, answer, contexts, ground_truth}] -> list of metric dicts (same order)"""
    global judge_embeddings
    if judge_embeddings is None:
        judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    dataset = EvaluationDataset.from_list([
        {
            "user_input" : r["question"],
            "response" : r["answer"],
            "retrieved_contexts" : r["contexts"],
            "reference" : r["ground_truth"],
        }
        for r in rows
    ])

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=180),
        show_progress=False,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    scores = []
    for _, row in df.iterrows():
        scores.append({ours: clean(row.get(theirs)) for theirs, ours in NAME_MAP.items()})
    return scores
