# 📊 LLM Evaluation & Observability Platform

Evaluates a RAG app (PDF book → Chroma → Gemini) for **quality, latency, tokens and cost**.
Every experiment is saved in **PostgreSQL**, every call is traced in **Langfuse**, and results are available through a **FastAPI** API and a **Streamlit** dashboard.

## Features
- **Automated RAG evaluation** with RAGAS: faithfulness, answer relevancy, context precision and context recall
- **Experiment tracking**: compare models, prompt versions, chunk sizes and retrieval depth (`k`)
- **Regression checks**: compare a candidate run against a baseline and fail if quality drops
- **Observability**: every question is traced in Langfuse, with scores attached to each trace
- **Failure analysis**: hallucinations, errors and low-recall questions, each linked to its trace
- **Dashboard and API**: browse runs in Streamlit, or use the FastAPI endpoints

## Project Structure
| File | Job |
|---|---|
| `config.py` | env, settings, price table, shared helpers |
| `database.py` | PostgreSQL tables: `test_cases`, `runs`, `run_items` |
| `rag_app.py` | the app under test (prompt version, chunk size, k and model are configurable) |
| `evaluator.py` | RAGAS scoring: faithfulness, answer relevancy, context precision, context recall |
| `runner.py` | runs one experiment end to end |
| `regression.py` | baseline vs candidate check (exit code 1 on regression) |
| `seed_dataset.py` | loads `data/test_cases.json` into PostgreSQL and Langfuse |
| `api.py` | FastAPI endpoints |
| `dashboard.py` | Streamlit UI: runs, comparison, failures and starting new runs |
| `data/test_cases.json` | test questions and ground-truth answers |
| `docker-compose.yml`, `Dockerfile` | PostgreSQL and the API as containers |
| `.env.example` | template for the keys and settings your `.env` needs |

## Setup (Windows + uv)
```
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and fill in your `GOOGLE_API_KEY` and Langfuse keys.
2. Put your book at `data/DL.pdf`. The book is not included in this repository.
3. Start PostgreSQL with Docker. It runs on **port 5433**, so it doesn't clash with a local PostgreSQL on port 5432:
   ```
   docker compose up db -d
   ```
   Check that it shows `healthy` with `docker compose ps`.

## Run
Seed the test questions, then run experiments:
```
python seed_dataset.py
python runner.py baseline-v1 dl-book-qa gemini-3.5-flash-lite v1 800 4
python runner.py k8-v1 dl-book-qa gemini-3.5-flash-lite v1 800 8
python regression.py 2 3
```

The runner takes these arguments in order:
```
python runner.py <run_name> <dataset> <model> <prompt_version> <chunk_size> <k>
```
- `run_name`: any label for the experiment, such as `baseline-v1`
- `dataset`: the dataset you seeded, `dl-book-qa`
- `model`: the Gemini model, `gemini-3.5-flash-lite`
- `prompt_version`: `v1` or `v2`
- `chunk_size`: size of each text chunk, `800`
- `k`: number of chunks retrieved per question, such as `4` or `8`

Each run prints an ID. Use those IDs with `regression.py <baseline_id> <candidate_id>`.

### Dashboard
```
streamlit run dashboard.py
```
Open http://localhost:8501. The dashboard has four pages:
- **Runs:** the latest scores, a table of all runs, and charts for quality, cost and latency
- **Compare:** pick a baseline and a candidate to see the regression verdict and per-metric changes
- **Failures:** flagged questions for a run, with answers, ground truths, retrieved chunks and Langfuse trace links
- **New Run:** start an experiment from the browser

### API
```
uvicorn api:app --reload
```
Open http://localhost:8000/docs to try the endpoints.

### Everything in Docker
```
docker compose up --build
```
This starts PostgreSQL and the API. The dashboard runs locally with `streamlit run`.

## Endpoints
`GET /health` · `GET /datasets` · `GET/POST /datasets/{name}/cases` · `POST /runs` · `GET /runs` ·
`GET /runs/{id}` · `GET /runs/{id}/items` · `GET /runs/{id}/failures` · `GET /compare?baseline_id=2&candidate_id=3`

## Results
Two runs on the 6 sample questions, same model and prompt, with the only change being retrieval depth:

| Metric | k=4 | k=8 |
|---|---|---|
| Faithfulness | 0.933 | 0.938 |
| Answer relevancy | 0.592 | 0.749 |
| Context precision | 0.333 | 0.329 |
| Context recall | 0.000 | 0.333 |
| Cost | $0.00072 | $0.00122 |

Retrieving 8 chunks improved answer relevancy and context recall, at about 70% higher cost. The regression check (`regression.py 2 3`) passes.

**Caveat:** the ground-truth answers in `data/test_cases.json` are written in my own words, not copied from the book, so context recall is understated. Replacing them with passages from `DL.pdf` is the next improvement.

## Notes
- Prices in `config.py` are placeholders. Update `PRICE_INPUT_PER_1M` and `PRICE_OUTPUT_PER_1M` for your model.
- An item with faithfulness below `HALLUCINATION_THRESHOLD` (0.5) counts as a hallucination.
- Each item stores a Langfuse `trace_url`, so you can open the exact trace of a failure.
- The first run in a session is slower because it loads the embedding model and index, so compare latency only between runs that started warm.