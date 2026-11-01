#Seed Dataset - load test cases into PostgreSQL and Langfuse
from config import *
from database import SessionLocal, TestCase, init_db

# Importing Libraries
import json
import sys
from rich import print
from langfuse import get_client

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "dl-book-qa"
file_path = sys.argv[2] if len(sys.argv) > 2 else "data/test_cases.json"

init_db()
with open(file_path, "r", encoding="utf-8") as f:
    cases = json.load(f)

# 🗄️PostgreSQL
added = 0
with SessionLocal() as db:
    existing = {c.question for c in db.query(TestCase).filter(TestCase.dataset_name == dataset_name).all()}
    for c in cases:
        if c["question"] in existing:
            continue
        db.add(TestCase(dataset_name=dataset_name, question=c["question"], ground_truth=c["ground_truth"]))
        added += 1
    db.commit()
print(f"[green]🗄️ PostgreSQL: added {added} new test cases to '{dataset_name}'[/green]")

# 🔭Langfuse dataset (so the same questions show up in the Langfuse UI)
langfuse = get_client()
try:
    langfuse.create_dataset(name=dataset_name)
except Exception:
    pass  # dataset already exists

for c in cases:
    langfuse.create_dataset_item(dataset_name=dataset_name, input={"question": c["question"]}, expected_output={"ground_truth": c["ground_truth"]})
langfuse.flush()
print(f"[green]🔭 Langfuse: synced dataset '{dataset_name}'[/green]")
