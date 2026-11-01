#Config - shared settings for the LLM Evaluation & Observability Platform
import os
import logging
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("langfuse").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ⚙️Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/llm_eval")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PDF_PATH = os.getenv("PDF_PATH", "data/DL.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# 🚨A run item with faithfulness below this is flagged as a hallucination
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.5"))

# 💰Price per 1M tokens in USD (placeholder values, update from the Google AI pricing page)
PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.10"))
PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.40"))

# 🧩Small helpers
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings

def get_text(msg) -> str:
    return msg.text

def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_INPUT_PER_1M + output_tokens * PRICE_OUTPUT_PER_1M) / 1_000_000
