#RAG App - the LLM application that we evaluate
from config import *
import os
import time

# Importing Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

# 📝Prompt versions (compare these in experiments)
PROMPTS = {
    "v1" : (
        "You are a helpful assistant. Answer the question using the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Give a clear, friendly, and precise answer."
    ),
    "v2" : (
        "You are a strict teaching assistant. Answer ONLY from the context below.\n"
        "If the context does not contain the answer, say: I don't know based on the book.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Give a short, precise answer in 3-4 sentences."
    ),
}

# 🔎Retriever (one per chunk_size + k, built once and cached)
_retrievers = {}

def build_retriever(chunk_size: int, k: int):
    persist_dir = f"{CHROMA_DIR}/cs{chunk_size}"
    embeddings = get_embeddings()

    if os.path.exists(persist_dir):
        db = Chroma(collection_name=f"book_cs{chunk_size}", embedding_function=embeddings, persist_directory=persist_dir)
    else:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Put your book in the data folder.")
        docs = PyPDFLoader(PDF_PATH).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        db = Chroma.from_documents(chunks, embeddings, collection_name=f"book_cs{chunk_size}", persist_directory=persist_dir)

    return db.as_retriever(search_kwargs={"k": k})

def get_retriever(chunk_size: int, k: int):
    key = (chunk_size, k)
    if key not in _retrievers:
        _retrievers[key] = build_retriever(chunk_size, k)
    return _retrievers[key]

# 🧠LLMs (one per model name)
_llms = {}

def get_llm(model: str):
    if model not in _llms:
        _llms[model] = ChatGoogleGenerativeAI(model=model)
    return _llms[model]

# 🚀Answer a question and trace everything in Langfuse
@observe(name="rag_answer")
def answer_question(question: str, model: str = DEFAULT_MODEL, prompt_version: str = "v1", chunk_size: int = 800, k: int = 4) -> dict:
    start = time.time()

    retriever = get_retriever(chunk_size, k)
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]

    prompt = PROMPTS[prompt_version].format(context="\n\n".join(contexts), question=question)

    handler = CallbackHandler()
    response = get_llm(model).invoke(prompt, config={"callbacks": [handler]})
    latency = time.time() - start

    usage = response.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "answer" : get_text(response),
        "contexts" : contexts,
        "latency" : latency,
        "input_tokens" : input_tokens,
        "output_tokens" : output_tokens,
        "cost" : calc_cost(input_tokens, output_tokens),
        "trace_id" : get_client().get_current_trace_id(),
    }
