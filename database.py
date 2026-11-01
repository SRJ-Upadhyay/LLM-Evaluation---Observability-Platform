#Database - PostgreSQL tables (SQLAlchemy)
from config import *
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


# 📚Test datasets
class TestCase(Base):
    __tablename__ = "test_cases"
    __test__ = False
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(100), index=True, nullable=False)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)


# 🧪One experiment = one model + prompt + retrieval config over a dataset
class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    run_name = Column(String(150), index=True, nullable=False)
    dataset_name = Column(String(100), index=True, nullable=False)
    status = Column(String(20), default="running")
    model = Column(String(100))
    prompt_version = Column(String(20))
    chunk_size = Column(Integer)
    k = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # quality
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_precision = Column(Float)
    context_recall = Column(Float)
    hallucination_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)

    # performance
    avg_latency = Column(Float)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)

    items = relationship("RunItem", back_populates="run", cascade="all, delete-orphan")


# 🔍Per-question result with its Langfuse trace
class RunItem(Base):
    __tablename__ = "run_items"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), index=True)
    question = Column(Text)
    ground_truth = Column(Text)
    answer = Column(Text)
    contexts = Column(JSON)
    latency = Column(Float)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_precision = Column(Float)
    context_recall = Column(Float)
    is_hallucination = Column(Integer, default=0)
    error = Column(Text)
    trace_id = Column(String(64))
    trace_url = Column(Text)

    run = relationship("Run", back_populates="items")


def init_db():
    Base.metadata.create_all(engine)
