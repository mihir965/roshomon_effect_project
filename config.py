import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
}

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"

DATA_DIR = Path("./data")
GOLDEN_TRUTH_PATH = DATA_DIR / "golden_truth.json"

DATASET_NAME = "avemio/German-RAG-ORPO-Alpaca-HESSIAN-AI"
DATASET_CONFIG = "hard-reasoning-en"  # subset/config name
DATASET_SPLIT = "train"               # actual split within that config

DEFAULT_WEIGHTS = {
    "w_aas": 0.20,
    "w_ras": 0.30,
    "w_slms": 0.25,
    "w_cs": 0.10,
    "w_dkus": 0.15,
}

T_RUNS = 3  # repeated runs per question for Consistency Score

COT_SYSTEM_PROMPT = """You are a precise reasoning assistant. For each question, think through it carefully step by step, then state your final answer.

Format your response EXACTLY as:
REASONING:
Step 1: <first reasoning step>
Step 2: <second reasoning step>
Step 3: <third reasoning step>
...
FINAL ANSWER: <concise final answer>"""
