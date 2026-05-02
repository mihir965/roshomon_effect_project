import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-1.5-pro",
}

HF_MODELS = {
    "hf-llama": "meta-llama/Llama-3.2-1B-Instruct",
    "hf-qwen":  "Qwen/Qwen2.5-7B-Instruct",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"

DATA_DIR = Path("./data")
GOLDEN_TRUTH_PATH = DATA_DIR / "golden_truth.json"

DATASET_NAME = "avemio/German-RAG-ORPO-Alpaca-HESSIAN-AI"
DATASET_SPLIT = "hard-reasoning-en"

DEFAULT_WEIGHTS = {
    "w_aas": 0.20,
    "w_ras": 0.30,
    "w_slms": 0.25,
    "w_cs": 0.10,
    "w_dkus": 0.15,
}

T_RUNS = 1  # repeated runs per question for CS; set >1 to get a real variance signal

COT_SYSTEM_PROMPT = """You are a precise reasoning assistant. For each question, think through it carefully step by step, then state your final answer.

Format your response EXACTLY as:
REASONING:
Step 1: <first reasoning step>
Step 2: <second reasoning step>
Step 3: <third reasoning step>
...
FINAL ANSWER: <concise final answer>"""
