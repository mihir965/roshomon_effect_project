"""Load the HuggingFace dataset and convert to the golden truth JSON schema."""

import json
import re
import sys
from pathlib import Path

# Allow running this file directly (python data/dataset_loader.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, GOLDEN_TRUTH_PATH


def load_hf_dataset(n_samples: int = None):
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def _detect_columns(ds) -> tuple[str, str]:
    cols = ds.column_names
    cols_lower = {c.lower(): c for c in cols}  # lowercase -> original

    question_col = None
    for candidate in ("instruction", "prompt", "question", "input"):
        if candidate in cols_lower:
            question_col = cols_lower[candidate]
            break
    if question_col is None:
        question_col = cols[0]

    answer_col = None
    for candidate in ("chosen", "output", "answer", "response"):
        if candidate in cols_lower:
            answer_col = cols_lower[candidate]
            break
    if answer_col is None:
        answer_col = cols[1] if len(cols) > 1 else cols[0]

    return question_col, answer_col


def _flatten(value) -> str:
    """Normalise chat-format or list values to a plain string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content", str(value))
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(item.get("content", str(item)))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def _parse_cot_response(text: str) -> tuple[list[str], str]:
    """Extract reasoning steps and final answer from a CoT-formatted string."""
    lines = text.strip().splitlines()
    steps: list[str] = []
    final_answer = ""
    in_reasoning = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        if lower.startswith("final answer:") or lower.startswith("answer:"):
            final_answer = stripped.split(":", 1)[-1].strip()
            in_reasoning = False
            continue

        if lower.startswith("reasoning:"):
            in_reasoning = True
            continue

        if in_reasoning or lower.startswith("step "):
            if lower.startswith("step ") and ":" in stripped:
                steps.append(stripped.split(":", 1)[-1].strip())
            elif re.match(r"^\d+[.)]\s", stripped):
                steps.append(re.sub(r"^\d+[.)]\s+", "", stripped))
            elif in_reasoning:
                steps.append(stripped)

    if not steps:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        steps = [s for s in sentences if len(s) > 20][:6]

    if not final_answer and lines:
        final_answer = lines[-1].strip()

    return steps, final_answer


def _extract_key_concepts(steps: list[str]) -> list[str]:
    stopwords = {
        "which", "where", "their", "there", "would", "could", "should",
        "about", "these", "those", "other", "first", "second", "third",
        "using", "since", "after", "before", "while", "until", "between",
    }
    concepts: set[str] = set()
    for step in steps:
        words = re.findall(r"\b[A-Za-z]{5,}\b", step)
        concepts.update(w.lower() for w in words if w.lower() not in stopwords)
    return list(concepts)[:10]


def dataset_to_golden_truth(ds, domain: str = "hard-reasoning", n_samples: int = None) -> list[dict]:
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    question_col, answer_col = _detect_columns(ds)
    records: list[dict] = []

    for idx, row in enumerate(ds):
        question = _flatten(row[question_col])
        raw_answer = _flatten(row[answer_col])
        steps, final_answer = _parse_cot_response(raw_answer)

        records.append({
            "question_id": f"q_{idx:04d}",
            "task_domain": domain,
            "difficulty_level": "hard",
            "question": question,
            "golden_reasoning_steps": [
                {"step_number": i + 1, "description": step}
                for i, step in enumerate(steps)
            ],
            "golden_final_answer": final_answer,
            "expected_knowledge_domain": domain,
            "reasoning_type": "multi-step",
            "evaluation_criteria": {
                "must_include_concepts": _extract_key_concepts(steps),
                "common_failure_modes": [],
                "logical_dependencies": [],
            },
        })

    return records


def save_golden_truth(records: list[dict], path=None) -> Path:
    path = Path(path or GOLDEN_TRUTH_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    return path


def load_golden_truth(path=None) -> list[dict]:
    with open(Path(path or GOLDEN_TRUTH_PATH)) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loading dataset...")
    ds = load_hf_dataset(n_samples=5)
    print(f"Columns: {ds.column_names}")
    print(f"Sample row keys: {list(ds[0].keys())}")

    records = dataset_to_golden_truth(ds, n_samples=5)
    path = save_golden_truth(records)
    print(f"Saved {len(records)} golden truth records → {path}")
    print(json.dumps(records[0], indent=2))
