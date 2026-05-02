"""CLI pipeline: build golden truth, query LLMs, compute metrics, print leaderboard."""

import argparse
import json
from pathlib import Path

from config import DEFAULT_WEIGHTS, T_RUNS, GOLDEN_TRUTH_PATH, HF_MODELS


def get_llm(name: str):
    if name == "openai":
        from llm.openai_llm import OpenAILLM
        return OpenAILLM()
    if name == "anthropic":
        from llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM()
    if name == "gemini":
        from llm.gemini_llm import GeminiLLM
        return GeminiLLM()
    if name in HF_MODELS:
        from llm.hf_llm import HuggingFaceLLM
        return HuggingFaceLLM(HF_MODELS[name])
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="LLM Reasoning Evaluation Pipeline")
    parser.add_argument("--models", nargs="+", default=["openai"],
                        choices=["openai", "anthropic", "gemini", *HF_MODELS.keys()])
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--t_runs", type=int, default=T_RUNS)
    parser.add_argument("--rebuild_golden", action="store_true",
                        help="Re-fetch dataset and rebuild golden_truth.json")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    golden_path = Path(GOLDEN_TRUTH_PATH)
    if args.rebuild_golden or not golden_path.exists():
        from data.dataset_loader import load_hf_dataset, dataset_to_golden_truth, save_golden_truth
        print("Building golden truth dataset...")
        ds = load_hf_dataset(n_samples=args.n_questions)
        records = dataset_to_golden_truth(ds, n_samples=args.n_questions)
        save_golden_truth(records)
        print(f"Saved {len(records)} records → {golden_path}")

    from data.dataset_loader import load_golden_truth
    golden = load_golden_truth()[: args.n_questions]
    print(f"Loaded {len(golden)} golden truth records.\n")

    from evaluation.evaluator import evaluate_model, results_to_dict
    all_results = []

    for model_name in args.models:
        print(f"Evaluating {model_name}...")
        llm = get_llm(model_name)
        result = evaluate_model(llm, golden, DEFAULT_WEIGHTS, args.t_runs)
        d = results_to_dict(result)
        all_results.append(d)
        print(
            f"  FPS={result.mean_fps:.4f} | AAS={result.mean_aas:.4f} | "
            f"RAS={result.mean_ras:.4f} | SLMS={result.mean_slms:.4f} | "
            f"CS={result.mean_cs:.4f} | DKUS={result.mean_dkus:.4f}\n"
        )

    all_results.sort(key=lambda r: r["mean_fps"], reverse=True)

    print("=== LEADERBOARD ===")
    for rank, r in enumerate(all_results, 1):
        print(f"  {rank}. {r['model_name']:30s}  FPS={r['mean_fps']:.4f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
