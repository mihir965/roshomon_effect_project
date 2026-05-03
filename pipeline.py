"""CLI pipeline: build golden truth, query LLMs, compute metrics, print leaderboard.

Model selection accepts the form `provider:model_id`:
    openai:gpt-4o-mini
    anthropic:claude-haiku-4-5-20251001
    gemini:gemini-2.5-flash
    ollama:llama3:latest

Bare strings (e.g. `llama3`) are treated as Ollama tags for backward compatibility.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from config import DEFAULT_WEIGHTS, T_RUNS, GOLDEN_TRUTH_PATH
from llm.model_registry import parse_model_spec, model_exists


def get_llm(spec: str):
    provider, model_id = parse_model_spec(spec)
    if provider == "openai":
        from llm.openai_llm import OpenAILLM
        return OpenAILLM(model_name=model_id)
    if provider == "anthropic":
        from llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(model_name=model_id)
    if provider == "gemini":
        from llm.gemini_llm import GeminiLLM
        return GeminiLLM(model_name=model_id)
    from llm.ollama_llm import OllamaLLM
    return OllamaLLM(model_name=model_id)


def merge_results(existing: list[dict], new_runs: list[dict]) -> list[dict]:
    """Replace any prior entry that shares model_name; keep all others."""
    new_names = {r["model_name"] for r in new_runs}
    kept = [r for r in existing if r["model_name"] not in new_names]
    return kept + new_runs


def main():
    parser = argparse.ArgumentParser(description="LLM Reasoning Evaluation Pipeline")
    parser.add_argument(
        "--models", nargs="+", default=["ollama:llama3:latest"],
        help="Model specs in 'provider:model_id' form (e.g. anthropic:claude-haiku-4-5-20251001). "
             "Bare strings are treated as Ollama tags.",
    )
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--t_runs", type=int, default=T_RUNS)
    parser.add_argument("--rebuild_golden", action="store_true",
                        help="Re-fetch dataset and rebuild golden_truth.json")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip pre-flight check that the model exists in the provider's catalog")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace results.json instead of merging by model_name")
    args = parser.parse_args()

    # ── Pre-flight validation ─────────────────────────────────────────────────
    if not args.no_validate:
        bad = []
        for spec in args.models:
            provider, model_id = parse_model_spec(spec)
            if not model_exists(provider, model_id):
                bad.append((spec, provider, model_id))
        if bad:
            print("Pre-flight validation failed:")
            for spec, provider, model_id in bad:
                print(f"  {spec!r} — '{model_id}' not found in {provider} catalog "
                      f"(or no API key / daemon not running)")
            print("Pass --no-validate to skip this check.")
            raise SystemExit(2)

    # ── Golden truth ──────────────────────────────────────────────────────────
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

    # ── Evaluation ────────────────────────────────────────────────────────────
    from evaluation.evaluator import evaluate_model, results_to_dict
    new_runs: list[dict] = []
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for spec in args.models:
        provider, model_id = parse_model_spec(spec)
        print(f"Evaluating {spec}  ({provider} / {model_id})...")
        llm = get_llm(spec)
        result = evaluate_model(llm, golden, DEFAULT_WEIGHTS, args.t_runs)
        d = results_to_dict(result)
        d["provider"] = provider
        d["timestamp"] = timestamp
        d["n_questions"] = len(golden)
        d["t_runs"] = args.t_runs
        new_runs.append(d)
        print(
            f"  FPS={result.mean_fps:.4f} | AAS={result.mean_aas:.4f} | "
            f"RAS={result.mean_ras:.4f} | SLMS={result.mean_slms:.4f} | "
            f"CS={result.mean_cs:.4f} | DKUS={result.mean_dkus:.4f}"
        )
        print(f"  cache: {result.cache_hits} hits / {result.cache_misses} misses\n")

    # ── Persist ───────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    if args.overwrite or not output_path.exists():
        all_results = new_runs
    else:
        with open(output_path) as f:
            existing = json.load(f)
        all_results = merge_results(existing, new_runs)

    all_results.sort(key=lambda r: r.get("mean_fps", 0), reverse=True)

    print("=== LEADERBOARD ===")
    for rank, r in enumerate(all_results, 1):
        print(f"  {rank}. {r['model_name']:40s}  FPS={r.get('mean_fps', 0):.4f}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
