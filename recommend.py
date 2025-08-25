
import argparse
import json
import os
import yaml


def load_models(models_yaml):
    """Load available LLM models metadata from YAML file."""
    with open(models_yaml, "r") as f:
        return yaml.safe_load(f)


def score_model(model, ctx):
    """Return score and breakdown for a model given user context."""
    score = 0
    breakdown = []

    # Match on domain sensitivity
    if ctx.get("safety_sensitive") and model.get("safety", False):
        score += 2
        breakdown.append("✅ Safety-critical supported (+2)")

    # Match on reasoning
    if ctx.get("needs_reasoning") and model.get("reasoning", False):
        score += 2
        breakdown.append("✅ Strong reasoning (+2)")

    # Multilingual support
    if ctx.get("multilingual") and model.get("multilingual", False):
        score += 1
        breakdown.append("✅ Multilingual support (+1)")

    # Cost sensitivity
    if ctx.get("cost_budget") == "low" and model.get("cost") == "low":
        score += 2
        breakdown.append("✅ Matches low cost budget (+2)")
    elif ctx.get("cost_budget") == model.get("cost"):
        score += 1
        breakdown.append("✅ Matches cost budget (+1)")

    # Latency budget
    if ctx.get("latency_budget") == "low" and model.get("latency") == "low":
        score += 2
        breakdown.append("✅ Matches low latency budget (+2)")
    elif ctx.get("latency_budget") == model.get("latency"):
        score += 1
        breakdown.append("✅ Matches latency budget (+1)")

    # Dataset size fit
    if ctx.get("dataset_size") == model.get("scale"):
        score += 2
        breakdown.append("✅ Matches dataset scale (+2)")

    return score, breakdown


def parse_ctx(arg):
    """Parse --ctx argument: either JSON string or JSON file path."""
    if not arg:
        raise ValueError("No context (--ctx) provided!")

    if os.path.isfile(arg):
        with open(arg, "r") as f:
            return json.load(f)

    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --ctx: {e}\nInput was: {arg}")


def recommend(models, ctx, top_k=3):
    """Return top-k recommended models with breakdown."""
    scored = []
    for m in models:
        score, breakdown = score_model(m, ctx)
        scored.append((score, m, breakdown))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-yaml", default="models.yaml")
    parser.add_argument("--ctx", required=True,
                        help="JSON string or path to ctx.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    ctx = parse_ctx(args.ctx)
    models = load_models(args.models_yaml)

    top_models = recommend(models, ctx, args.top_k)

    print("\n=== Recommendation Results ===")
    for rank, (score, model, breakdown) in enumerate(top_models, start=1):
        print(f"\n{rank}. {model['name']} (score={score})")
        for b in breakdown:
            print("   -", b)
    print("==============================")


if __name__ == "__main__":
    main()
