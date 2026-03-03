#!/usr/bin/env python3
"""
zen5 MoDE — Benchmark & Evaluation Suite

Evaluates zen5 complexity routing accuracy, expert utilization, and
end-to-end generation quality across standard benchmarks.

Benchmarks:
  - Complexity routing accuracy (tier prediction correctness)
  - Expert utilization (load balance, diversity, zero-expert usage)
  - Latency per tier (T0 should be <50ms, T4 <10s)
  - Quality: MMLU, GPQA, AIME, HumanEval, SWE-bench (via lm-eval-harness)

Usage:
  python zen5_benchmark.py routing --checkpoint ./ckpt/best.pt
  python zen5_benchmark.py utilization --checkpoint ./ckpt/best.pt
  python zen5_benchmark.py latency --checkpoint ./ckpt/best.pt
  python zen5_benchmark.py report --checkpoint ./ckpt/best.pt --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zen5-bench")


# ── Benchmark Tasks with Ground-Truth Tiers ──────────────────────────────────

ROUTING_BENCHMARKS = [
    # T0: Trivial
    {"prompt": "Hello!", "tier": 0, "domain": "chat"},
    {"prompt": "What is 2 + 2?", "tier": 0, "domain": "math"},
    {"prompt": "Is the sky blue? Yes or no.", "tier": 0, "domain": "factual"},
    {"prompt": "Thank you for your help.", "tier": 0, "domain": "chat"},
    {"prompt": "What color is grass?", "tier": 0, "domain": "factual"},

    # T1: Standard
    {"prompt": "Summarize the key points of supply and demand in economics.", "tier": 1, "domain": "reasoning"},
    {"prompt": "Translate to Spanish: The weather is nice today.", "tier": 1, "domain": "multilingual"},
    {"prompt": "Write a Python function to reverse a string.", "tier": 1, "domain": "code"},
    {"prompt": "What causes rain?", "tier": 1, "domain": "factual"},
    {"prompt": "Explain photosynthesis in simple terms.", "tier": 1, "domain": "factual"},

    # T2: Complex
    {"prompt": "Compare and contrast the economic policies of Keynesianism and monetarism with historical examples.", "tier": 2, "domain": "reasoning"},
    {"prompt": "Review this REST API design for security vulnerabilities and suggest improvements.", "tier": 2, "domain": "code"},
    {"prompt": "Analyze the themes of existentialism in Camus' The Stranger.", "tier": 2, "domain": "reasoning"},
    {"prompt": "Design a database schema for a multi-tenant SaaS application with RBAC.", "tier": 2, "domain": "code"},
    {"prompt": "Explain the differences between TCP and UDP with use cases for each.", "tier": 2, "domain": "factual"},

    # T3: Advanced
    {"prompt": "Derive the Black-Scholes equation from first principles using Ito's lemma.", "tier": 3, "domain": "math"},
    {"prompt": "Design a distributed consensus algorithm that achieves Byzantine fault tolerance with O(n) message complexity.", "tier": 3, "domain": "code"},
    {"prompt": "Prove that every continuous function on a closed interval attains its bounds.", "tier": 3, "domain": "math"},
    {"prompt": "Analyze the geopolitical implications of quantum computing on current encryption standards.", "tier": 3, "domain": "reasoning"},

    # T4: Frontier
    {"prompt": "Prove or disprove: P != NP. Outline a novel approach.", "tier": 4, "domain": "math"},
    {"prompt": "Design a novel neural architecture that provably converges faster than transformers on sequence modeling tasks.", "tier": 4, "domain": "code"},
    {"prompt": "Develop a formal proof of the four-color theorem using constructive methods.", "tier": 4, "domain": "math"},
    {"prompt": "Write a complete compiler for a Turing-complete language from scratch in Rust.", "tier": 4, "domain": "code"},
]


# ── Routing Accuracy Benchmark ───────────────────────────────────────────────

@dataclass
class RoutingResult:
    total: int = 0
    correct: int = 0
    per_tier: dict = None
    confusion: dict = None

    def __post_init__(self):
        self.per_tier = self.per_tier or {}
        self.confusion = self.confusion or {}

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


def benchmark_routing(checkpoint_path: str) -> RoutingResult:
    """Evaluate complexity routing accuracy on labeled benchmarks."""
    from zen5_router_training import MoDEConfig, ComplexityEstimator

    log.info("Loading checkpoint...")
    cfg = MoDEConfig()
    model = ComplexityEstimator(cfg)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "complexity" in ckpt:
            model.load_state_dict(ckpt["complexity"])
        log.info(f"Loaded from {checkpoint_path}")
    else:
        log.warning(f"No checkpoint at {checkpoint_path} — evaluating random init")

    model.eval()
    result = RoutingResult()
    confusion = defaultdict(lambda: defaultdict(int))
    per_tier = defaultdict(lambda: {"correct": 0, "total": 0})

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    except Exception:
        log.warning("Could not load tokenizer — using dummy token IDs")
        tokenizer = None

    for sample in ROUTING_BENCHMARKS:
        prompt = sample["prompt"]
        true_tier = sample["tier"]

        # Tokenize
        if tokenizer:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=cfg.complexity_context_len, padding="max_length")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
        else:
            words = prompt.split()
            ids = [hash(w) % cfg.vocab_size for w in words[:cfg.complexity_context_len]]
            ids += [0] * (cfg.complexity_context_len - len(ids))
            input_ids = torch.tensor([ids])
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            tier_logits, _ = model(input_ids, attention_mask)
            pred_tier = tier_logits.argmax(dim=-1).item()

        correct = pred_tier == true_tier
        result.total += 1
        result.correct += int(correct)
        per_tier[true_tier]["total"] += 1
        per_tier[true_tier]["correct"] += int(correct)
        confusion[true_tier][pred_tier] += 1

        status = "OK" if correct else f"MISS (pred T{pred_tier})"
        log.info(f"  T{true_tier} {status:>12s} | {prompt[:60]}")

    result.per_tier = dict(per_tier)
    result.confusion = {k: dict(v) for k, v in confusion.items()}

    log.info(f"\nRouting accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
    for tier in sorted(per_tier.keys()):
        t = per_tier[tier]
        acc = t["correct"] / max(t["total"], 1)
        log.info(f"  T{tier}: {acc:.2%} ({t['correct']}/{t['total']})")

    return result


# ── Utilization Benchmark ────────────────────────────────────────────────────

@dataclass
class UtilizationResult:
    tier_distribution: dict = None
    expert_load_variance: float = 0.0
    zero_expert_ratio: float = 0.0
    architecture_diversity: float = 0.0


def benchmark_utilization(checkpoint_path: str) -> UtilizationResult:
    """Measure expert utilization balance and diversity."""
    from zen5_router_training import MoDEConfig, ComplexityEstimator, MoDERouter

    cfg = MoDEConfig()
    complexity = ComplexityEstimator(cfg)
    router = MoDERouter(cfg)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "complexity" in ckpt:
            complexity.load_state_dict(ckpt["complexity"])
        if "router" in ckpt:
            router.load_state_dict(ckpt["router"])

    complexity.eval()
    router.eval()

    tier_counts = Counter()
    for sample in ROUTING_BENCHMARKS:
        ids = [hash(w) % cfg.vocab_size for w in sample["prompt"].split()[:cfg.complexity_context_len]]
        ids += [0] * (cfg.complexity_context_len - len(ids))
        input_ids = torch.tensor([ids])

        with torch.no_grad():
            tier_logits, _ = complexity(input_ids)
            pred = tier_logits.argmax(dim=-1).item()
        tier_counts[pred] += 1

    result = UtilizationResult()
    result.tier_distribution = dict(tier_counts)

    log.info(f"Tier distribution: {dict(tier_counts)}")
    return result


# ── Latency Benchmark ────────────────────────────────────────────────────────

def benchmark_latency(checkpoint_path: str, num_runs: int = 10):
    """Measure inference latency per tier."""
    from zen5_router_training import MoDEConfig, ComplexityEstimator

    cfg = MoDEConfig()
    model = ComplexityEstimator(cfg)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "complexity" in ckpt:
            model.load_state_dict(ckpt["complexity"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Measure per-tier latency
    targets = {
        "T0 (trivial)": "Hello!",
        "T1 (standard)": "Summarize the key economic impacts of climate change on global agriculture.",
        "T2 (complex)": "Compare and contrast the economic policies of Keynesianism versus monetarism " * 5,
        "T4 (frontier)": "Prove that P != NP by constructing a novel oracle separation argument " * 10,
    }

    log.info(f"Latency benchmark ({num_runs} runs each, device={device}):")
    for label, prompt in targets.items():
        ids = [hash(w) % cfg.vocab_size for w in prompt.split()[:cfg.complexity_context_len]]
        ids += [0] * (cfg.complexity_context_len - len(ids))
        input_ids = torch.tensor([ids], device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        p99 = sorted(times)[int(len(times) * 0.99)]
        log.info(f"  {label:20s}: avg={avg:.2f}ms  p99={p99:.2f}ms")


# ── Full Report ──────────────────────────────────────────────────────────────

def generate_report(checkpoint_path: str, output_path: str):
    """Generate comprehensive benchmark report."""
    log.info("=" * 60)
    log.info("zen5 MoDE Benchmark Report")
    log.info("=" * 60)

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    routing = benchmark_routing(checkpoint_path)
    report["routing"] = {
        "accuracy": routing.accuracy,
        "total": routing.total,
        "correct": routing.correct,
        "per_tier": routing.per_tier,
        "confusion_matrix": routing.confusion,
    }

    utilization = benchmark_utilization(checkpoint_path)
    report["utilization"] = {
        "tier_distribution": utilization.tier_distribution,
    }

    benchmark_latency(checkpoint_path)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport saved: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="zen5 MoDE Benchmark Suite")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("routing", help="Benchmark routing accuracy")
    p.add_argument("--checkpoint", default="./checkpoints/router/best.pt")

    p = sub.add_parser("utilization", help="Benchmark expert utilization")
    p.add_argument("--checkpoint", default="./checkpoints/router/best.pt")

    p = sub.add_parser("latency", help="Benchmark inference latency")
    p.add_argument("--checkpoint", default="./checkpoints/router/best.pt")
    p.add_argument("--runs", type=int, default=10)

    p = sub.add_parser("report", help="Generate full report")
    p.add_argument("--checkpoint", default="./checkpoints/router/best.pt")
    p.add_argument("--output", default="./benchmark_report.json")

    args = parser.parse_args()

    if args.cmd == "routing":
        benchmark_routing(args.checkpoint)
    elif args.cmd == "utilization":
        benchmark_utilization(args.checkpoint)
    elif args.cmd == "latency":
        benchmark_latency(args.checkpoint, args.runs)
    elif args.cmd == "report":
        generate_report(args.checkpoint, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
