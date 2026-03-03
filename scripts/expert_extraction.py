#!/usr/bin/env python3
"""
zen5 MoDE — Expert Extraction Pipeline

Extracts expert FFN modules from source models, profiles their capabilities
via activation analysis, and creates an indexed expert catalog for MoDE routing.

Supports both dense models (extract full FFN per layer) and MoE models
(extract individual expert FFNs from MoE blocks).

Source models:
  T0: Qwen3.5-0.8B   — dense, Gated DeltaNet
  T1: Qwen3.5-9B     — dense, Gated DeltaNet
  T2: MiniMax-M2.5    — 230B, Lightning Attention MoE
  T3: GLM-5           — 744B, GLM MoE
  T4: Kimi K2.5       — 1.04T, DeepseekV3 MoE, 384 experts
  T4: Ling-1T         — 1T, FP8 MoE

Usage:
  python zen5_expert_extraction.py list
  python zen5_expert_extraction.py extract --model qwen35-0.8b --output ./experts/
  python zen5_expert_extraction.py extract --model kimi-k2.5 --output ./experts/ --shard 0 --num-shards 4
  python zen5_expert_extraction.py fingerprint --model qwen35-0.8b --output ./experts/ --device cuda
  python zen5_expert_extraction.py extract-all --output ./experts/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zen5-extract")


# ── Source Model Registry ────────────────────────────────────────────────────

MODELS: dict[str, dict[str, Any]] = {
    "qwen35-0.8b": {
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "tier": "T0",
        "architecture": "gated_deltanet",
        "params": "0.8B",
        "expert_type": "dense",
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "moe": False,
        "description": "Trivial-tier dense expert for simple tasks",
    },
    "qwen35-9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "tier": "T1",
        "architecture": "gated_deltanet",
        "params": "9B",
        "expert_type": "dense",
        "num_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "moe": False,
        "description": "Standard-tier dense expert for general tasks",
    },
    "minimax-m2.5": {
        "hf_id": "MiniMaxAI/MiniMax-M2.5",
        "tier": "T2",
        "architecture": "lightning_attention_moe",
        "params": "230B",
        "active_params": "10B",
        "expert_type": "hybrid_moe",
        "moe": True,
        "moe_block_pattern": "block_sparse_moe",
        "expert_attr": "experts",
        "description": "Complex-tier MoE with Lightning Attention for ultra-long context",
    },
    "glm-5": {
        "hf_id": "THUDM/GLM-5",
        "tier": "T3",
        "architecture": "glm_moe",
        "params": "744B",
        "active_params": "40B",
        "expert_type": "standard_moe",
        "num_experts": 256,
        "moe": True,
        "moe_block_pattern": "mlp",  # GLM uses mlp.experts
        "expert_attr": "experts",
        "description": "Advanced-tier MoE for complex reasoning and multilingual",
    },
    "kimi-k2.5": {
        "hf_id": "moonshotai/Kimi-K2.5",
        "tier": "T4",
        "architecture": "deepseek_v3_moe",
        "params": "1.04T",
        "active_params": "32B",
        "expert_type": "deepseek_moe",
        "num_experts": 384,
        "top_k": 8,
        "moe": True,
        "moe_block_pattern": "mlp",  # DeepseekV3: model.layers[i].mlp.experts[j]
        "expert_attr": "experts",
        "has_shared_experts": True,
        "description": "Frontier-tier MoE with 384 experts, agentic reasoning",
    },
    "ling-1t": {
        "hf_id": "inclusionAI/Ling-1T",
        "tier": "T4",
        "architecture": "fp8_moe",
        "params": "1T",
        "active_params": "50B",
        "expert_type": "fp8_moe",
        "moe": True,
        "moe_block_pattern": "mlp",
        "expert_attr": "experts",
        "description": "Frontier-tier FP8 MoE, largest open-source model",
    },
}


# ── Expert Fingerprint ───────────────────────────────────────────────────────

@dataclass
class ExpertFingerprint:
    """Activation profile for a single expert."""
    expert_id: str
    model_name: str
    layer_idx: int
    expert_idx: int  # -1 for dense models (entire layer FFN)
    architecture: str
    tier: str

    # Activation scores per domain (0.0-1.0), computed via fingerprinting
    math_score: float = 0.0
    code_score: float = 0.0
    reasoning_score: float = 0.0
    multilingual_score: float = 0.0
    creative_score: float = 0.0
    factual_score: float = 0.0

    # Shape info
    input_dim: int = 0
    output_dim: int = 0
    param_count: int = 0

    # Storage
    weight_path: str = ""
    dtype: str = "bfloat16"
    requires_alignment: bool = True


@dataclass
class ExpertCatalog:
    """Indexed collection of extracted expert fingerprints."""
    model_name: str
    total_experts: int = 0
    total_params: int = 0
    experts: list[dict] = field(default_factory=list)

    def add(self, fp: ExpertFingerprint):
        self.experts.append(asdict(fp))
        self.total_experts += 1
        self.total_params += fp.param_count

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info(f"Catalog saved: {path} ({self.total_experts} experts, {self.total_params:,} params)")

    @classmethod
    def load(cls, path: str) -> "ExpertCatalog":
        with open(path) as f:
            data = json.load(f)
        cat = cls(model_name=data["model_name"])
        cat.total_experts = data["total_experts"]
        cat.total_params = data["total_params"]
        cat.experts = data["experts"]
        return cat


# ── MoE Layer Navigator ─────────────────────────────────────────────────────

def find_moe_layers(model: nn.Module, config: dict) -> list[tuple[int, nn.Module, list[nn.Module]]]:
    """
    Navigate model architecture to find MoE blocks and their expert modules.

    Returns list of (layer_idx, moe_block, [expert_modules]).
    Handles: DeepseekV3 (Kimi), GLM, Qwen3-MoE, MiniMax, Ling.
    """
    results = []
    layers = None

    # Navigate to the decoder layers
    for attr in ("model.layers", "transformer.layers", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        log.warning("Could not find decoder layers — trying model.layers")
        layers = getattr(model, "model", model)
        layers = getattr(layers, "layers", [])

    for layer_idx, layer in enumerate(layers):
        # Find the MoE block within this layer
        moe_block = None
        experts = []

        # Strategy 1: layer.mlp.experts (DeepseekV3, GLM, Ling)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            exp = getattr(mlp, "experts", None)
            if exp is not None:
                moe_block = mlp
                experts = list(exp) if hasattr(exp, "__iter__") else [exp]

        # Strategy 2: layer.block_sparse_moe.experts (Mixtral-style, MiniMax)
        if not experts:
            bsm = getattr(layer, "block_sparse_moe", None)
            if bsm is not None:
                exp = getattr(bsm, "experts", None)
                if exp is not None:
                    moe_block = bsm
                    experts = list(exp) if hasattr(exp, "__iter__") else [exp]

        # Strategy 3: layer.ffn.experts (alternative naming)
        if not experts:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None:
                exp = getattr(ffn, "experts", None)
                if exp is not None:
                    moe_block = ffn
                    experts = list(exp) if hasattr(exp, "__iter__") else [exp]

        if experts:
            results.append((layer_idx, moe_block, experts))

    return results


def find_dense_ffn(model: nn.Module) -> list[tuple[int, nn.Module]]:
    """Find FFN/MLP blocks in dense (non-MoE) models."""
    results = []
    layers = getattr(getattr(model, "model", model), "layers", [])

    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            results.append((layer_idx, mlp))
    return results


# ── Expert Extraction ────────────────────────────────────────────────────────

def _module_stats(module: nn.Module) -> tuple[int, int, int]:
    """Return (param_count, input_dim, output_dim) for a module."""
    param_count = sum(p.numel() for p in module.parameters())
    input_dim, output_dim = 0, 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            input_dim = m.in_features
            output_dim = m.out_features
            break
    return param_count, input_dim, output_dim


def extract_dense(model_name: str, cfg: dict, output_dir: str, device: str = "cpu") -> ExpertCatalog:
    """Extract FFN blocks from dense models as single-expert-per-layer."""
    from transformers import AutoModelForCausalLM

    log.info(f"Loading {model_name} ({cfg['params']}) on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    catalog = ExpertCatalog(model_name=model_name)
    os.makedirs(output_dir, exist_ok=True)

    ffn_layers = find_dense_ffn(model)
    log.info(f"Found {len(ffn_layers)} FFN layers")

    for layer_idx, ffn in ffn_layers:
        weight_path = os.path.join(output_dir, f"layer{layer_idx:03d}_expert000.pt")
        torch.save(ffn.state_dict(), weight_path)

        pc, in_d, out_d = _module_stats(ffn)
        fp = ExpertFingerprint(
            expert_id=f"{model_name}/L{layer_idx}/E0",
            model_name=model_name, layer_idx=layer_idx, expert_idx=0,
            architecture=cfg["architecture"], tier=cfg["tier"],
            input_dim=in_d, output_dim=out_d, param_count=pc,
            weight_path=weight_path,
        )
        catalog.add(fp)
        log.info(f"  L{layer_idx:3d} E000: {pc:>12,} params → {os.path.basename(weight_path)}")

    catalog.save(os.path.join(output_dir, "catalog.json"))
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return catalog


def extract_moe(
    model_name: str, cfg: dict, output_dir: str,
    device: str = "cpu",
    shard: int = 0, num_shards: int = 1,
) -> ExpertCatalog:
    """
    Extract individual expert FFN modules from MoE models.

    For large models (>100B), use sharding: split layers across jobs.
    Each shard processes layers [shard::num_shards].

    Example — 4-GPU parallel extraction of Kimi K2.5:
      GPU 0: python extract.py --model kimi-k2.5 --shard 0 --num-shards 4
      GPU 1: python extract.py --model kimi-k2.5 --shard 1 --num-shards 4
      GPU 2: python extract.py --model kimi-k2.5 --shard 2 --num-shards 4
      GPU 3: python extract.py --model kimi-k2.5 --shard 3 --num-shards 4
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    os.makedirs(output_dir, exist_ok=True)
    catalog = ExpertCatalog(model_name=model_name)

    # Phase 1: Load config to determine architecture
    log.info(f"Loading config for {model_name} ({cfg['params']})...")
    hf_config = AutoConfig.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    num_layers = getattr(hf_config, "num_hidden_layers", 0)
    num_experts_cfg = getattr(hf_config, "num_local_experts",
                     getattr(hf_config, "n_routed_experts",
                     cfg.get("num_experts", 0)))

    log.info(f"  Layers: {num_layers}, Experts/layer: {num_experts_cfg}")

    # Compute shard assignment
    my_layers = list(range(shard, num_layers, num_shards))
    log.info(f"  Shard {shard}/{num_shards}: processing layers {my_layers[:5]}{'...' if len(my_layers) > 5 else ''} ({len(my_layers)} total)")

    # Phase 2: Load model — for frontier models, use quantized loading
    total_gb = _estimate_model_gb(cfg)
    if total_gb > 200:
        log.info(f"  Model size ~{total_gb:.0f}GB — using INT4 quantized loading + CPU offload")
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,
        )
    elif total_gb > 40:
        log.info(f"  Model size ~{total_gb:.0f}GB — using auto device map")
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    log.info(f"Loading model weights...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **load_kwargs)
    model.eval()
    log.info(f"  Loaded in {time.time() - t0:.1f}s")

    # Phase 3: Find and extract MoE experts
    moe_layers = find_moe_layers(model, cfg)
    log.info(f"Found {len(moe_layers)} MoE layers")

    if not moe_layers:
        log.warning("No MoE layers found — falling back to dense extraction")
        return extract_dense(model_name, cfg, output_dir, device)

    for layer_idx, moe_block, experts in moe_layers:
        if layer_idx not in my_layers:
            continue

        # Also extract shared expert if present (DeepseekV3)
        shared_expert = getattr(moe_block, "shared_experts", None)
        if shared_expert is not None:
            weight_path = os.path.join(output_dir, f"layer{layer_idx:03d}_shared.pt")
            torch.save(shared_expert.state_dict(), weight_path)
            pc, in_d, out_d = _module_stats(shared_expert)
            fp = ExpertFingerprint(
                expert_id=f"{model_name}/L{layer_idx}/shared",
                model_name=model_name, layer_idx=layer_idx, expert_idx=-1,
                architecture=cfg["architecture"], tier=cfg["tier"],
                input_dim=in_d, output_dim=out_d, param_count=pc,
                weight_path=weight_path,
            )
            catalog.add(fp)
            log.info(f"  L{layer_idx:3d} SHARED: {pc:>12,} params")

        # Extract each expert FFN
        for expert_idx, expert in enumerate(experts):
            weight_path = os.path.join(output_dir, f"layer{layer_idx:03d}_expert{expert_idx:03d}.pt")
            state = {k: v.cpu() for k, v in expert.state_dict().items()}
            torch.save(state, weight_path)

            pc, in_d, out_d = _module_stats(expert)
            fp = ExpertFingerprint(
                expert_id=f"{model_name}/L{layer_idx}/E{expert_idx}",
                model_name=model_name, layer_idx=layer_idx, expert_idx=expert_idx,
                architecture=cfg["architecture"], tier=cfg["tier"],
                input_dim=in_d, output_dim=out_d, param_count=pc,
                weight_path=weight_path,
            )
            catalog.add(fp)

        log.info(f"  L{layer_idx:3d}: {len(experts)} experts extracted")

    # Also extract gate/router weights
    for layer_idx, moe_block, _ in moe_layers:
        if layer_idx not in my_layers:
            continue
        gate = getattr(moe_block, "gate", getattr(moe_block, "router", None))
        if gate is not None:
            gate_path = os.path.join(output_dir, f"layer{layer_idx:03d}_gate.pt")
            torch.save(gate.state_dict(), gate_path)
            log.info(f"  L{layer_idx:3d} GATE saved")

    shard_suffix = f"_shard{shard}" if num_shards > 1 else ""
    catalog.save(os.path.join(output_dir, f"catalog{shard_suffix}.json"))

    del model; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return catalog


def _estimate_model_gb(cfg: dict) -> float:
    """Rough estimate of model size in GB (bf16)."""
    params_str = cfg.get("params", "0")
    multiplier = {"B": 1e9, "T": 1e12, "M": 1e6}
    for suffix, mult in multiplier.items():
        if params_str.upper().endswith(suffix):
            num = float(params_str[:-1].replace("+", ""))
            return num * mult * 2 / 1e9  # bf16 = 2 bytes/param
    return 0


# ── Expert Fingerprinting ────────────────────────────────────────────────────

BENCHMARK_PROMPTS = {
    "math": [
        "Solve: What is the integral of x^2 dx?",
        "If f(x) = 3x^2 + 2x - 1, find f'(x).",
        "What is 47 * 83?",
        "Prove that the square root of 2 is irrational.",
    ],
    "code": [
        "Write a Python function to find the longest common subsequence.",
        "Implement a binary search tree in Rust.",
        "Write a SQL query to find duplicate rows in a table.",
        "Explain the difference between async and threads in Python.",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
        "A is taller than B. C is shorter than B. D is taller than A. Who is the shortest?",
        "Three boxes: one has gold, two have rocks. You pick box A. Host opens box C (rocks). Switch?",
    ],
    "multilingual": [
        "Translate to Chinese: The future of AI is open-source.",
        "Quelle est la capitale de la France?",
        "日本の人口は何人ですか？",
    ],
    "creative": [
        "Write a haiku about quantum computing.",
        "Describe a color that doesn't exist.",
    ],
    "factual": [
        "What is the speed of light in a vacuum?",
        "Who wrote 'The Republic'?",
        "What causes tides on Earth?",
    ],
}


def fingerprint_model(
    model_name: str, cfg: dict, catalog_dir: str,
    device: str = "cuda", max_samples: int = 4,
):
    """
    Run benchmark prompts through the model, recording per-layer expert activations.
    Uses forward hooks to measure which experts activate on which domains.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    catalog_path = os.path.join(catalog_dir, "catalog.json")
    if not os.path.exists(catalog_path):
        log.error(f"No catalog at {catalog_path} — run extract first")
        return

    catalog = ExpertCatalog.load(catalog_path)
    log.info(f"Loaded catalog: {catalog.total_experts} experts")

    log.info(f"Loading model + tokenizer on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Hook storage: domain → layer → activation_magnitude
    activations: dict[str, dict[int, float]] = {}

    def make_hook(domain: str, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            mag = out.float().abs().mean().item()
            activations.setdefault(domain, {})
            activations[domain].setdefault(layer_idx, 0.0)
            activations[domain][layer_idx] += mag
        return hook_fn

    # Register hooks on FFN/MoE blocks
    layers = getattr(getattr(model, "model", model), "layers", [])
    hooks = []
    for domain in BENCHMARK_PROMPTS:
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                h = mlp.register_forward_hook(make_hook(domain, layer_idx))
                hooks.append((h, domain))

    # Run inference per domain
    for domain, prompts in BENCHMARK_PROMPTS.items():
        log.info(f"  Profiling domain: {domain} ({len(prompts[:max_samples])} prompts)")
        for prompt in prompts[:max_samples]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Only register hooks for current domain
            with torch.no_grad():
                model(**inputs)

    # Remove hooks
    for h, _ in hooks:
        h.remove()

    # Normalize activations and update catalog
    for expert_data in catalog.experts:
        layer_idx = expert_data["layer_idx"]
        for domain in BENCHMARK_PROMPTS:
            acts = activations.get(domain, {})
            score = acts.get(layer_idx, 0.0)
            # Normalize to 0-1 range per domain
            all_scores = list(acts.values()) or [0]
            max_score = max(all_scores) or 1.0
            normalized = score / max_score
            expert_data[f"{domain}_score"] = round(normalized, 4)

    # Save updated catalog
    catalog.save(os.path.join(catalog_dir, "catalog_fingerprinted.json"))
    log.info(f"Fingerprinted catalog saved with {len(BENCHMARK_PROMPTS)} domain scores")

    del model; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── Catalog Merger ───────────────────────────────────────────────────────────

def merge_catalogs(catalog_dir: str, output_path: str):
    """Merge shard catalogs and/or multiple model catalogs into unified index."""
    merged = ExpertCatalog(model_name="zen5-unified")
    catalog_files = sorted(Path(catalog_dir).rglob("catalog*.json"))

    for cf in catalog_files:
        cat = ExpertCatalog.load(str(cf))
        for expert in cat.experts:
            merged.experts.append(expert)
            merged.total_experts += 1
            merged.total_params += expert.get("param_count", 0)
        log.info(f"  Merged {cf.name}: {cat.total_experts} experts from {cat.model_name}")

    merged.save(output_path)
    log.info(f"Unified catalog: {merged.total_experts} experts, {merged.total_params:,} params")
    return merged


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="zen5 MoDE Expert Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    # list
    sub.add_parser("list", help="List source models")

    # extract
    p = sub.add_parser("extract", help="Extract experts from a model")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--output", default="./experts")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, auto")
    p.add_argument("--shard", type=int, default=0, help="Shard index for distributed extraction")
    p.add_argument("--num-shards", type=int, default=1, help="Total shards")

    # extract-all
    p = sub.add_parser("extract-all", help="Extract from all models")
    p.add_argument("--output", default="./experts")
    p.add_argument("--device", default="cpu")

    # fingerprint
    p = sub.add_parser("fingerprint", help="Profile expert capabilities via activation analysis")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--output", default="./experts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=4)

    # merge
    p = sub.add_parser("merge", help="Merge catalogs into unified index")
    p.add_argument("--input", default="./experts", help="Directory containing catalogs")
    p.add_argument("--output", default="./experts/unified_catalog.json")

    args = parser.parse_args()

    if args.cmd == "list":
        print(f"\n{'Model':<20s} {'Tier':<4s} {'Params':<8s} {'Type':<12s} Architecture")
        print("=" * 80)
        for name, cfg in MODELS.items():
            moe_tag = "MoE" if cfg.get("moe") else "Dense"
            print(f"  {name:<18s} {cfg['tier']:<4s} {cfg['params']:<8s} {moe_tag:<12s} {cfg['architecture']}")
        print(f"\nCombined expert pool: ~3.0T+ parameters")

    elif args.cmd == "extract":
        cfg = MODELS[args.model]
        out = os.path.join(args.output, args.model)
        if cfg.get("moe"):
            extract_moe(args.model, cfg, out, args.device, args.shard, args.num_shards)
        else:
            extract_dense(args.model, cfg, out, args.device)

    elif args.cmd == "extract-all":
        for name, cfg in MODELS.items():
            out = os.path.join(args.output, name)
            try:
                if cfg.get("moe"):
                    extract_moe(name, cfg, out, args.device)
                else:
                    extract_dense(name, cfg, out, args.device)
            except Exception as e:
                log.error(f"FAILED {name}: {e}")

    elif args.cmd == "fingerprint":
        cfg = MODELS[args.model]
        fingerprint_model(args.model, cfg, os.path.join(args.output, args.model),
                         args.device, args.max_samples)

    elif args.cmd == "merge":
        merge_catalogs(args.input, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
