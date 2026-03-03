#!/usr/bin/env python3
"""
zen5 MoDE — HuggingFace Repository Setup

Creates HuggingFace model repos for the zen5 lineup with proper model cards.

Models:
  zen5       — General (750B, 0.8-50B active)
  zen5-coder — Code-focused (1.8T, 0.8-80B active)
  zen5-omni  — Omnimodal (2.5T, 0.8-100B active)
  zen5-max   — Frontier (3.1T, 0.8-100B+ active)

Usage:
  python zen5_hf_repos.py           # Create all repos
  python zen5_hf_repos.py zen5      # Create specific repo
  python zen5_hf_repos.py --dry-run # Preview READMEs without creating
"""

import io
import sys
from huggingface_hub import HfApi

api = HfApi()

ZEN5_MODELS = [
    {
        "zen_name": "zen5",
        "display": "Zen5",
        "params": "750B",
        "active": "0.8-50B",
        "expert_pool": "Qwen3.5-0.8B + Qwen3.5-9B + GLM-5",
        "target": "General",
        "modalities": "Text",
        "context": "262K",
        "license": "apache-2.0",
        "desc": "General-purpose MoDE model with complexity-aware routing. Uses 0.8B for simple tasks, scales to 50B for complex reasoning.",
    },
    {
        "zen_name": "zen5-coder",
        "display": "Zen5 Coder",
        "params": "1.8T",
        "active": "0.8-80B",
        "expert_pool": "Qwen3.5 + GLM-5 + Kimi K2.5 code experts",
        "target": "Code",
        "modalities": "Text + Code",
        "context": "262K",
        "license": "apache-2.0",
        "desc": "Code-focused MoDE with 384 Kimi K2.5 experts for frontier coding capability. SWE-bench class performance with adaptive compute.",
    },
    {
        "zen_name": "zen5-omni",
        "display": "Zen5 Omni",
        "params": "2.5T",
        "active": "0.8-100B",
        "expert_pool": "All text + Vision + Video + 3D + Audio",
        "target": "Omnimodal",
        "modalities": "Text + Vision + Video + 3D + Audio",
        "context": "262K",
        "license": "apache-2.0",
        "desc": "True omnimodal AI: text, vision, video, 3D generation, and audio in a single unified model with Transfusion architecture.",
    },
    {
        "zen_name": "zen5-max",
        "display": "Zen5 Max",
        "params": "3.1T",
        "active": "0.8-100B+",
        "expert_pool": "All sources, all modalities",
        "target": "Frontier",
        "modalities": "Text + Vision + Video + 3D + Audio",
        "context": "262K",
        "license": "apache-2.0",
        "desc": "Frontier 3.1T MoDE — the largest open-source model. Full expert pool with complexity-aware routing across 6 model families and 5 modalities.",
    },
]


def generate_readme(m):
    """Generate zen5 model card."""

    # Build family table
    family_rows = []
    for model in ZEN5_MODELS:
        bold = "**" if model["zen_name"] == m["zen_name"] else ""
        family_rows.append(
            f"| {bold}{model['display']}{bold} | {bold}{model['params']}{bold} | "
            f"{bold}{model['active']}{bold} | {bold}{model['modalities']}{bold} | "
            f"[zenlm/{model['zen_name']}](https://huggingface.co/zenlm/{model['zen_name']}) |"
        )
    family_table = "\n".join(family_rows)

    return f"""---
license: {m['license']}
language:
- en
- zh
- ja
- ko
- fr
- de
- es
- pt
- ru
- ar
tags:
- zen5
- zenlm
- hanzo
- mode
- mixture-of-diverse-experts
- frontier-ai
- {'omnimodal' if 'Vision' in m['modalities'] else 'text-generation'}
pipeline_tag: text-generation
library_name: transformers
---

# {m['display']}

**{m['display']}** is a {m['params']} parameter AI model from the [Zen5 family](https://zenlm.org) by [Zen LM](https://huggingface.co/zenlm) and [Hanzo AI](https://hanzo.ai).

{m['desc']}

## Architecture: MoDE (Mixture of Diverse Experts)

zen5 introduces **MoDE** — a novel architecture that routes across expert modules harvested from the largest open-source models with **complexity-aware hierarchical routing**.

| Feature | Value |
|---------|-------|
| **Total Parameters** | {m['params']} |
| **Active Parameters** | {m['active']} (adaptive) |
| **Architecture** | MoDE (Mixture of Diverse Experts) |
| **Expert Pool** | {m['expert_pool']} |
| **Modalities** | {m['modalities']} |
| **Context** | {m['context']} tokens |
| **Complexity Tiers** | T0 (0.8B) → T1 (9B) → T2 (10-40B) → T3 (40-80B) → T4 (80B+) |
| **License** | {m['license'].upper()} |
| **Family** | Zen5 |
| **Target** | {m['target']} |
| **Creator** | Zen LM / Hanzo AI |

## How It Works

Unlike standard models that always use the same compute, zen5 **adapts compute to task difficulty**:

- **"Hello, how are you?"** → T0 tier, 0.8B active params, <50ms
- **"Summarize this article"** → T1 tier, 9B active params, <200ms
- **"Prove the Riemann hypothesis approach"** → T4 tier, 100B+ active params, full reasoning

The complexity estimator runs on the first 64 tokens and routes to the appropriate expert tier.

## Key Innovations

1. **Complexity-Aware Routing**: Lightweight estimator predicts task difficulty, activates appropriate tier
2. **Cross-Architecture Experts**: Experts from 6+ diverse model families (Gated DeltaNet, Lightning Attention, DeepseekV3 MoE, FP8 MoE, GLM MoE)
3. **MoE++ Zero-Computation Experts**: Zero/Copy/Constant experts for trivial tokens (1.1-2.1x throughput)
4. **ReMoE ReLU Routing**: Adaptive expert count via ReLU gating (no fixed Top-K)
5. **Adaptive Escalation**: Mid-generation promotion to higher tiers if confidence drops
{'6. **Transfusion**: Unified autoregressive (text) + diffusion (3D/video) in single model' if 'Vision' in m['modalities'] else ''}

## Zen5 Family

| Model | Total Params | Active Params | Modalities | HuggingFace |
|-------|-------------|--------------|-----------|-------------|
{family_table}

## Links

- [Zen LM](https://zenlm.org) | [Hanzo AI](https://hanzo.ai) | [Hanzo Chat](https://hanzo.chat)
- [All Zen Models](https://huggingface.co/zenlm)
- [Architecture Paper](https://zenlm.org/zen5-mode) (coming soon)

---

*Zen5: Mixture of Diverse Experts — Clarity Through Diversity*
"""


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = args[0] if args else "all"

    if target == "all":
        models = ZEN5_MODELS
    else:
        models = [m for m in ZEN5_MODELS if m["zen_name"] == target]

    if not models:
        print(f"No model matched: {target}")
        print(f"Available: {', '.join(m['zen_name'] for m in ZEN5_MODELS)}")
        sys.exit(1)

    for m in models:
        repo_id = f"zenlm/{m['zen_name']}"
        readme = generate_readme(m)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"DRY RUN: {repo_id}")
            print(f"{'='*60}")
            print(readme[:500] + "...")
            continue

        try:
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"feat: {m['display']} MoDE model card",
            )
            print(f"OK: {repo_id} ({m['params']})")
        except Exception as e:
            print(f"FAIL: {repo_id} — {e}")


if __name__ == "__main__":
    main()
