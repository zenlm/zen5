# zen5 — Mixture of Diverse Experts (MoDE)

**zen5** is a next-generation AI architecture that routes across expert modules harvested from the largest open-source models using **complexity-aware hierarchical routing**.

Unlike standard models that always use the same compute, zen5 **adapts compute to task difficulty**:

| Tier | Active Params | Latency | Example Task |
|------|--------------|---------|-------------|
| T0 | 0.8B | <50ms | "Hello, how are you?" |
| T1 | 9B | <200ms | "Summarize this article" |
| T2 | 10-40B | <2s | "Compare Keynesian vs monetarist economics" |
| T3 | 40-80B | <5s | "Derive Black-Scholes from first principles" |
| T4 | 80-100B+ | <10s | "Design a novel consensus algorithm" |

## Architecture

```
Input → ComplexityEstimator (64 tokens) → Tier Assignment
  T0: 0.8B dense → direct generation
  T1: 9B dense → standard generation
  T2: 10-40B MoE → multi-expert routing
  T3: 40-80B MoE → deep expert routing
  T4: 80-100B+ MoE → frontier expert routing + adaptive escalation
```

### Key Innovations

1. **Complexity-Aware Routing** — Lightweight estimator predicts task difficulty from first 64 tokens, activates appropriate tier
2. **Cross-Architecture Experts** — Experts from 6+ diverse model families (Gated DeltaNet, Lightning Attention, DeepseekV3 MoE, FP8 MoE, GLM MoE)
3. **MoE++ Zero-Computation Experts** (ICLR 2025 Oral) — Zero/Copy/Constant experts for trivial tokens, 1.1-2.1x throughput
4. **ReMoE ReLU Routing** (ICLR 2025) — Adaptive expert count via ReLU gating (no fixed Top-K)
5. **Adaptive Escalation** — Mid-generation promotion to higher tiers when confidence drops
6. **Transfusion** — Unified autoregressive (text) + diffusion (3D/video) in single architecture

### Expert Pool (~3.1T total parameters)

| Source | Params | Type | Architecture | Tier |
|--------|--------|------|-------------|------|
| Qwen3.5-0.8B | 0.8B | Dense | Gated DeltaNet | T0 |
| Qwen3.5-9B | 9B | Dense | Gated DeltaNet | T1 |
| MiniMax-M2.5 | 230B | MoE | Lightning Attention | T2 |
| GLM-5 | 744B | MoE | GLM MoE | T3 |
| Kimi K2.5 | 1.04T | MoE (384 experts) | DeepseekV3 | T4 |
| Ling-1T | 1T | MoE | FP8 MoE | T4 |

### Omnimodal Experts

| Modality | Source | Architecture |
|----------|--------|-------------|
| Vision | Qwen3-VL | Native ViT |
| Video | Wan2.2 / CogVideoX | MoE video diffusion |
| 3D | TRELLIS.2 | Rectified Flow DiT + SC-VAE |
| Audio | Qwen3-Omni / zen-tts | Thinker-Talker |

## Model Lineup

| Model | Total Params | Active Params | Target |
|-------|-------------|--------------|--------|
| **zen5** | 750B | 0.8-50B | General |
| **zen5-coder** | 1.8T | 0.8-80B | Code |
| **zen5-omni** | 2.5T | 0.8-100B | Omnimodal |
| **zen5-max** | 3.1T | 0.8-100B+ | Frontier |

## Trainable Parameters

Only **~394M parameters** are trained (0.013% of total model):

- **ComplexityEstimator**: 207M — predicts task difficulty tier
- **AlignmentLayer**: 134M — projects diverse architectures to shared space
- **MoDERouter**: 52M — per-tier expert routing with ReLU gating

All expert weights are **frozen** — extracted and served as-is from source models.

## Quick Start

```bash
# List expert pool
python scripts/expert_extraction.py list

# Extract experts from source models
python scripts/expert_extraction.py extract-all --output ./experts/

# Train complexity router
python scripts/router_training.py train --output ./checkpoints/

# Benchmark routing accuracy
python scripts/benchmark.py routing --checkpoint ./checkpoints/router/best.pt

# Generate full evaluation report
python scripts/benchmark.py report --checkpoint ./checkpoints/router/best.pt
```

## Training

zen5 is trained in the open on the [Hanzo Network](https://hanzo.ai). Training consists of 5 phases:

1. **Expert Extraction** — Harvest FFN/MoE blocks from source models
2. **Alignment Pre-training** — Learn cross-architecture projections
3. **Router Training** — Train complexity estimator and tier routers
4. **Integration** — Joint fine-tuning with frozen experts
5. **Omnimodal Fusion** — Add vision/video/3D/audio experts

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed training strategy and hardware requirements.

## Repository Structure

```
zen5/
├── scripts/
│   ├── expert_extraction.py   # Extract experts from source models
│   ├── router_training.py     # Train complexity router + alignment
│   ├── benchmark.py           # Evaluation and benchmarking suite
│   └── hf_repos.py           # HuggingFace model card generation
├── docs/
│   └── ARCHITECTURE.md        # Full architecture specification
├── training/                  # Training configs and data
├── checkpoints/               # Model checkpoints
└── paper/                     # Technical report
```

## Requirements

```
torch>=2.4
transformers>=4.48
safetensors
huggingface_hub
```

## References

- [MoE++ (ICLR 2025 Oral)](https://arxiv.org/abs/2410.07348) — Zero-Computation Experts
- [ReMoE (ICLR 2025)](https://arxiv.org/abs/2412.14711) — ReLU-based MoE routing
- [HMoE (EMNLP 2025)](https://arxiv.org/abs/2505.12209) — Heterogeneous expert sizes
- [Symbolic-MoE (2025)](https://arxiv.org/abs/2503.05641) — Skill-based routing
- [Uni-MoE-2.0](https://arxiv.org/abs/2511.12609) — Dynamic capacity multimodal MoE
- [Transfusion (Meta 2024)](https://arxiv.org/abs/2408.11039) — Unified AR + diffusion

## Links

- [Zen LM](https://zenlm.org) — Model family home
- [Hanzo AI](https://hanzo.ai) — Infrastructure and training
- [HuggingFace](https://huggingface.co/zenlm) — Model downloads
- [Architecture Paper](https://zenlm.org/zen5-mode) — Coming soon

## License

Apache 2.0

---

*zen5: Mixture of Diverse Experts — Clarity Through Diversity*
