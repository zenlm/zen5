# Zen5 MoDE Architecture

## Mixture of Diverse Experts — 2T+ Omnimodal

### Vision

zen5 is a **2T+ parameter omnimodal AI** built on **MoDE (Mixture of Diverse Experts)** — a novel architecture that routes across expert modules harvested from the largest and best open-source models, with **complexity-aware hierarchical routing** that adapts compute to task difficulty. Simple chat uses 0.8B params. Frontier reasoning activates the full 2T+.

zen5 handles **every modality**: text, vision, 3D, video, and audio — in a single unified model.

### Why MoDE

1. **Largest open-source model** (2T+ total params)
2. **Adaptive compute** — 0.8B for "hello", 2T+ for proving theorems
3. **100-2000x cheaper to train** — router + alignment only, experts frozen
4. **Architecturally diverse** — experts from 6+ model families learn different representations
5. **True omnimodal** — text + vision + 3D + video + audio in one model
6. **Genuinely novel** — no existing model combines cross-architecture experts with complexity routing

---

## Expert Pool

### Text Experts (Core)

| Source | Total | Active | Architecture | Expert Type | Complexity Tier |
|--------|-------|--------|-------------|-------------|-----------------|
| Qwen3.5-0.8B | 0.8B | 0.8B | Gated DeltaNet | Dense | T0 — Trivial |
| Qwen3.5-9B | 9B | 9B | Gated DeltaNet | Dense | T1 — Standard |
| MiniMax-M2.5 | 230B | 10B | Lightning Attention MoE | Hybrid MoE | T2 — Complex |
| GLM-5 | 744B | 40B | GLM MoE | Standard MoE | T3 — Advanced |
| Kimi K2.5 | 1.04T | 32B | DeepseekV3 MoE | 384 experts, top-8 | T4 — Frontier |
| Ling-1T | 1T | 50B | FP8 MoE | MoE | T4 — Frontier |

### Multimodal Experts

| Source | Modality | Total | Active | Architecture |
|--------|----------|-------|--------|-------------|
| Qwen3-VL-30B | Vision | 30B | 30B | Vision-Language Transformer |
| Wan2.2 | Video | ~14B | ~5B | MoE Diffusion (3D Causal VAE) |
| CogVideoX-5B | Video | 5B | 5B | Expert Transformer + 3D-RoPE |
| TRELLIS.2 (Microsoft) | 3D | 4B | 4B | Rectified Flow DiT + SC-VAE |
| Qwen3-Omni | Audio | 30B | 3B | Omni MoE (speech + text) |

### Combined Expert Pool

| Category | Total Params | Active (max) | Expert Count |
|----------|-------------|-------------|-------------|
| Text (6 models) | ~3.0T | ~142B | 1000+ |
| Vision | 30B | 30B | Dense |
| Video | ~19B | ~10B | MoE + Dense |
| 3D | 4B | 4B | DiT blocks |
| Audio | 30B | 3B | MoE |
| **Total** | **~3.1T** | **~189B max** | **1000+** |

**Active params at any time**: 0.8B (trivial) to ~100B+ (frontier omnimodal)

---

## Complexity-Aware Hierarchical Router

### The Key Innovation

Unlike standard MoE which routes every token the same way, zen5's router **estimates task complexity first**, then activates the appropriate tier of experts. This gives zen5 the efficiency of a 0.8B model on simple tasks and the capability of a 2T+ model on hard ones.

### Complexity Tiers

```
                    Complexity Estimation
                           │
              ┌────────────┼────────────────┐
              │            │                │
              ▼            ▼                ▼
         ┌────────┐  ┌─────────┐     ┌──────────┐
         │ T0-T1  │  │  T2-T3  │     │   T4     │
         │ Simple │  │ Complex │     │ Frontier │
         └───┬────┘  └────┬────┘     └────┬─────┘
             │            │               │
             ▼            ▼               ▼
        ┌─────────┐ ┌──────────┐   ┌───────────┐
        │Qwen 0.8B│ │MiniMax   │   │Kimi K2.5  │
        │Qwen 9B  │ │GLM-5     │   │Ling-1T    │
        │(dense)  │ │(MoE)     │   │(full MoE) │
        └─────────┘ └──────────┘   └───────────┘
```

### Tier Definitions

| Tier | Complexity | Active Params | Latency | Example Tasks |
|------|-----------|--------------|---------|---------------|
| **T0** | Trivial | 0.8B | <50ms | "Hi", classification, yes/no, entity extraction |
| **T1** | Standard | 9B | <200ms | Summarization, translation, general Q&A, simple code |
| **T2** | Complex | 10-40B | <1s | Multi-step reasoning, long-doc analysis, code review |
| **T3** | Advanced | 40-80B | <3s | Research synthesis, complex math, architecture design |
| **T4** | Frontier | 80-100B+ | <10s | Theorem proving, novel research, agentic multi-step |

### Complexity Estimator

A lightweight classifier (~100M params) trained to predict complexity from the first few tokens:

```
Input tokens (first 64)
         │
         ▼
┌──────────────────┐
│ Complexity Head   │  ← Small transformer (100M params, trainable)
│                   │
│ Features:         │
│ • Token entropy   │  — High entropy = more complex
│ • Query length    │  — Longer queries tend to be harder
│ • Domain signals  │  — Math/code tokens → higher tier
│ • Instruction     │  — "prove", "design" → T3+
│   complexity      │     "translate", "summarize" → T1
│ • Reasoning depth │  — Chain-of-thought indicators
└────────┬─────────┘
         │
         ▼
    [T0, T1, T2, T3, T4] probability distribution
         │
         ▼
    Activate corresponding expert tier
```

### Advanced Routing Techniques

zen5's router incorporates three proven innovations:

1. **MoE++ Zero-Computation Experts** (ICLR 2025 Oral, Skywork): Three special expert types — **zero** (discard), **copy** (skip layer), **constant** (learned vector). Simple tokens get routed to copy/constant experts with zero FFN compute. Provides 1.1-2.1x throughput improvement.

2. **ReMoE ReLU Routing** (ICLR 2025): Replaces fixed TopK gating with ReLU activation. The number of active experts is determined by ReLU sparsity — naturally adaptive to input complexity. No fixed K parameter needed.

3. **Uni-MoE-2.0 Dynamic Capacity** (arXiv 2511.12609): Three expert categories — **shared** (1/8 size, cross-modal knowledge), **routed** (full size, modality-specific), **null** (token skipping). Plus Omni-3D-RoPE for spatio-temporal cross-modal positional encoding.

### Adaptive Escalation

The router supports **mid-generation escalation**: if a T1 expert produces low-confidence outputs, the system can escalate to T2+ without restarting:

```python
# Pseudocode for adaptive escalation
def generate(prompt):
    tier = complexity_estimator(prompt)
    expert = activate_tier(tier)

    for step in generation:
        token, confidence = expert.next_token()

        if confidence < ESCALATION_THRESHOLD and tier < T4:
            # Escalate: pass KV cache to higher-tier expert
            tier += 1
            expert = activate_tier(tier, kv_cache=expert.kv_cache)

        yield token
```

---

## Omnimodal Fusion Architecture

### Transfusion Design

zen5 uses a **Transfusion-inspired** architecture: autoregressive generation for text tokens and diffusion for continuous modalities (images, 3D, video, audio). This runs in a single unified model.

```
                        Input (any modality)
                              │
                 ┌────────────┼────────────────┐
                 │            │                │
            Text tokens   Image/Video     3D/Audio
                 │        patches          features
                 │            │                │
                 ▼            ▼                ▼
          ┌───────────┐ ┌──────────┐    ┌──────────┐
          │ Tokenizer │ │   VAE    │    │ Encoder  │
          │ (unified) │ │ Encoder  │    │ (modal)  │
          └─────┬─────┘ └────┬─────┘    └────┬─────┘
                │            │                │
                └────────────┼────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Shared Embed   │  ← Alignment layer (trainable)
                    │  + Modality     │     Maps all modalities to
                    │    Tokens       │     shared latent space
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Complexity     │  ← Estimates task difficulty
                    │  Estimator     │     Routes to appropriate tier
                    └────────┬────────┘
                             │
                    ┌────────┼────────────────┐
                    │        │                │
                    ▼        ▼                ▼
              ┌──────┐  ┌────────┐     ┌──────────┐
              │ Text │  │Modality│     │ Cross-   │
              │Expert│  │Expert  │     │ Modal    │
              │Router│  │Router  │     │ Router   │
              └──┬───┘  └───┬────┘     └────┬─────┘
                 │          │               │
                 ▼          ▼               ▼
        ┌──────────────────────────────────────────┐
        │         Expert Pool (frozen + LoRA)       │
        │                                           │
        │  TEXT EXPERTS:                             │
        │  ┌──────┐ ┌──────┐ ┌───────┐ ┌────────┐ │
        │  │Qwen  │ │Qwen  │ │MiniMax│ │ GLM-5  │ │
        │  │0.8B  │ │ 9B   │ │M2.5   │ │ 744B   │ │
        │  └──────┘ └──────┘ └───────┘ └────────┘ │
        │  ┌────────┐ ┌────────┐                   │
        │  │Kimi    │ │Ling    │                   │
        │  │K2.5 1T │ │1T FP8 │                   │
        │  └────────┘ └────────┘                   │
        │                                           │
        │  VISION EXPERTS:                          │
        │  ┌──────────┐                             │
        │  │Qwen3-VL  │                             │
        │  │30B       │                             │
        │  └──────────┘                             │
        │                                           │
        │  VIDEO EXPERTS:                           │
        │  ┌──────────┐ ┌──────────┐               │
        │  │ Wan2.2   │ │CogVideoX │               │
        │  │MoE Diff  │ │5B Expert │               │
        │  └──────────┘ └──────────┘               │
        │                                           │
        │  3D EXPERTS:                              │
        │  ┌──────────┐                             │
        │  │ TRELLIS.2  │  (triplane-NeRF)            │
        │  │ + zen-3d │  (multi-view fusion)        │
        │  └──────────┘                             │
        │                                           │
        │  AUDIO EXPERTS:                           │
        │  ┌──────────┐ ┌──────────┐               │
        │  │Qwen3-Omni│ │ zen-tts  │               │
        │  │Audio     │ │ 24kHz    │               │
        │  └──────────┘ └──────────┘               │
        └──────────────────────────────────────────┘
                             │
                    ┌────────┼────────┐
                    │        │        │
                    ▼        ▼        ▼
              ┌──────┐  ┌──────┐  ┌──────┐
              │ Text │  │Diff  │  │ 3D   │
              │ Head │  │Head  │  │ Head │
              │(AR)  │  │(DDPM)│  │(NeRF)│
              └──────┘  └──────┘  └──────┘
                    │        │        │
                    ▼        ▼        ▼
              Text tokens  Video    3D mesh
                          frames   + texture
```

### Cross-Modal Routing

For tasks that span modalities (e.g., "generate a 3D model of the car in this video"), zen5 activates **cross-modal expert chains**:

```
"Generate a 3D model of the car in this video"
    │
    ├─→ Video Expert (Wan2.2) — extract key frames, understand scene
    ├─→ Vision Expert (Qwen3-VL) — identify car, extract features
    ├─→ Text Expert (Kimi K2.5) — plan 3D generation strategy
    └─→ 3D Expert (TRELLIS.2 + zen-3d) — generate mesh + texture
```

The **cross-modal router** learns which expert chains to activate for multi-modal tasks. It's trained on multi-modal instruction data where ground truth chains are known.

---

## Why Fuse These Specific Models

### Architectural Diversity = Complementary Capabilities

Each model family learns fundamentally different representations:

| Architecture | What It Learns Best | Unique Strength |
|-------------|-------------------|-----------------|
| **Gated DeltaNet** (Qwen3.5) | Long-range linear attention patterns | Efficient 262K context, hybrid attention |
| **Lightning Attention** (MiniMax) | Ultra-long context (1M tokens) | Near-constant memory for million-token docs |
| **DeepseekV3 MoE** (Kimi K2.5) | Expert specialization via 384 experts | Fine-grained routing, agentic reasoning |
| **FP8 MoE** (Ling-1T) | Efficient large-scale computation | 1T params in FP8, high throughput |
| **GLM MoE** (GLM-5) | Multilingual + reasoning | Strong Chinese/English balance |
| **3D Causal VAE** (CogVideoX/Wan2.2) | Spatiotemporal video understanding | Native video generation |

### Empirical Evidence

1. **HMoE** (EMNLP 2025): Heterogeneous experts of different sizes outperform homogeneous ones. zen5's tiered experts (0.8B→1T) is HMoE taken to the extreme.

2. **Symbolic-MoE** (arXiv 2503.05641): Skill-based routing across diverse LLMs beats GPT-4o-mini by 8.15% on MMLU-Pro/GPQA/AIME. zen5 does this at the weight level, not inference-time.

3. **Transfusion** (Meta, 2024): Unified autoregressive + diffusion achieves parity with specialized models at 3x less compute. zen5 extends this to 3D and video.

### Advantages of Fusing All Together

**vs. Running Separate Models**:
- Single model, single API, single KV cache
- Cross-modal reasoning (text informs 3D, video informs text)
- Shared representations reduce redundancy
- One set of LoRA adapters for identity/personality

**vs. Training 2T from Scratch**:
- $200K-500K vs $50M+ training cost
- Experts already proven on benchmarks
- No catastrophic forgetting — experts are frozen
- Faster iteration — swap experts without retraining

**vs. Homogeneous MoE (Llama 4 Behemoth)**:
- Diverse architectures learn different features
- No single point of failure in architecture design
- Can upgrade individual expert families as new models release
- True omnimodal vs text-only

### Specific Fusion Benefits

| Fusion | Benefit |
|--------|---------|
| Qwen 0.8B + Kimi 1T | 2000x compute range in one model |
| MiniMax + Qwen3.5 | Lightning Attention for 1M context + DeltaNet for 262K precision |
| Kimi K2.5 + GLM-5 | 384 experts (code/math) + 256 experts (multilingual/reasoning) |
| Wan2.2 + TRELLIS.2 | Video understanding → 3D reconstruction pipeline |
| Qwen3-VL + CogVideoX | Image understanding → video generation |
| Qwen3-Omni + zen-tts | Speech understanding → speech synthesis |

---

## Training Strategy

### Phase 0: Expert Catalog (2 weeks)

Profile each source model's expert capabilities:

```python
# Expert fingerprinting
for model in [qwen_08b, qwen_9b, minimax, glm5, kimi, ling]:
    for expert in model.experts:
        # Run activation analysis on benchmark tasks
        fingerprint = {
            "math_activation": measure(expert, math_benchmark),
            "code_activation": measure(expert, code_benchmark),
            "reasoning_activation": measure(expert, reasoning_benchmark),
            "multilingual_activation": measure(expert, multilingual_benchmark),
            "creative_activation": measure(expert, creative_benchmark),
        }
        catalog[expert.id] = fingerprint
```

### Phase 1: Expert Extraction (1 week)

Extract expert FFN modules from each source model:

```python
# Extract from each architecture
extractors = {
    "qwen35": QwenExpertExtractor,      # Gated DeltaNet blocks
    "minimax": MiniMaxExpertExtractor,    # Lightning Attention + FFN
    "glm5": GLMExpertExtractor,           # Standard MoE blocks
    "kimi": DeepseekV3ExpertExtractor,    # 384 expert FFNs
    "ling": LingExpertExtractor,          # FP8 MoE blocks
}

expert_pool = {}
for name, extractor in extractors.items():
    experts = extractor.extract(model_path=f"models/{name}")
    expert_pool[name] = experts
```

### Phase 2: Alignment Layer Training (2-3 weeks)

Train the shared embedding and output projections:

```
Training data: 1T+ tokens
- 40% general text (web, books, wikipedia)
- 20% code (GitHub, StackOverflow)
- 15% math/science (arXiv, textbooks)
- 10% multilingual (parallel corpora)
- 10% multimodal (image-text, video-text, 3D-text pairs)
- 5% instruction following (conversations, tool use)

Alignment objectives:
1. Token prediction loss (standard LM loss)
2. Cross-architecture alignment loss (experts from different families
   should produce compatible representations)
3. Modality alignment loss (text and vision representations should
   be geometrically aligned in shared space)
```

### Phase 3: Complexity Router Training (2-4 weeks)

Train the complexity estimator and tier routers:

```
Loss = task_performance
     + λ₁ × complexity_accuracy     (correct tier prediction)
     + λ₂ × diversity_bonus         (reward cross-architecture routing)
     + λ₃ × efficiency_penalty      (penalize using T4 for T0 tasks)
     + λ₄ × anti_collapse_reg       (prevent ignoring any expert family)
     + λ₅ × cross_modal_alignment   (reward coherent multi-modal chains)
```

Complexity ground truth from:
- **Synthetic labels**: Run each task on T0-T4 separately, label with smallest tier achieving >95% of T4 quality
- **Human labels**: 50K human-labeled complexity ratings
- **Benchmark mapping**: MMLU → T1, GPQA → T3, AIME → T4, etc.

### Phase 4: Omnimodal Integration (2-3 weeks)

Train cross-modal routing and diffusion heads:

1. **Vision**: Fine-tune alignment between Qwen3-VL and text experts
2. **Video**: Train Transfusion-style diffusion head on video-text pairs
3. **3D**: Connect TRELLIS.2/zen-3d experts with vision→3D routing
4. **Audio**: Align Qwen3-Omni audio encoder with text experts

### Phase 5: Expert LoRA Fine-tuning (Optional, 1-2 weeks)

If cross-architecture mixing causes coherence issues:

```python
# Lightweight LoRA on frozen experts
lora_config = LoraConfig(
    r=16,                    # Low rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention only
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
# Target: <0.5% of expert params modified
```

---

## Hardware Requirements

| Phase | Compute | Duration | Est. Cost |
|-------|---------|----------|-----------|
| Expert catalog | 8x A100 80GB | 2 weeks | $5K |
| Expert extraction | 8x A100 80GB | 1 week | $2.5K |
| Alignment training | 64x A100 80GB | 2-3 weeks | $50-75K |
| Router training | 64x A100 80GB | 2-4 weeks | $50-100K |
| Omnimodal integration | 128x A100 80GB | 2-3 weeks | $75-100K |
| Expert LoRA (optional) | 128x A100 80GB | 1-2 weeks | $25-50K |

**Total estimated**: $200K-350K (vs $50M+ for 2T from scratch)

**Memory strategy**: FP8/INT4 quantized experts, gradient checkpointing, expert offloading (only active tier loaded to GPU, others on CPU/NVMe)

---

## Inference Architecture

### Expert Offloading Strategy

At inference, only the active tier's experts are in GPU memory:

```
GPU VRAM (80GB per device):
├── Complexity Estimator: ~200MB (always loaded)
├── Shared Embedding/Output: ~2GB (always loaded)
├── T0 Experts (Qwen 0.8B): ~1.6GB (always loaded — it's tiny)
├── Active Tier Experts: 10-50GB (loaded on demand)
└── KV Cache: remaining VRAM

CPU/NVMe (offloaded):
├── Inactive tier experts
├── Multimodal expert weights
└── LoRA adapters for non-active experts
```

### Speculative Execution

For latency optimization, T0 generates speculatively while T2+ verifies:

```
1. T0 (0.8B) generates 8 tokens speculatively  [fast, ~10ms]
2. If complexity estimator says T0 is sufficient → accept all
3. If escalation needed → T2+ verifies/corrects  [slower, but only when needed]

Result: T0-level latency for 80%+ of queries
```

### Serving Configuration

```yaml
# zen5 inference config
model:
  total_params: "3.1T"
  max_active_params: "100B"
  min_active_params: "0.8B"

tiers:
  T0:
    models: ["qwen35-0.8b"]
    gpu_requirement: "1x A100"
    latency_target: "50ms"
  T1:
    models: ["qwen35-9b"]
    gpu_requirement: "1x A100"
    latency_target: "200ms"
  T2:
    models: ["minimax-m2.5"]
    gpu_requirement: "2x A100"
    latency_target: "1s"
  T3:
    models: ["glm-5"]
    gpu_requirement: "4x A100"
    latency_target: "3s"
  T4:
    models: ["kimi-k2.5", "ling-1t"]
    gpu_requirement: "8x A100"
    latency_target: "10s"

omnimodal:
  vision: "qwen3-vl-30b"
  video: ["wan2.2", "cogvideox"]
  3d: ["trellis", "zen-3d"]
  audio: ["qwen3-omni", "zen-tts"]
```

---

## zen5 Model Lineup

| Model | Expert Pool | Total Params | Active Range | Target |
|-------|-------------|-------------|-------------|--------|
| **zen5** | Qwen3.5 0.8B + 9B, GLM-5 | 750B | 0.8-50B | General |
| **zen5-coder** | + Kimi K2.5 code experts | 1.8T | 0.8-80B | Code |
| **zen5-omni** | + Vision + Video + 3D + Audio | 2.5T | 0.8-100B | Omnimodal |
| **zen5-max** | All sources, all modalities | 3.1T | 0.8-100B+ | Frontier |

### zen5 (General, 750B)
- Text-only, complexity-routed
- Qwen3.5-0.8B (T0) + Qwen3.5-9B (T1) + GLM-5 (T2-T3)
- Good for: Chat, summarization, general Q&A, translation

### zen5-coder (Code, 1.8T)
- Text + code, complexity-routed
- Adds Kimi K2.5 code experts (384 specialists) for T4
- Good for: Code generation, debugging, architecture, SWE-bench

### zen5-omni (Omnimodal, 2.5T)
- Text + vision + video + 3D + audio
- Adds Qwen3-VL, Wan2.2, CogVideoX, TRELLIS.2, zen-3d, Qwen3-Omni
- Good for: "Describe this video", "Generate a 3D model of X", multimodal reasoning

### zen5-max (Frontier, 3.1T)
- Everything: all text experts + all modalities
- Full expert pool with all tiers active
- Good for: Frontier capability, research, agentic multi-modal workflows

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Token/vocab misalignment across 6 architectures | Shared embedding + per-family alignment adapters |
| Complexity estimator mislabels tier | Adaptive escalation (can promote mid-generation) |
| Expert interference across architectures | Frozen experts + lightweight LoRA for coherence |
| GPU memory for 3T+ model | Tiered offloading — only active tier on GPU |
| Latency for high-tier routing | T0 speculative execution + KV cache transfer |
| Cross-modal generation quality | Separate diffusion heads per modality, trained on paired data |
| Routing collapse (ignoring some families) | Anti-collapse regularization + minimum diversity constraint |
| Diffusion + autoregressive incompatibility | Transfusion architecture proven at 7B (Meta) — we scale it |

---

## Comparison with Existing Models

| Model | Params | Active | Modalities | Complexity Routing | Diverse Experts |
|-------|--------|--------|-----------|-------------------|----------------|
| Llama 4 Behemoth | 2T | 288B | Text | No | No (homogeneous) |
| GPT-5 (est.) | ~2T | ~200B | Omni | Unknown | Unknown |
| Gemini 2.0 Ultra | ~2T | Unknown | Omni | No | No |
| **zen5-max** | **3.1T** | **0.8-100B+** | **Omni** | **Yes** | **Yes (6 families)** |

zen5's key differentiator: **adaptive compute**. Behemoth always uses 288B active. zen5 uses 0.8B when that's enough.

---

## Implementation Roadmap

### Q1 2026 (Now → March)
- [x] Architecture design document (this file)
- [ ] Expert extraction tooling
- [ ] Complexity estimator prototype
- [ ] Alignment layer design

### Q2 2026 (April → June)
- [ ] Expert extraction from all 6 text models
- [ ] Alignment layer training (64x A100, 2-3 weeks)
- [ ] Complexity router prototype + evaluation
- [ ] Vision expert integration (Qwen3-VL)

### Q3 2026 (July → September)
- [ ] Full router training (64x A100, 2-4 weeks)
- [ ] Omnimodal integration (video, 3D, audio)
- [ ] zen5 and zen5-coder release
- [ ] Benchmark against Llama 4, GPT-5, Gemini 2.0

### Q4 2026 (October → December)
- [ ] zen5-omni and zen5-max release
- [ ] Speculative execution optimization
- [ ] Expert LoRA fine-tuning (if needed)
- [ ] zen5: largest and most efficient open-source model

---

## Key Research References

- **HMoE** (EMNLP 2025) — Heterogeneous expert sizes outperform homogeneous
- **Symbolic-MoE** (arXiv 2503.05641) — Skill-based routing, 8.15% gain over GPT-4o-mini
- **Transfusion** (Meta 2024) — Unified autoregressive + diffusion in single model
- **DeepSeek-V3** — 671B MoE with 384 experts, top-8 routing
- **Wan2.2** — MoE diffusion for video generation
- **CogVideoX** (ICLR 2025) — Expert transformer with 3D Causal VAE
- **TripoSR** — Feed-forward 3D from single image, MIT licensed

---

*zen5: Mixture of Diverse Experts — Clarity Through Diversity*
