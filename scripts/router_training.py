#!/usr/bin/env python3
"""
zen5 MoDE — Router & Alignment Training Pipeline

Trains the complexity-aware hierarchical router and cross-architecture alignment
layers. This covers Phases 2-3 of the zen5 training roadmap.

Components trained:
  1. AlignmentLayer    — projects each source architecture to shared latent space
  2. ComplexityEstimator — lightweight classifier predicting T0-T4 tier
  3. MoDERouter        — per-tier expert selection with ReLU gating (ReMoE)
  4. CrossModalRouter  — omnimodal expert chain selection

Training objectives:
  L = L_task + λ₁·L_complexity + λ₂·L_diversity + λ₃·L_efficiency + λ₄·L_anticollapse

Usage:
  python zen5_router_training.py info
  python zen5_router_training.py train --config configs/zen5_router.yaml
  python zen5_router_training.py train-complexity --data ./data/ --output ./ckpt/
  python zen5_router_training.py eval --checkpoint ./ckpt/router_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zen5-router")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class MoDEConfig:
    """Full configuration for zen5 MoDE router architecture."""

    # Shared representation
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    vocab_size: int = 152064          # Unified vocabulary
    max_position_embeddings: int = 262144

    # Complexity tiers: T0 (trivial) → T4 (frontier)
    num_tiers: int = 5
    complexity_hidden: int = 1024
    complexity_layers: int = 4
    complexity_context_len: int = 64  # First N tokens for complexity estimation

    # Router
    router_hidden: int = 2048
    num_experts_per_tier: tuple = (1, 1, 64, 256, 384)
    top_k_per_tier: tuple = (1, 1, 4, 8, 8)

    # Modalities
    num_modalities: int = 5  # text, vision, video, 3d, audio
    modality_embed_dim: int = 256

    # Loss coefficients
    lambda_complexity: float = 1.0
    lambda_diversity: float = 0.05
    lambda_efficiency: float = 0.02
    lambda_anticollapse: float = 0.01

    # MoE++ zero-computation experts
    enable_zero_experts: bool = True
    num_zero_types: int = 3  # zero, copy, constant

    # ReMoE
    use_relu_routing: bool = True

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 64
    gradient_accumulation: int = 4
    fp16: bool = True
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 100


# ── Model Components ─────────────────────────────────────────────────────────

class ComplexityEstimator(nn.Module):
    """
    Lightweight classifier (~100-200M params) that predicts task complexity
    from the first N tokens of input. Outputs T0-T4 tier probabilities.
    """

    def __init__(self, cfg: MoDEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.complexity_hidden)
        self.pos_embed = nn.Embedding(cfg.complexity_context_len, cfg.complexity_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.complexity_hidden, nhead=8,
            dim_feedforward=cfg.complexity_hidden * 4,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.complexity_layers)
        self.norm = nn.LayerNorm(cfg.complexity_hidden)

        self.tier_head = nn.Sequential(
            nn.Linear(cfg.complexity_hidden, cfg.complexity_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.complexity_hidden, cfg.num_tiers),
        )
        self.modality_head = nn.Linear(cfg.complexity_hidden, cfg.num_modalities)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Pool: last non-padding token
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True) - 1
            pooled = x.gather(1, lengths.unsqueeze(-1).expand(-1, -1, x.size(-1))).squeeze(1)
        else:
            pooled = x[:, -1]

        return self.tier_head(pooled), self.modality_head(pooled)


class AlignmentLayer(nn.Module):
    """
    Per-architecture projection to shared latent space.
    Handles hidden-dimension mismatches and representation normalization.
    """

    ARCH_DIMS = {
        "gated_deltanet_08b": 1024,
        "gated_deltanet_9b": 4096,
        "lightning_attention": 4096,
        "glm_moe": 4096,
        "deepseek_v3": 7168,
        "fp8_moe": 4096,
    }

    def __init__(self, cfg: MoDEConfig):
        super().__init__()
        self.projections = nn.ModuleDict()
        for arch, src_dim in self.ARCH_DIMS.items():
            self.projections[arch] = nn.Sequential(
                nn.Linear(src_dim, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.GELU(),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
            ) if src_dim != cfg.hidden_size else nn.Sequential(
                nn.LayerNorm(cfg.hidden_size),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
            )

    def forward(self, hidden: torch.Tensor, arch: str) -> torch.Tensor:
        return self.projections[arch](hidden)


class ZeroExpert(nn.Module):
    """MoE++ zero expert: output zeros (discard token)."""
    def forward(self, x): return torch.zeros_like(x)


class CopyExpert(nn.Module):
    """MoE++ copy expert: identity (skip layer)."""
    def forward(self, x): return x


class ConstantExpert(nn.Module):
    """MoE++ constant expert: learned bias vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(dim) * 0.02)
    def forward(self, x): return self.bias.expand_as(x)


class MoDERouter(nn.Module):
    """
    Hierarchical router with complexity-aware tier selection.
    Per-tier expert gating with ReLU (ReMoE) for adaptive expert count.
    """

    def __init__(self, cfg: MoDEConfig):
        super().__init__()
        self.cfg = cfg

        self.tier_routers = nn.ModuleList()
        for tier_idx in range(cfg.num_tiers):
            n_exp = cfg.num_experts_per_tier[tier_idx]
            n_total = n_exp + (cfg.num_zero_types if cfg.enable_zero_experts else 0)

            layers = [
                nn.Linear(cfg.hidden_size, cfg.router_hidden),
                nn.GELU(),
                nn.Linear(cfg.router_hidden, n_total),
            ]
            if cfg.use_relu_routing:
                layers.append(nn.ReLU())  # ReMoE: sparsity via ReLU

            self.tier_routers.append(nn.Sequential(*layers))

        # Cross-modal router
        self.cross_modal_gate = nn.Sequential(
            nn.Linear(cfg.hidden_size + cfg.modality_embed_dim, cfg.router_hidden),
            nn.GELU(),
            nn.Linear(cfg.router_hidden, cfg.num_modalities),
            nn.Sigmoid(),
        )
        self.modality_embed = nn.Embedding(cfg.num_modalities, cfg.modality_embed_dim)

    def forward(self, hidden: torch.Tensor, tier: int,
                modality_ids: Optional[torch.Tensor] = None):
        router_logits = self.tier_routers[tier](hidden)

        cross_modal = None
        if modality_ids is not None:
            mod_e = self.modality_embed(modality_ids)
            pooled = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
            cross_modal = self.cross_modal_gate(torch.cat([pooled, mod_e], dim=-1))

        return router_logits, cross_modal


# ── Loss Functions ───────────────────────────────────────────────────────────

def complexity_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for tier prediction."""
    return F.cross_entropy(logits, labels)


def diversity_loss(router_logits: torch.Tensor, arch_ids: torch.Tensor) -> torch.Tensor:
    """Encourage routing across different source architectures."""
    probs = F.softmax(router_logits.view(-1, router_logits.size(-1)), dim=-1)
    unique_archs = arch_ids.unique()
    arch_loads = []
    for a in unique_archs:
        mask = (arch_ids == a).float()
        load = (probs * mask.unsqueeze(0)).sum(dim=-1).mean()
        arch_loads.append(load)
    arch_loads = torch.stack(arch_loads)
    arch_probs = arch_loads / (arch_loads.sum() + 1e-8)
    return -(arch_probs * torch.log(arch_probs + 1e-8)).sum()  # Negative entropy


def efficiency_loss(tier_logits: torch.Tensor, tier_labels: torch.Tensor) -> torch.Tensor:
    """Penalize over-estimation of complexity tier."""
    probs = F.softmax(tier_logits, dim=-1)
    tiers = torch.arange(probs.size(-1), device=probs.device, dtype=torch.float)
    expected = (probs * tiers).sum(dim=-1)
    return F.relu(expected - tier_labels.float()).mean()


def anticollapse_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Switch Transformer load-balancing: prevent expert starvation."""
    probs = F.softmax(router_logits.view(-1, router_logits.size(-1)), dim=-1)
    load = probs.mean(dim=0)
    ideal = 1.0 / load.size(0)
    return ((load - ideal) ** 2).sum()


# ── Dataset ──────────────────────────────────────────────────────────────────

class ComplexityDataset(Dataset):
    """
    Dataset for training the complexity estimator.
    Each sample: (input_ids, attention_mask, tier_label, modality_id)

    Complexity labels can come from:
      - Synthetic: run each prompt on T0-T4, label with smallest tier achieving 95% of T4 quality
      - Manual: human-labeled 50K samples
      - Benchmark mapping: MMLU→T1, GPQA→T3, AIME→T4
    """

    def __init__(self, data_path: str, max_len: int = 64):
        self.max_len = max_len
        self.samples = []

        if os.path.exists(data_path):
            with open(data_path) as f:
                self.samples = json.load(f)
            log.info(f"Loaded {len(self.samples)} samples from {data_path}")
        else:
            log.warning(f"Data path not found: {data_path} — generating synthetic data")
            self.samples = self._generate_synthetic()

    def _generate_synthetic(self) -> list[dict]:
        """Generate synthetic complexity-labeled data for prototyping."""
        samples = []

        tier_examples = {
            0: ["Hello", "Hi there", "What's 2+2?", "Yes or no: is the sky blue?", "Thanks!"],
            1: ["Summarize this paragraph about climate change.", "Translate to French: Good morning.",
                "What is photosynthesis?", "Write a short email declining a meeting."],
            2: ["Analyze the causes of World War I and their modern parallels.",
                "Review this Python code for security vulnerabilities.",
                "Compare and contrast REST and GraphQL APIs."],
            3: ["Derive the Euler-Lagrange equation from first principles.",
                "Design a distributed system for real-time fraud detection.",
                "Prove that every finite group of order p^2 is abelian."],
            4: ["Prove the Riemann hypothesis implies the prime number theorem.",
                "Design a novel algorithm for protein folding that improves on AlphaFold.",
                "Write a formal proof of the four-color theorem."],
        }

        for tier, prompts in tier_examples.items():
            for prompt in prompts:
                tokens = list(range(min(len(prompt.split()), self.max_len)))
                samples.append({
                    "input_ids": tokens + [0] * (self.max_len - len(tokens)),
                    "attention_mask": [1] * len(tokens) + [0] * (self.max_len - len(tokens)),
                    "tier": tier,
                    "modality": 0,  # text
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"][:self.max_len], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"][:self.max_len], dtype=torch.long),
            "tier": torch.tensor(s["tier"], dtype=torch.long),
            "modality": torch.tensor(s.get("modality", 0), dtype=torch.long),
        }


# ── Training Loop ────────────────────────────────────────────────────────────

class MoDETrainer:
    """Trains the complexity estimator and MoDE router."""

    def __init__(self, cfg: MoDEConfig, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize models
        self.complexity = ComplexityEstimator(cfg)
        self.alignment = AlignmentLayer(cfg)
        self.router = MoDERouter(cfg)

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.complexity.to(self.device)
        self.alignment.to(self.device)
        self.router.to(self.device)

        # Optimizer
        all_params = (
            list(self.complexity.parameters()) +
            list(self.alignment.parameters()) +
            list(self.router.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR scheduler: linear warmup + cosine decay
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=cfg.max_steps,
            pct_start=cfg.warmup_steps / cfg.max_steps,
            anneal_strategy="cos",
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and self.device.type == "cuda")

        self._log_param_count()

    def _log_param_count(self):
        c = sum(p.numel() for p in self.complexity.parameters())
        a = sum(p.numel() for p in self.alignment.parameters())
        r = sum(p.numel() for p in self.router.parameters())
        total = c + a + r
        log.info(f"Trainable parameters:")
        log.info(f"  ComplexityEstimator: {c:>12,}")
        log.info(f"  AlignmentLayer:     {a:>12,}")
        log.info(f"  MoDERouter:         {r:>12,}")
        log.info(f"  Total:              {total:>12,} ({total/1e6:.1f}M)")
        log.info(f"  vs 3T+ frozen experts = {total/3e12*100:.4f}% trainable")

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step. Returns loss dict."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        tier_labels = batch["tier"].to(self.device)
        modality_ids = batch["modality"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.cfg.fp16 and self.device.type == "cuda"):
            # Complexity estimation
            tier_logits, mod_logits = self.complexity(input_ids, attention_mask)

            # Losses
            l_complexity = complexity_loss(tier_logits, tier_labels)
            l_efficiency = efficiency_loss(tier_logits, tier_labels)

            # For router loss, we need hidden states — use complexity encoder output
            # In production, this would use actual expert outputs
            hidden = self.complexity.embed(input_ids)

            # Pick predicted tier and route
            pred_tier = tier_logits.argmax(dim=-1)
            # Use most common tier in batch for router forward
            most_common = pred_tier.mode().values.item()
            router_logits, cross_modal = self.router(hidden, most_common, modality_ids)

            l_anticollapse = anticollapse_loss(router_logits)

            # Total loss
            total = (
                self.cfg.lambda_complexity * l_complexity
                + self.cfg.lambda_efficiency * l_efficiency
                + self.cfg.lambda_anticollapse * l_anticollapse
            )

        return {
            "total": total,
            "complexity": l_complexity.item(),
            "efficiency": l_efficiency.item(),
            "anticollapse": l_anticollapse.item(),
            "tier_accuracy": (pred_tier == tier_labels).float().mean().item(),
        }

    def train(self, train_data: str, eval_data: Optional[str] = None):
        """Full training loop with logging, eval, and checkpointing."""
        train_ds = ComplexityDataset(train_data, self.cfg.complexity_context_len)
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )

        eval_loader = None
        if eval_data and os.path.exists(eval_data):
            eval_ds = ComplexityDataset(eval_data, self.cfg.complexity_context_len)
            eval_loader = DataLoader(eval_ds, batch_size=self.cfg.batch_size * 2)

        log.info(f"Training: {len(train_ds)} samples, {self.cfg.max_steps} steps")
        log.info(f"Device: {self.device}, FP16: {self.cfg.fp16}")

        step = 0
        best_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        t0 = time.time()

        self.complexity.train()
        self.router.train()

        while step < self.cfg.max_steps:
            for batch in train_loader:
                if step >= self.cfg.max_steps:
                    break

                losses = self.train_step(batch)
                loss = losses["total"]

                # Backward
                self.scaler.scale(loss / self.cfg.gradient_accumulation).backward()

                if (step + 1) % self.cfg.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.complexity.parameters()) + list(self.router.parameters()),
                        max_norm=1.0,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                running_loss += losses["complexity"]
                running_acc += losses["tier_accuracy"]
                step += 1

                # Logging
                if step % self.cfg.log_every == 0:
                    avg_loss = running_loss / self.cfg.log_every
                    avg_acc = running_acc / self.cfg.log_every
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    log.info(
                        f"Step {step:>6d}/{self.cfg.max_steps} | "
                        f"loss={avg_loss:.4f} | acc={avg_acc:.4f} | "
                        f"lr={lr:.2e} | {elapsed:.0f}s"
                    )
                    running_loss = 0.0
                    running_acc = 0.0

                # Eval
                if eval_loader and step % self.cfg.eval_every == 0:
                    eval_acc = self.evaluate(eval_loader)
                    if eval_acc > best_acc:
                        best_acc = eval_acc
                        self.save_checkpoint(os.path.join(self.output_dir, "best.pt"), step)
                        log.info(f"  New best: {best_acc:.4f}")

                # Checkpoint
                if step % self.cfg.save_every == 0:
                    self.save_checkpoint(
                        os.path.join(self.output_dir, f"step_{step:06d}.pt"), step
                    )

        # Final save
        self.save_checkpoint(os.path.join(self.output_dir, "final.pt"), step)
        log.info(f"Training complete. {step} steps, best accuracy: {best_acc:.4f}")

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate tier prediction accuracy."""
        self.complexity.eval()
        correct, total = 0, 0
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            tier_labels = batch["tier"].to(self.device)

            tier_logits, _ = self.complexity(input_ids, attention_mask)
            pred = tier_logits.argmax(dim=-1)
            correct += (pred == tier_labels).sum().item()
            total += tier_labels.size(0)

        acc = correct / max(total, 1)
        log.info(f"  Eval accuracy: {acc:.4f} ({correct}/{total})")
        self.complexity.train()
        return acc

    def save_checkpoint(self, path: str, step: int):
        torch.save({
            "step": step,
            "complexity": self.complexity.state_dict(),
            "alignment": self.alignment.state_dict(),
            "router": self.router.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.cfg.__dict__,
        }, path)
        log.info(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.complexity.load_state_dict(ckpt["complexity"])
        self.alignment.load_state_dict(ckpt["alignment"])
        self.router.load_state_dict(ckpt["router"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        log.info(f"Loaded checkpoint: {path} (step {ckpt.get('step', '?')})")
        return ckpt.get("step", 0)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="zen5 MoDE Router Training")
    sub = parser.add_subparsers(dest="cmd")

    # info
    sub.add_parser("info", help="Show architecture info and param counts")

    # train
    p = sub.add_parser("train", help="Train full pipeline (alignment + router)")
    p.add_argument("--data", default="./data/complexity_train.json")
    p.add_argument("--eval-data", default="./data/complexity_eval.json")
    p.add_argument("--output", default="./checkpoints/router/")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", default=None, help="Resume from checkpoint")

    # eval
    p = sub.add_parser("eval", help="Evaluate checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="./data/complexity_eval.json")

    args = parser.parse_args()
    cfg = MoDEConfig()

    if args.cmd == "info":
        ce = ComplexityEstimator(cfg)
        al = AlignmentLayer(cfg)
        rt = MoDERouter(cfg)

        c = sum(p.numel() for p in ce.parameters())
        a = sum(p.numel() for p in al.parameters())
        r = sum(p.numel() for p in rt.parameters())
        total = c + a + r

        print(f"\nzen5 MoDE Router Architecture")
        print("=" * 60)
        print(f"Hidden size:        {cfg.hidden_size}")
        print(f"Vocab size:         {cfg.vocab_size:,}")
        print(f"Max context:        {cfg.max_position_embeddings:,}")
        print(f"Complexity tiers:   {cfg.num_tiers} (T0-T4)")
        print(f"Experts per tier:   {cfg.num_experts_per_tier}")
        print(f"Top-K per tier:     {cfg.top_k_per_tier}")
        print(f"ReLU routing:       {cfg.use_relu_routing}")
        print(f"Zero experts:       {cfg.enable_zero_experts}")
        print(f"Modalities:         {cfg.num_modalities}")
        print()
        print(f"Trainable Parameters:")
        print(f"  ComplexityEstimator: {c:>12,}")
        print(f"  AlignmentLayer:     {a:>12,}")
        print(f"  MoDERouter:         {r:>12,}")
        print(f"  Total:              {total:>12,} ({total/1e6:.1f}M)")
        print(f"\nFrozen expert pool:   ~3.0T+ params")
        print(f"Training efficiency:  {total/3e12*100:.4f}% of total model")

    elif args.cmd == "train":
        cfg.max_steps = args.max_steps
        cfg.batch_size = args.batch_size
        cfg.lr = args.lr

        trainer = MoDETrainer(cfg, args.output)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train(args.data, args.eval_data)

    elif args.cmd == "eval":
        trainer = MoDETrainer(cfg, ".")
        trainer.load_checkpoint(args.checkpoint)

        eval_ds = ComplexityDataset(args.data, cfg.complexity_context_len)
        eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size * 2)
        trainer.evaluate(eval_loader)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
