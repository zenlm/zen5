#!/usr/bin/env python3
"""
Export the alignment-study tensors so the SAME experiment can be run on the Rust stack.

The point is verification, not convenience: the PyTorch harness already produced
linear=0.450 / random=0.571 / real=0.521 for Qwen3-0.6B <- SmolLM2-360M. If a pure
hanzo-ml/hanzo-nn implementation reproduces those numbers from these exact tensors, the
Rust training stack is *verified against a known answer* rather than merely asserted to work.

Writes a safetensors file containing:
  x            [N, d_host]      host MLP input activations (real text)
  y            [N, d_host]      host MLP output = the regression target
  gate,up,down [.., ..]         the frozen donor SwiGLU expert's weights
"""

from __future__ import annotations

import argparse
import sys

import torch
from safetensors.torch import save_file

from alignment_premise import collect_host, get_text, load_expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--expert", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--out", default="alignment_data.safetensors")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = get_text(a.tokens * 8)
    x, y, d_host, layer = collect_host(a.host, -1, text, a.tokens, device)
    e = load_expert(a.expert, -1, device)

    tensors = {
        "x": x.contiguous().cpu(),
        "y": y.contiguous().cpu(),
        "gate": e.gate.weight.data.contiguous().cpu(),
        "up": e.up.weight.data.contiguous().cpu(),
        "down": e.down.weight.data.contiguous().cpu(),
    }
    meta = {"host": a.host, "expert": a.expert, "host_layer": str(layer),
            "d_host": str(d_host), "d_exp": str(e.d), "n": str(x.shape[0])}
    save_file(tensors, a.out, metadata=meta)
    print(f"[export] {a.out}")
    for k, v in tensors.items():
        print(f"   {k:5s} {tuple(v.shape)} {v.dtype}")
    print(f"   meta: {meta}")


if __name__ == "__main__":
    sys.exit(main())
