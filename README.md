# Zen5

**Zen5** is the [zenlm](https://zenlm.org) inference engine for large-context, low-active-parameter MoE models on Apple Silicon and CUDA.

It is a fork of [`antirez/ds4`](https://github.com/antirez/ds4) ("DwarfStar 4"), Salvatore Sanfilippo's self-contained native inference engine for **DeepSeek V4 Flash**, repackaged as the engine for the `zen5` model line that follows `zen4-{nano,micro,mini,full}`.

We preserve full upstream attribution: see [`NOTICE.md`](NOTICE.md), [`LICENSE`](LICENSE), and the in-place comments. The original upstream README is preserved verbatim as [`README-UPSTREAM.md`](README-UPSTREAM.md).

> Status: **alpha**. Same alpha status as upstream — inference and model serving are complicated, and the engine targets one model layout at a time.

## What zen5 inherits from ds4

* **Metal-first** on Apple Silicon (96 GB+ memory class), CUDA on Linux (DGX Spark / GB10 and generic).
* Self-contained C implementation — no GGML link at runtime, but the GGUF/quant lineage is acknowledged in [`LICENSE`](LICENSE).
* **2-bit quantization** that actually works for agent/tool use: routed MoE experts at `IQ2_XXS` (up/gate) / `Q2_K` (down), shared experts and projections untouched.
* **On-disk KV cache** with SHA1-keyed rendered-prefix lookup, byte-exact tool-call replay via DSML — survives server restarts.
* **OpenAI / Anthropic / Responses-compatible HTTP server** with SSE streaming, thinking-mode separation, and tool-call canonicalization.
* **1M-token context window** ceiling (model card max); ~100k–300k recommended on 128 GB.
* Steering via single activation directions (refusal-direction-style).

## Build

```sh
make           # macOS Metal (default) → ./zen5, ./zen5-server, ./zen5-bench
make cuda-spark    # Linux CUDA, DGX Spark / GB10
make cuda-generic  # Linux CUDA, other local CUDA GPUs
make cpu       # CPU diagnostics (do not run real inference on macOS — see upstream README)
make test
```

The Makefile still produces `ds4`, `ds4-server`, `ds4-bench` as primary artifacts for upstream-merge compatibility; `make` then creates `zen5*` symlinks alongside them, so either name works.

## Run

After building, download the DeepSeek V4 Flash weights (96/128 GB RAM class recommended):

```sh
./download_model.sh q2-imatrix
./zen5 -p "Hello in three sentences."
```

For the HTTP server with disk KV cache:

```sh
./zen5-server --ctx 100000 --kv-disk-dir /tmp/zen5-kv --kv-disk-space-mb 8192
```

For full CLI flags, model variants (q2 / q2-imatrix / q4 / q4-imatrix / mtp), tool-call wiring (opencode / Pi / Codex / Claude Code), the disk-KV-cache format, and debugging tooling, see [`README-UPSTREAM.md`](README-UPSTREAM.md).

## Where zen5 will diverge from upstream

Tracked separately so upstream pulls stay clean:

* zenlm-branded GGUF assets and the option to point `download_model.sh` at zenlm-hosted weights.
* Steering vectors tuned for the zenlm agent stack and the persona used by `zen-coder` / `zen-agent`.
* Server defaults aligned with the zenlm coding-agent integrations (Claude Code, Codex CLI, Pi, opencode).
* Quant variants validated against the zen-eval harness.

Anything not in that list should be sent upstream first.

## Credits

* **Upstream:** [antirez/ds4](https://github.com/antirez/ds4) — Salvatore Sanfilippo and the ds4.c authors.
* **Lineage:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and contributors. GGUF quant formats, dot kernels, and engineering knowledge that made this possible. Their copyright is preserved in [`LICENSE`](LICENSE).
* **Model:** [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai) — the model this engine serves.
