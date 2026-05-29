# zen5 — engine fork of antirez/ds4

**Path:** `/Users/a/work/zen/zen5`
**Purpose:** zenlm-branded inference engine for the zen5 model line. Fork of [`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4 — DeepSeek V4 Flash native engine).

## State

* MIT-licensed fork. Upstream copyrights preserved in [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
* Git remote: `upstream → antirez/ds4` (no `origin` yet — set when the zenlm fork repo exists).
* Build: `make` on macOS Metal produces `ds4`, `ds4-server`, `ds4-bench` plus `zen5*` symlinks.
* Model weights are NOT in this repo. Use `./download_model.sh q2-imatrix` to fetch ~81 GB of GGUF into `./gguf/`.

## Quick orient

| File | What it is |
|------|------------|
| [`README.md`](README.md) | zen5-branded readme |
| [`README-UPSTREAM.md`](README-UPSTREAM.md) | original antirez/ds4 README, verbatim |
| [`NOTICE.md`](NOTICE.md) | attribution, fork policy, how to track upstream |
| [`LICENSE`](LICENSE) | MIT — `ds4.c authors` + `ggml authors` (unchanged) |
| [`Makefile`](Makefile) | Metal / CUDA / CPU builds |
| `ds4.c` (800 KB) | core inference loop |
| `ds4_metal.m` (640 KB) | Metal backend |
| `ds4_cuda.cu` (465 KB) | CUDA backend |
| `ds4_server.c` (550 KB) | OpenAI/Anthropic/Responses HTTP server |
| `ds4_cli.c` (52 KB) | interactive CLI with linenoise |
| `gguf-tools/` | offline GGUF gen, imatrix collection, quality testing |
| `dir-steering/` | activation-direction steering data |
| `speed-bench/` | benchmark CSVs and plots |
| `tests/test-vectors/` | official DeepSeek V4 Flash continuation vectors |

## Sibling context

* `~/work/zen/LLM.md` — zen ecosystem index.
* `~/work/zen/models/` — model weight directories (`zen4`, `zen4-nano`, etc.).
* Future zenlm `origin` for this fork: TBD (likely `zenlm/zen5`).
