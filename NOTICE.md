# NOTICE

`zen5` is a fork of [`antirez/ds4`](https://github.com/antirez/ds4) ("DwarfStar 4"), released under the MIT License. All upstream copyright notices in [`LICENSE`](LICENSE) and source headers are retained unchanged.

## Upstream lineage

```
zen5  ←  antirez/ds4  ←  ggml-org/llama.cpp (GGUF formats, kernel ideas)
```

* `ds4.c` and friends © 2026 The ds4.c authors (MIT).
* GGML quant layouts, CPU dot kernels, and select adapted pieces © 2023–2026 The ggml authors (MIT). See [`LICENSE`](LICENSE).

## Tracking upstream

```sh
git remote -v          # upstream → https://github.com/antirez/ds4
git fetch upstream
git merge upstream/main   # or rebase, depending on policy
```

The remote is named `upstream` (not `origin`) so that a `git push` to a future zenlm `origin` cannot accidentally publish back to antirez's repository.

## What changed in zen5 vs ds4

* Project rebrand: README, NOTICE, binary aliases (`zen5`, `zen5-server`, `zen5-bench` alongside `ds4*`).
* `README-UPSTREAM.md` preserves the upstream README verbatim.
* `LICENSE` is unchanged.
* Source files (`ds4*.c`, `ds4*.h`, `ds4*.m`, `ds4*.cu`, Metal shaders) are unchanged, so upstream pulls apply cleanly.

Anything beyond renames/branding lives behind a clear commit message so it is auditable against upstream `main`.

## Reporting issues

* **Engine bugs that are not zen5-specific:** report upstream at https://github.com/antirez/ds4/issues — that helps everyone.
* **zenlm-specific integrations, GGUF variants, or rebrand issues:** report in the zenlm repo for this fork.
