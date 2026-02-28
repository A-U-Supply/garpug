# `tune` Subcommand — GPT-2 Fine-Tuning Design

## Overview

Adds a `tune` subcommand that fine-tunes pretrained GPT-2 on a book. This complements the existing `train` command (character-level from scratch). Fine-tuned output is more coherent; char-level output is stranger. Both stay available.

## Architecture

### New files
- `src/gpt2.rs` — GPT-2 model with HuggingFace-compatible weight names
- `src/tune.rs` — Fine-tuning loop (download, tokenize, train, save)

### Modified files
- `src/cli.rs` — `Tune` variant + `TuneArgs`
- `src/main.rs` — `Tune` match arm, `Prompt` auto-detects model type via `config.json`
- `src/train.rs` — Saves `"model_type": "char"` in config.json

## Key decisions

1. **GPT-2 small only** — 124M params, fits in 16 GB RAM with batch_size=2, context=256
2. **Separate model module** — GPT-2 uses fused QKV (Conv1D) vs char-level's per-head attention. Mixing architectures would be messy.
3. **Conv1D transpose on load** — HuggingFace stores Conv1D weights as `[in, out]`; candle Linear expects `[out, in]`. Transposed during `load_pretrained()`.
4. **Weight tying** — `lm_head.weight` copied from `transformer.wte.weight` after loading pretrained weights.
5. **BPE tokenizer** — Uses `tokenizers` crate to load HuggingFace's tokenizer.json (50257 vocab).
6. **Auto-detect in prompt** — `config.json` includes `"model_type"` field (`"char"` or `"gpt2"`). Prompt command loads the right model + tokenizer.

## Disk impact

- ~600 MB: GPT-2 weights + tokenizer cached in `~/.cache/huggingface/`
- ~548 MB: Fine-tuned model saved to output dir

## Training config

- Optimizer: AdamW, lr=3e-5 (10x lower than char-level training)
- Batch size: 2, context: 256 (conservative for 16 GB RAM)
- Default epochs: 50
