# Garpug Design

A Rust CLI for training small character-level GPT models on individual books and generating text from them. Built for art projects — the weird, nonsensical output is the feature.

## CLI Interface

Two subcommands:

```
garpug train --input <file_or_url> --size medium [--dim N] [--layers N] [--heads N] [--context N] [--epochs N] [--output model.bin]
garpug prompt --model model.bin [--length 500] [--temperature 1.0] "seed text"
```

### `train`
- Takes a text file path or Project Gutenberg URL
- Tokenizes at character level, trains a GPT, saves model weights
- Prints loss every N steps

### `prompt`
- Loads a saved model, takes seed text, generates continuation
- `--temperature` controls randomness (higher = weirder)
- Output to stdout

## Model Presets

| Preset | Dim | Layers | Heads | Context | ~Model Size | ~Train Time (M1 Pro) |
|--------|-----|--------|-------|---------|-------------|----------------------|
| small  | 128 | 4      | 4     | 256     | ~5 MB       | ~2-5 min             |
| medium | 384 | 8      | 6     | 512     | ~40 MB      | ~10-20 min           |
| large  | 768 | 12     | 12    | 1024    | ~200 MB     | ~1-2 hrs             |
| xl     | 1024| 16     | 16    | 1024    | ~500 MB     | ~4-8 hrs             |

Default: `medium`. Individual flags (`--dim`, `--layers`, `--heads`, `--context`) override preset values.

## Tech Stack

- **Language**: Rust
- **ML framework**: `candle` (HuggingFace) with Metal backend for M1 Pro GPU acceleration
- **CLI**: `clap` for argument parsing
- **Tokenization**: Character-level (each unique byte maps to an integer)

## Model Architecture

Small GPT (decoder-only transformer):
- Token + positional embeddings
- N transformer blocks, each with:
  - Multi-head causal self-attention
  - Feed-forward network (4x expansion)
  - Layer normalization
  - Residual connections
  - Dropout
- Final layer norm + linear projection to vocabulary size

Optimizer: AdamW
Loss: Cross-entropy (next character prediction)

## Data Pipeline

1. **Fetch**: Download from URL or read from file path
2. **Clean**: Strip Project Gutenberg header/footer boilerplate
3. **Tokenize**: Build character vocabulary, map chars to integers, save vocab
4. **Split**: 90% train, 10% validation
5. **Batch**: Random chunks of `context` length

## Output Artifacts

All saved to a single output directory (default `./output/`):
- `model.bin` — trained weights (safetensors format)
- `vocab.json` — character-to-integer mapping
- `config.json` — model hyperparameters

## Disk Budget

- Cargo build cache: ~150-200 MB (one-time)
- Compiled binary: ~20 MB
- Training data: ~2 MB
- Model checkpoint (medium): ~40 MB
- **Total: ~300 MB**

Machine: M1 Pro MacBook Pro, 16 GB RAM, ~20 GB free disk.

## Design Decisions

- **Character-level tokenization**: Zero tokenizer complexity, works with any language/text, vocabulary is just the unique characters in the input (~100-200 for English).
- **Candle over burn/ndarray**: Metal GPU support with clean Rust API, most mature for transformers, builds with standard `cargo build`.
- **Presets + overrides**: Human-readable size names for quick use, individual flags for experimentation.
- **Single book scope**: Deliberately small models that will overfit — producing fluent remixes of the source text rather than coherent generalization. This is the desired behavior for art projects.
