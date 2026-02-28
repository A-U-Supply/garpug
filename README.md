# garpug

Train GPT models on individual books. Generate weird text for art projects.

Two modes: **train** a character-level model from scratch (stranger output) or **tune** a pretrained GPT-2 (more coherent output). Both stay available — it's a tradeoff, not an upgrade.

## Build

```bash
# With Metal GPU acceleration (macOS, recommended)
cargo build --release --features metal

# CPU only
cargo build --release
```

## Usage

### Train a character-level model (from scratch)

```bash
# From a Project Gutenberg URL
garpug train --input "https://www.gutenberg.org/cache/epub/1200/pg1200.txt" --size medium

# From a local file
garpug train --input book.txt --size large --epochs 100
```

### Fine-tune GPT-2

```bash
# Fine-tune pretrained GPT-2 on a book
garpug tune --input "https://www.gutenberg.org/cache/epub/1200/pg1200.txt"

# With custom epochs and output dir
garpug tune --input book.txt --epochs 100 --output my-tuned-model
```

GPT-2 weights (~600 MB) are downloaded once and cached in `~/.cache/huggingface/`.

### Generate text

```bash
# From a char-level model
garpug prompt --model output --temperature 1.2 "Gargantua did"

# From a fine-tuned GPT-2 model
garpug prompt --model tuned-output --temperature 0.8 "The king said"
```

The `prompt` command auto-detects the model type from `config.json`.

## Model Sizes

### Character-level (train)

| Size   | Parameters | Train Time (M1 Pro) |
|--------|-----------|---------------------|
| small  | ~1M       | ~2-5 min            |
| medium | ~10M      | ~10-20 min          |
| large  | ~85M      | ~1-2 hrs            |
| xl     | ~170M     | ~4-8 hrs            |

You can override individual parameters:

```bash
garpug train --input book.txt --size medium --dim 512 --layers 10
```

### GPT-2 (tune)

| Model    | Parameters |
|----------|-----------|
| GPT-2 small | 124M  |

GPT-2 small is the only size that fits in 16 GB RAM for fine-tuning.

## Output

### Character-level model (`train`)
Saves to `./output/` (or `--output <dir>`):
- `model.safetensors` — trained weights
- `vocab.json` — character vocabulary
- `config.json` — model architecture (includes `"model_type": "char"`)

### Fine-tuned GPT-2 (`tune`)
Saves to `./tuned-output/` (or `--output <dir>`):
- `model.safetensors` — fine-tuned weights (~548 MB)
- `tokenizer.json` — BPE tokenizer
- `config.json` — model architecture (includes `"model_type": "gpt2"`)
