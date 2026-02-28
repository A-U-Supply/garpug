# garpug

Train character-level GPT models on individual books. Generate weird text for art projects.

## Build

```bash
# With Metal GPU acceleration (macOS, recommended)
cargo build --release --features metal

# CPU only
cargo build --release
```

## Usage

### Train a model

```bash
# From a Project Gutenberg URL
garpug train --input "https://www.gutenberg.org/cache/epub/1200/pg1200.txt" --size medium

# From a local file
garpug train --input book.txt --size large --epochs 100
```

### Generate text

```bash
garpug prompt --model output --temperature 1.2 "Gargantua did"
```

## Model Sizes

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

## Output

Training saves to `./output/` (or `--output <dir>`):
- `model.safetensors` — trained weights
- `vocab.json` — character vocabulary
- `config.json` — model architecture
