# Garpug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Rust CLI that trains character-level GPT models on individual books and generates text from them.

**Architecture:** Two-subcommand CLI (`train` / `prompt`) using candle for GPU-accelerated transformer training on Metal. Character-level tokenization keeps the pipeline simple. Model weights saved as safetensors.

**Tech Stack:** Rust, candle-core + candle-nn (with Metal feature), clap 4 (derive), serde + serde_json, rand, reqwest (for fetching URLs)

---

### Task 1: Scaffold Project

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs`
- Create: `.gitignore`

**Step 1: Initialize cargo project**

Run: `cargo init --name garpug /Users/jake/au-supply/garpug`
Expected: Creates Cargo.toml and src/main.rs

**Step 2: Set up Cargo.toml with all dependencies**

Replace `Cargo.toml` with:

```toml
[package]
name = "garpug"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"
anyhow = "1"
reqwest = { version = "0.12", features = ["blocking"] }
indicatif = "0.17"

[features]
default = []
metal = ["candle-core/metal", "candle-nn/metal"]
```

**Step 3: Create .gitignore**

```
/target
/output
*.safetensors
.env
```

**Step 4: Verify it compiles**

Run: `cd /Users/jake/au-supply/garpug && cargo check`
Expected: Compiles with no errors (warnings OK)

**Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/main.rs .gitignore
git commit -m "feat: scaffold garpug project with dependencies"
```

---

### Task 2: CLI Parsing with Clap

**Files:**
- Create: `src/cli.rs`
- Modify: `src/main.rs`

**Step 1: Write a test for CLI parsing**

Create `src/cli.rs`:

```rust
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "garpug")]
#[command(about = "Train character-level GPT models on books and generate text")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train a model on a text file or URL
    Train(TrainArgs),
    /// Generate text from a trained model
    Prompt(PromptArgs),
}

#[derive(Parser)]
pub struct TrainArgs {
    /// Path to text file or Project Gutenberg URL
    #[arg(short, long)]
    pub input: String,

    /// Model size preset
    #[arg(short, long, value_enum, default_value_t = SizePreset::Medium)]
    pub size: SizePreset,

    /// Override embedding dimension
    #[arg(long)]
    pub dim: Option<usize>,

    /// Override number of transformer layers
    #[arg(long)]
    pub layers: Option<usize>,

    /// Override number of attention heads
    #[arg(long)]
    pub heads: Option<usize>,

    /// Override context window size (characters)
    #[arg(long)]
    pub context: Option<usize>,

    /// Number of training epochs
    #[arg(long, default_value_t = 50)]
    pub epochs: usize,

    /// Output directory for model artifacts
    #[arg(short, long, default_value = "output")]
    pub output: PathBuf,
}

#[derive(Parser)]
pub struct PromptArgs {
    /// Path to output directory containing model artifacts
    #[arg(short, long)]
    pub model: PathBuf,

    /// Seed text for generation
    pub seed: String,

    /// Number of characters to generate
    #[arg(short, long, default_value_t = 500)]
    pub length: usize,

    /// Sampling temperature (higher = weirder)
    #[arg(short, long, default_value_t = 1.0)]
    pub temperature: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SizePreset {
    Small,
    Medium,
    Large,
    #[value(name = "xl")]
    Xl,
}

/// Model hyperparameters resolved from preset + overrides
pub struct ModelConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub dropout: f32,
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn from_preset(preset: SizePreset) -> Self {
        match preset {
            SizePreset::Small => Self {
                n_embd: 128, n_layer: 4, n_head: 4, block_size: 256,
                dropout: 0.1, vocab_size: 0,
            },
            SizePreset::Medium => Self {
                n_embd: 384, n_layer: 8, n_head: 6, block_size: 512,
                dropout: 0.1, vocab_size: 0,
            },
            SizePreset::Large => Self {
                n_embd: 768, n_layer: 12, n_head: 12, block_size: 1024,
                dropout: 0.1, vocab_size: 0,
            },
            SizePreset::Xl => Self {
                n_embd: 1024, n_layer: 16, n_head: 16, block_size: 1024,
                dropout: 0.1, vocab_size: 0,
            },
        }
    }

    pub fn apply_overrides(&mut self, args: &TrainArgs) {
        if let Some(d) = args.dim { self.n_embd = d; }
        if let Some(l) = args.layers { self.n_layer = l; }
        if let Some(h) = args.heads { self.n_head = h; }
        if let Some(c) = args.context { self.block_size = c; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_train_defaults() {
        let cli = Cli::parse_from(["garpug", "train", "--input", "book.txt"]);
        match cli.command {
            Commands::Train(args) => {
                assert_eq!(args.input, "book.txt");
                assert_eq!(args.size, SizePreset::Medium);
                assert_eq!(args.epochs, 50);
                assert_eq!(args.output.to_str().unwrap(), "output");
                assert!(args.dim.is_none());
            }
            _ => panic!("expected Train"),
        }
    }

    #[test]
    fn test_train_overrides() {
        let cli = Cli::parse_from([
            "garpug", "train", "--input", "book.txt",
            "--size", "large", "--dim", "512", "--layers", "6",
        ]);
        match cli.command {
            Commands::Train(args) => {
                assert_eq!(args.size, SizePreset::Large);
                assert_eq!(args.dim, Some(512));
                assert_eq!(args.layers, Some(6));
            }
            _ => panic!("expected Train"),
        }
    }

    #[test]
    fn test_prompt_defaults() {
        let cli = Cli::parse_from([
            "garpug", "prompt", "--model", "output", "hello world",
        ]);
        match cli.command {
            Commands::Prompt(args) => {
                assert_eq!(args.seed, "hello world");
                assert_eq!(args.length, 500);
                assert_eq!(args.temperature, 1.0);
            }
            _ => panic!("expected Prompt"),
        }
    }

    #[test]
    fn test_model_config_preset_with_overrides() {
        let args = TrainArgs {
            input: "x".into(),
            size: SizePreset::Small,
            dim: Some(256),
            layers: None,
            heads: None,
            context: Some(128),
            epochs: 10,
            output: "out".into(),
        };
        let mut config = ModelConfig::from_preset(args.size);
        config.apply_overrides(&args);
        assert_eq!(config.n_embd, 256);
        assert_eq!(config.n_layer, 4); // unchanged from Small
        assert_eq!(config.block_size, 128);
    }
}
```

**Step 2: Wire up main.rs**

Replace `src/main.rs` with:

```rust
mod cli;

use clap::Parser;
use cli::{Cli, Commands};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => {
            println!("Training not yet implemented");
            println!("  input: {}", args.input);
            println!("  size: {:?}", args.size);
        }
        Commands::Prompt(args) => {
            println!("Prompting not yet implemented");
            println!("  model: {}", args.model.display());
            println!("  seed: {}", args.seed);
        }
    }
    Ok(())
}
```

**Step 3: Run the tests**

Run: `cargo test`
Expected: All 4 tests pass

**Step 4: Commit**

```bash
git add src/cli.rs src/main.rs
git commit -m "feat: add CLI parsing with clap (train/prompt subcommands, presets)"
```

---

### Task 3: Data Pipeline (Fetch, Clean, Tokenize)

**Files:**
- Create: `src/data.rs`
- Modify: `src/main.rs`

**Step 1: Write tests for data pipeline**

Create `src/data.rs` with the full module including tests:

```rust
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Vocab {
    pub char_to_idx: HashMap<char, u32>,
    pub idx_to_char: HashMap<u32, char>,
}

impl Vocab {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        chars.sort();
        let char_to_idx: HashMap<char, u32> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        let idx_to_char: HashMap<u32, char> = char_to_idx.iter()
            .map(|(&c, &i)| (i, c))
            .collect();
        Self { char_to_idx, idx_to_char }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.idx_to_char.get(&id))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.char_to_idx.len()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let vocab: Self = serde_json::from_str(&json)?;
        Ok(vocab)
    }
}

/// Fetch text from a file path or URL
pub fn fetch_text(input: &str) -> Result<String> {
    if input.starts_with("http://") || input.starts_with("https://") {
        let resp = reqwest::blocking::get(input)
            .context("Failed to fetch URL")?;
        let text = resp.text().context("Failed to read response body")?;
        Ok(text)
    } else {
        std::fs::read_to_string(input)
            .context(format!("Failed to read file: {input}"))
    }
}

/// Strip Project Gutenberg header and footer boilerplate
pub fn strip_gutenberg(text: &str) -> &str {
    let start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ];
    let end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ];

    let start = start_markers.iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .map(|pos| {
            // Skip past the marker line
            text[pos..].find('\n').map(|nl| pos + nl + 1).unwrap_or(pos)
        })
        .unwrap_or(0);

    let end = end_markers.iter()
        .filter_map(|marker| text[start..].find(marker).map(|p| p + start))
        .min()
        .unwrap_or(text.len());

    text[start..end].trim()
}

pub struct Dataset {
    pub train: Vec<u32>,
    pub val: Vec<u32>,
}

impl Dataset {
    pub fn from_tokens(tokens: Vec<u32>, val_fraction: f32) -> Self {
        let split = ((tokens.len() as f32) * (1.0 - val_fraction)) as usize;
        let train = tokens[..split].to_vec();
        let val = tokens[split..].to_vec();
        Self { train, val }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_roundtrip() {
        let vocab = Vocab::from_text("hello world");
        let encoded = vocab.encode("hello");
        let decoded = vocab.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_vocab_size() {
        let vocab = Vocab::from_text("aabbc");
        assert_eq!(vocab.size(), 3); // a, b, c
    }

    #[test]
    fn test_strip_gutenberg() {
        let text = "Some header stuff\n*** START OF THIS PROJECT GUTENBERG EBOOK ***\nActual book content here.\n*** END OF THIS PROJECT GUTENBERG EBOOK ***\nSome footer.";
        let stripped = strip_gutenberg(text);
        assert_eq!(stripped, "Actual book content here.");
    }

    #[test]
    fn test_strip_gutenberg_no_markers() {
        let text = "Just plain text with no markers.";
        let stripped = strip_gutenberg(text);
        assert_eq!(stripped, "Just plain text with no markers.");
    }

    #[test]
    fn test_dataset_split() {
        let tokens: Vec<u32> = (0..100).collect();
        let ds = Dataset::from_tokens(tokens, 0.1);
        assert_eq!(ds.train.len(), 90);
        assert_eq!(ds.val.len(), 10);
    }

    #[test]
    fn test_fetch_text_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("garpug_test.txt");
        std::fs::write(&path, "test content").unwrap();
        let result = fetch_text(path.to_str().unwrap()).unwrap();
        assert_eq!(result, "test content");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_vocab_save_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("garpug_vocab_test.json");
        let vocab = Vocab::from_text("abc");
        vocab.save(&path).unwrap();
        let loaded = Vocab::load(&path).unwrap();
        assert_eq!(loaded.size(), 3);
        assert_eq!(loaded.encode("abc"), vocab.encode("abc"));
        std::fs::remove_file(&path).ok();
    }
}
```

**Step 2: Add module to main.rs**

Add `mod data;` to the top of `src/main.rs`.

**Step 3: Run tests**

Run: `cargo test`
Expected: All tests pass (CLI tests + data tests)

**Step 4: Commit**

```bash
git add src/data.rs src/main.rs
git commit -m "feat: add data pipeline (fetch, gutenberg strip, vocab, dataset)"
```

---

### Task 4: GPT Model

**Files:**
- Create: `src/model.rs`
- Modify: `src/main.rs`

**Step 1: Implement the GPT model**

Create `src/model.rs`:

```rust
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, loss, ops, sequential,
    Activation, Embedding, LayerNorm, LayerNormConfig, Linear, Module,
    Sequential, VarBuilder, VarMap,
};

const FF_SCALE: usize = 4;
const LN_EPS: f64 = 1e-5;

// ─── Attention Head ───────────────────────────────────────────────────────────

struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    tril: Tensor,
    neg_inf: Tensor,
    dropout: f32,
}

impl Head {
    fn new(
        n_embd: usize,
        head_size: usize,
        block_size: usize,
        dropout: f32,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            key: linear_no_bias(n_embd, head_size, vb.pp("key"))?,
            query: linear_no_bias(n_embd, head_size, vb.pp("query"))?,
            value: linear_no_bias(n_embd, head_size, vb.pp("value"))?,
            tril: Tensor::tril2(block_size, DType::U32, device)?,
            neg_inf: Tensor::try_from(f32::NEG_INFINITY)?.to_device(device)?,
            dropout,
        })
    }
}

impl Module for Head {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let k = self.key.forward(xs)?;
        let q = self.query.forward(xs)?;
        let (_, t, c) = xs.shape().dims3()?;
        let scale = (c as f64).powf(-0.5);
        let mut w = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let mask = self.tril.i((..t, ..t))?.broadcast_as(w.shape())?;
        w = mask.where_cond(&w, &self.neg_inf.broadcast_as(w.shape())?)?;
        w = ops::softmax(&w, D::Minus1)?;
        w = ops::dropout(&w, self.dropout)?;
        w.matmul(&self.value.forward(xs)?)
    }
}

// ─── Multi-Head Attention ─────────────────────────────────────────────────────

struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout: f32,
}

impl MultiHeadAttention {
    fn new(
        n_embd: usize,
        n_head: usize,
        block_size: usize,
        dropout: f32,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let head_size = n_embd / n_head;
        let heads = (0..n_head)
            .map(|i| Head::new(n_embd, head_size, block_size, dropout, vb.pp(format!("head_{i}")), device))
            .collect::<Result<Vec<_>>>()?;
        let proj = linear(n_embd, n_embd, vb.pp("proj"))?;
        Ok(Self { heads, proj, dropout })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let outs: Vec<Tensor> = self.heads.iter()
            .map(|h| h.forward(xs))
            .collect::<Result<Vec<_>>>()?;
        let cat = Tensor::cat(&outs, D::Minus1)?.contiguous()?;
        let proj = self.proj.forward(&cat)?;
        ops::dropout(&proj, self.dropout)
    }
}

// ─── Feed-Forward ─────────────────────────────────────────────────────────────

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: f32,
}

impl FeedForward {
    fn new(n_embd: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(n_embd, FF_SCALE * n_embd, vb.pp("linear1"))?,
            linear2: linear(FF_SCALE * n_embd, n_embd, vb.pp("linear2"))?,
            dropout,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(xs)?;
        let x = x.gelu_erf()?;
        let x = self.linear2.forward(&x)?;
        ops::dropout(&x, self.dropout)
    }
}

// ─── Transformer Block ───────────────────────────────────────────────────────

struct Block {
    attn: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl Block {
    fn new(
        n_embd: usize,
        n_head: usize,
        block_size: usize,
        dropout: f32,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let ln_cfg = LayerNormConfig { eps: LN_EPS, ..Default::default() };
        Ok(Self {
            attn: MultiHeadAttention::new(n_embd, n_head, block_size, dropout, vb.pp("attn"), device)?,
            ffwd: FeedForward::new(n_embd, dropout, vb.pp("ffwd"))?,
            ln1: layer_norm(n_embd, ln_cfg, vb.pp("ln1"))?,
            ln2: layer_norm(n_embd, ln_cfg, vb.pp("ln2"))?,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.ln1.forward(xs)?)?)?;
        let xs = (&xs + self.ffwd.forward(&self.ln2.forward(&xs)?)?)?;
        Ok(xs)
    }
}

// ─── GPT Model ────────────────────────────────────────────────────────────────

pub struct GPT {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    pub var_map: VarMap,
    pub block_size: usize,
}

impl GPT {
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_layer: usize,
        n_head: usize,
        block_size: usize,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let tok_emb = embedding(vocab_size, n_embd, vb.pp("tok_emb"))?;
        let pos_emb = embedding(block_size, n_embd, vb.pp("pos_emb"))?;

        let blocks = (0..n_layer)
            .map(|i| Block::new(n_embd, n_head, block_size, dropout, vb.pp(format!("block_{i}")), device))
            .collect::<Result<Vec<_>>>()?;

        let ln_cfg = LayerNormConfig { eps: LN_EPS, ..Default::default() };
        let ln_f = layer_norm(n_embd, ln_cfg, vb.pp("ln_f"))?;
        let lm_head = linear(n_embd, vocab_size, vb.pp("lm_head"))?;

        Ok(Self { tok_emb, pos_emb, blocks, ln_f, lm_head, var_map, block_size })
    }

    /// Forward pass returning logits [batch, seq_len, vocab_size]
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_, t) = idx.shape().dims2()?;
        let tok = self.tok_emb.forward(idx)?;
        let pos_ids = Tensor::arange(0u32, t as u32, idx.device())?;
        let pos = self.pos_emb.forward(&pos_ids)?;
        let mut x = tok.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Compute cross-entropy loss
    pub fn loss(&self, idx: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward(idx)?;
        let (b, t, c) = logits.shape().dims3()?;
        loss::cross_entropy(
            &logits.reshape((b * t, c))?,
            &targets.reshape((b * t,))?,
        )
    }

    /// Generate tokens autoregressively
    pub fn generate(
        &self,
        seed_ids: &[u32],
        max_new_tokens: usize,
        temperature: f64,
        device: &Device,
    ) -> Result<Vec<u32>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        let mut ids = seed_ids.to_vec();

        for _ in 0..max_new_tokens {
            let start = if ids.len() > self.block_size {
                ids.len() - self.block_size
            } else {
                0
            };
            let ctx: Vec<u32> = ids[start..].to_vec();
            let ctx_len = ctx.len();
            let input = Tensor::from_vec(ctx, (1, ctx_len), device)?;
            let logits = self.forward(&input)?;
            let logits = logits.i((0, ctx_len - 1, ..))?;
            let logits = (logits / temperature)?;
            let probs = ops::softmax(&logits, 0)?;
            let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

            // Multinomial sample
            let r: f32 = rng.gen();
            let mut cumsum = 0.0f32;
            let mut next = (probs_vec.len() - 1) as u32;
            for (i, &p) in probs_vec.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    next = i as u32;
                    break;
                }
            }
            ids.push(next);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_forward_shape() {
        let device = Device::Cpu;
        let model = GPT::new(26, 32, 2, 2, 16, 0.0, &device).unwrap();
        let input = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let logits = model.forward(&input).unwrap();
        let (b, t, c) = logits.shape().dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(t, 8);
        assert_eq!(c, 26);
    }

    #[test]
    fn test_gpt_loss_computes() {
        let device = Device::Cpu;
        let model = GPT::new(26, 32, 2, 2, 16, 0.0, &device).unwrap();
        let input = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let targets = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let loss = model.loss(&input, &targets).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val > 0.0); // cross-entropy on random weights should be positive
    }

    #[test]
    fn test_gpt_generate() {
        let device = Device::Cpu;
        let model = GPT::new(26, 32, 2, 2, 16, 0.0, &device).unwrap();
        let result = model.generate(&[0, 1, 2], 5, 1.0, &device).unwrap();
        assert_eq!(result.len(), 8); // 3 seed + 5 generated
    }

    #[test]
    fn test_save_load_roundtrip() {
        let device = Device::Cpu;
        let model = GPT::new(10, 16, 1, 2, 8, 0.0, &device).unwrap();
        let dir = std::env::temp_dir().join("garpug_model_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        model.var_map.save(&path).unwrap();

        // Load into fresh model
        let model2 = GPT::new(10, 16, 1, 2, 8, 0.0, &device).unwrap();
        model2.var_map.load(&path).unwrap();

        // Same output for same input
        let input = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let out1 = model.forward(&input).unwrap();
        let out2 = model2.forward(&input).unwrap();
        let diff: f32 = (out1 - out2).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(diff < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }
}
```

**Step 2: Add module to main.rs**

Add `mod model;` to the top of `src/main.rs`.

**Step 3: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/model.rs src/main.rs
git commit -m "feat: add GPT model (transformer, attention, generation)"
```

---

### Task 5: Training Loop

**Files:**
- Create: `src/train.rs`
- Modify: `src/main.rs`

**Step 1: Implement the training loop**

Create `src/train.rs`:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use std::path::Path;

use crate::cli::ModelConfig;
use crate::data::{Dataset, Vocab};
use crate::model::GPT;

/// Create a random batch of (input, target) pairs from the dataset
fn random_batch(
    data: &[u32],
    block_size: usize,
    batch_size: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::with_capacity(batch_size * block_size);
    let mut targets = Vec::with_capacity(batch_size * block_size);

    for _ in 0..batch_size {
        let start = rng.gen_range(0..data.len() - block_size);
        inputs.extend_from_slice(&data[start..start + block_size]);
        targets.extend_from_slice(&data[start + 1..start + block_size + 1]);
    }

    let x = Tensor::from_vec(inputs, (batch_size, block_size), device)?;
    let y = Tensor::from_vec(targets, (batch_size, block_size), device)?;
    Ok((x, y))
}

pub fn run_training(
    config: &ModelConfig,
    dataset: &Dataset,
    epochs: usize,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let batch_size = 64;
    let eval_interval = 100;
    let eval_batches = 8;

    println!("Model: {} layers, {} dim, {} heads, {} context",
        config.n_layer, config.n_embd, config.n_head, config.block_size);
    println!("Vocab size: {}", config.vocab_size);
    println!("Training on {} chars, validating on {} chars",
        dataset.train.len(), dataset.val.len());

    let model = GPT::new(
        config.vocab_size, config.n_embd, config.n_layer,
        config.n_head, config.block_size, config.dropout, device,
    )?;

    let params = model.var_map.all_vars();
    let total_params: usize = params.iter().map(|v| v.as_tensor().elem_count()).sum();
    println!("Parameters: {}", format_params(total_params));

    let adamw_params = ParamsAdamW {
        lr: 3e-4,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.1,
    };
    let mut optimizer = AdamW::new(params, adamw_params)?;

    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} | loss: {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut best_val_loss = f32::INFINITY;

    for epoch in 0..epochs {
        let (x, y) = random_batch(&dataset.train, config.block_size, batch_size, device)?;
        let loss = model.loss(&x, &y)?;
        optimizer.backward_step(&loss)?;

        let train_loss: f32 = loss.to_scalar()?;

        if epoch % eval_interval == 0 || epoch == epochs - 1 {
            let val_loss = estimate_loss(&model, &dataset.val, config.block_size, eval_batches, device)?;
            pb.set_message(format!("train={train_loss:.4} val={val_loss:.4}"));

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                std::fs::create_dir_all(output_dir)?;
                model.var_map.save(output_dir.join("model.safetensors"))?;
            }
        } else {
            pb.set_message(format!("{train_loss:.4}"));
        }
        pb.inc(1);
    }
    pb.finish_with_message(format!("done | best val loss: {best_val_loss:.4}"));

    // Save final model too
    std::fs::create_dir_all(output_dir)?;
    model.var_map.save(output_dir.join("model.safetensors"))?;

    // Save config
    let config_json = serde_json::json!({
        "vocab_size": config.vocab_size,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "block_size": config.block_size,
    });
    std::fs::write(
        output_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json)?,
    )?;

    println!("Model saved to {}", output_dir.display());
    Ok(())
}

fn estimate_loss(
    model: &GPT,
    data: &[u32],
    block_size: usize,
    n_batches: usize,
    device: &Device,
) -> candle_core::Result<f32> {
    let mut total = 0.0f32;
    for _ in 0..n_batches {
        let (x, y) = random_batch(data, block_size, 32, device)?;
        let loss = model.loss(&x, &y)?;
        total += loss.to_scalar::<f32>()?;
    }
    Ok(total / n_batches as f32)
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}
```

**Step 2: Add module to main.rs**

Add `mod train;` to main.rs.

**Step 3: Run `cargo check`**

Run: `cargo check`
Expected: Compiles

**Step 4: Commit**

```bash
git add src/train.rs src/main.rs
git commit -m "feat: add training loop with AdamW, progress bar, checkpointing"
```

---

### Task 6: Wire Everything Together in main.rs

**Files:**
- Modify: `src/main.rs`

**Step 1: Implement the full main.rs**

Replace `src/main.rs` with:

```rust
mod cli;
mod data;
mod model;
mod train;

use anyhow::Result;
use candle_core::{Device, DType};
use clap::Parser;

use cli::{Cli, Commands, ModelConfig};
use data::{fetch_text, strip_gutenberg, Dataset, Vocab};
use model::GPT;

fn get_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                println!("Using Metal GPU");
                return d;
            }
            Err(e) => {
                eprintln!("Metal not available ({e}), falling back to CPU");
            }
        }
    }
    println!("Using CPU");
    Device::Cpu
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => {
            let device = get_device();

            // Fetch and prepare text
            println!("Loading text from: {}", args.input);
            let raw = fetch_text(&args.input)?;
            let text = strip_gutenberg(&raw);
            println!("Text length: {} characters", text.len());

            // Build vocabulary
            let vocab = Vocab::from_text(text);
            println!("Vocabulary: {} unique characters", vocab.size());

            // Tokenize and split
            let tokens = vocab.encode(text);
            let dataset = Dataset::from_tokens(tokens, 0.1);

            // Resolve model config
            let mut config = ModelConfig::from_preset(args.size);
            config.apply_overrides(&args);
            config.vocab_size = vocab.size();

            // Save vocab
            std::fs::create_dir_all(&args.output)?;
            vocab.save(&args.output.join("vocab.json"))?;

            // Train
            train::run_training(&config, &dataset, args.epochs, &args.output, &device)?;
        }

        Commands::Prompt(args) => {
            let device = get_device();

            // Load config
            let config_str = std::fs::read_to_string(args.model.join("config.json"))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;

            let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
            let n_embd = config["n_embd"].as_u64().unwrap() as usize;
            let n_layer = config["n_layer"].as_u64().unwrap() as usize;
            let n_head = config["n_head"].as_u64().unwrap() as usize;
            let block_size = config["block_size"].as_u64().unwrap() as usize;

            // Load vocab
            let vocab = Vocab::load(&args.model.join("vocab.json"))?;

            // Load model
            let model = GPT::new(vocab_size, n_embd, n_layer, n_head, block_size, 0.0, &device)?;
            model.var_map.load(args.model.join("model.safetensors"))?;

            // Encode seed and generate
            let seed_ids = vocab.encode(&args.seed);
            if seed_ids.is_empty() {
                anyhow::bail!("Seed text produced no tokens — check that characters exist in the vocabulary");
            }

            let generated = model.generate(&seed_ids, args.length, args.temperature, &device)?;
            let text = vocab.decode(&generated);
            println!("{text}");
        }
    }
    Ok(())
}
```

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: Compiles

**Step 3: Run all tests**

Run: `cargo test`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire up train and prompt subcommands in main"
```

---

### Task 7: Build, Fetch Gutenberg Text, and End-to-End Test

**Step 1: Build release binary with Metal**

Run: `cargo build --release --features metal`
Expected: Compiles successfully

**Step 2: Fetch Gargantua and Pantagruel and do a quick smoke test**

Run a quick training with `--size small --epochs 5` to verify the full pipeline works end-to-end:

```bash
./target/release/garpug train \
  --input "https://www.gutenberg.org/cache/epub/1200/pg1200.txt" \
  --size small --epochs 5 --output test_output
```

Expected: Downloads text, trains for 5 epochs, saves model to `test_output/`

**Step 3: Test prompting**

```bash
./target/release/garpug prompt \
  --model test_output --length 100 --temperature 1.2 \
  "Gargantua did"
```

Expected: Prints ~100 characters of generated text (will be nonsensical after 5 epochs — that's fine)

**Step 4: Clean up test output and commit**

```bash
rm -rf test_output
git add -A
git commit -m "chore: build verified, end-to-end pipeline works"
```

---

### Task 8: README

**Files:**
- Create: `README.md`

**Step 1: Write README**

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with build and usage instructions"
```
