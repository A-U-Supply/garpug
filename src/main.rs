mod cli;
mod data;
mod gpt2;
mod model;
mod train;
mod tune;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;

use cli::{Cli, Commands, ModelConfig};
use data::{fetch_text, strip_gutenberg, Dataset, Vocab};
use gpt2::GPT2;
use model::GPT;

fn get_metal_device() -> Option<Device> {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => return Some(d),
            Err(_) => {}
        }
    }
    None
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => {
            // CPU is faster than Metal for training (candle's Metal backward pass has high overhead)
            let device = Device::Cpu;
            println!("Using CPU (faster than Metal for training)");

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

        Commands::Tune(args) => {
            tune::run_tuning(&args.input, args.epochs, &args.output)?;
        }

        Commands::Prompt(args) => {
            // Metal is fine for inference (forward pass only)
            let device = get_metal_device().unwrap_or(Device::Cpu);
            if matches!(device, Device::Cpu) {
                println!("Using CPU");
            } else {
                println!("Using Metal GPU");
            }

            // Load config and auto-detect model type
            let config_str = std::fs::read_to_string(args.model.join("config.json"))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;

            let model_type = config["model_type"]
                .as_str()
                .unwrap_or("char");

            let vocab_size = config["vocab_size"].as_u64()
                .context("config.json missing or invalid 'vocab_size'")? as usize;
            let n_embd = config["n_embd"].as_u64()
                .context("config.json missing or invalid 'n_embd'")? as usize;
            let n_layer = config["n_layer"].as_u64()
                .context("config.json missing or invalid 'n_layer'")? as usize;
            let n_head = config["n_head"].as_u64()
                .context("config.json missing or invalid 'n_head'")? as usize;
            let block_size = config["block_size"].as_u64()
                .context("config.json missing or invalid 'block_size'")? as usize;

            match model_type {
                "gpt2" => {
                    println!("Loading GPT-2 model...");

                    // Load BPE tokenizer
                    let tokenizer = tokenizers::Tokenizer::from_file(
                        args.model.join("tokenizer.json"),
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

                    // Load model
                    let mut model = GPT2::new(vocab_size, n_embd, n_layer, n_head, block_size, &device)?;
                    model.var_map.load(args.model.join("model.safetensors"))?;

                    // Encode seed with BPE
                    let encoding = tokenizer
                        .encode(args.seed.as_str(), false)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
                    let seed_ids: Vec<u32> = encoding.get_ids().to_vec();
                    if seed_ids.is_empty() {
                        anyhow::bail!("Seed text produced no tokens");
                    }

                    let generated = model.generate(&seed_ids, args.length, args.temperature, &device)?;

                    // Decode BPE tokens
                    let text = tokenizer
                        .decode(&generated, true)
                        .map_err(|e| anyhow::anyhow!("Decoding failed: {e}"))?;
                    println!("{text}");
                }
                _ => {
                    // char-level model (original behavior)
                    let vocab = Vocab::load(&args.model.join("vocab.json"))?;

                    let mut model = GPT::new(vocab_size, n_embd, n_layer, n_head, block_size, 0.0, &device)?;
                    model.var_map.load(args.model.join("model.safetensors"))?;

                    let seed_ids = vocab.encode(&args.seed);
                    if seed_ids.is_empty() {
                        anyhow::bail!("Seed text produced no tokens — check that characters exist in the vocabulary");
                    }

                    let generated = model.generate(&seed_ids, args.length, args.temperature, &device)?;
                    let text = vocab.decode(&generated);
                    println!("{text}");
                }
            }
        }
    }
    Ok(())
}
