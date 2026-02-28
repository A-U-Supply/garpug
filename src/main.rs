mod cli;
mod data;
mod model;
mod train;

use anyhow::{Context, Result};
use candle_core::Device;
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

            // Load vocab
            let vocab = Vocab::load(&args.model.join("vocab.json"))?;

            // Load model
            let mut model = GPT::new(vocab_size, n_embd, n_layer, n_head, block_size, 0.0, &device)?;
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
