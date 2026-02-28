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
    /// Train a character-level model on a text file or URL
    Train(TrainArgs),
    /// Fine-tune pretrained GPT-2 on a text file or URL
    Tune(TuneArgs),
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
pub struct TuneArgs {
    /// Path to text file or Project Gutenberg URL
    #[arg(short, long)]
    pub input: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 50)]
    pub epochs: usize,

    /// Output directory for fine-tuned model
    #[arg(short, long, default_value = "tuned-output")]
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
    fn test_tune_defaults() {
        let cli = Cli::parse_from(["garpug", "tune", "--input", "book.txt"]);
        match cli.command {
            Commands::Tune(args) => {
                assert_eq!(args.input, "book.txt");
                assert_eq!(args.epochs, 50);
                assert_eq!(args.output.to_str().unwrap(), "tuned-output");
            }
            _ => panic!("expected Tune"),
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
        assert_eq!(config.n_layer, 4);
        assert_eq!(config.block_size, 128);
    }
}
