mod cli;
mod data;
mod model;

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
