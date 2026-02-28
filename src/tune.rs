use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::gpt2::{
    GPT2, GPT2_BLOCK_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_VOCAB_SIZE,
};

const BATCH_SIZE: usize = 2;
const CONTEXT: usize = 256;
const HF_MODEL: &str = "openai-community/gpt2";

fn random_batch(
    data: &[u32],
    context: usize,
    batch_size: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::with_capacity(batch_size * context);
    let mut targets = Vec::with_capacity(batch_size * context);
    for _ in 0..batch_size {
        let start = rng.gen_range(0..data.len() - context);
        inputs.extend_from_slice(&data[start..start + context]);
        targets.extend_from_slice(&data[start + 1..start + context + 1]);
    }
    let x = Tensor::from_vec(inputs, (batch_size, context), device)?;
    let y = Tensor::from_vec(targets, (batch_size, context), device)?;
    Ok((x, y))
}

pub fn run_tuning(input: &str, epochs: usize, output_dir: &Path) -> Result<()> {
    let device = Device::Cpu;
    println!("Using CPU");

    // Download pretrained GPT-2
    println!("Downloading GPT-2 weights and tokenizer...");
    let api = Api::new()?;
    let repo = api.model(HF_MODEL.to_string());
    let weights_path = repo.get("model.safetensors")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    println!("Downloaded.");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Fetch and tokenize text
    println!("Loading text from: {input}");
    let raw = crate::data::fetch_text(input)?;
    let text = crate::data::strip_gutenberg(&raw);
    println!("Text length: {} characters", text.len());

    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    println!("BPE tokens: {}", tokens.len());

    // Split into train/val (90/10)
    let split = (tokens.len() as f32 * 0.9) as usize;
    let train_data = &tokens[..split];
    let val_data = &tokens[split..];

    if train_data.len() <= CONTEXT + 1 {
        anyhow::bail!(
            "Training data ({} tokens) is too short for context window ({CONTEXT}).",
            train_data.len()
        );
    }
    if val_data.len() <= CONTEXT + 1 {
        anyhow::bail!(
            "Validation data ({} tokens) is too short for context window ({CONTEXT}).",
            val_data.len()
        );
    }

    println!(
        "Training on {} tokens, validating on {} tokens",
        train_data.len(),
        val_data.len()
    );

    // Build model and load pretrained weights
    println!("Loading pretrained GPT-2 weights...");
    let model = GPT2::new(
        GPT2_VOCAB_SIZE,
        GPT2_N_EMBD,
        GPT2_N_LAYER,
        GPT2_N_HEAD,
        GPT2_BLOCK_SIZE,
        &device,
    )?;
    model.load_pretrained(&weights_path)?;

    let params = model.var_map.all_vars();
    let total_params: usize = params.iter().map(|v| v.as_tensor().elem_count()).sum();
    println!("Parameters: {}", format_params(total_params));

    // Fine-tune with lower learning rate than training from scratch
    let adamw_params = ParamsAdamW {
        lr: 3e-5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.1,
    };
    let mut optimizer = AdamW::new(params, adamw_params)?;

    let eval_interval = 100;

    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} | loss: {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut best_val_loss = f32::INFINITY;

    for epoch in 0..epochs {
        let (x, y) = random_batch(train_data, CONTEXT, BATCH_SIZE, &device)?;
        let loss = model.loss(&x, &y)?;
        optimizer.backward_step(&loss)?;

        let train_loss: f32 = loss.to_scalar()?;

        if epoch % eval_interval == 0 || epoch == epochs - 1 {
            let val_loss = estimate_loss(&model, val_data, &device)?;
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

    // Save final model
    std::fs::create_dir_all(output_dir)?;
    model.var_map.save(output_dir.join("model.safetensors"))?;

    // Save config
    let config_json = serde_json::json!({
        "model_type": "gpt2",
        "vocab_size": GPT2_VOCAB_SIZE,
        "n_embd": GPT2_N_EMBD,
        "n_layer": GPT2_N_LAYER,
        "n_head": GPT2_N_HEAD,
        "block_size": GPT2_BLOCK_SIZE,
    });
    std::fs::write(
        output_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json)?,
    )?;

    // Copy tokenizer to output dir
    std::fs::copy(&tokenizer_path, output_dir.join("tokenizer.json"))?;

    println!("Model saved to {}", output_dir.display());
    Ok(())
}

fn estimate_loss(model: &GPT2, data: &[u32], device: &Device) -> candle_core::Result<f32> {
    let eval_batches = 8;
    let mut total = 0.0f32;
    for _ in 0..eval_batches {
        let (x, y) = random_batch(data, CONTEXT, BATCH_SIZE, device)?;
        let loss = model.loss(&x, &y)?;
        total += loss.to_scalar::<f32>()?;
    }
    Ok(total / eval_batches as f32)
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
