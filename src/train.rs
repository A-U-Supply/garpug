use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use std::path::Path;

use crate::cli::ModelConfig;
use crate::data::Dataset;
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
