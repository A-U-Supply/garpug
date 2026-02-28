use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, loss, ops, Embedding, LayerNorm,
    LayerNormConfig, Linear, Module, VarBuilder, VarMap,
};
use std::collections::HashSet;
use std::path::Path;

const LN_EPS: f64 = 1e-5;

// GPT-2 small constants
pub const GPT2_VOCAB_SIZE: usize = 50257;
pub const GPT2_N_EMBD: usize = 768;
pub const GPT2_N_LAYER: usize = 12;
pub const GPT2_N_HEAD: usize = 12;
pub const GPT2_BLOCK_SIZE: usize = 1024;

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct GPT2MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl GPT2MLP {
    fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_fc: linear(n_embd, 4 * n_embd, vb.pp("c_fc"))?,
            c_proj: linear(4 * n_embd, n_embd, vb.pp("c_proj"))?,
        })
    }
}

impl Module for GPT2MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(xs)?.gelu_erf()?)
    }
}

// ─── Fused QKV Attention ─────────────────────────────────────────────────────

struct GPT2Attention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
}

impl GPT2Attention {
    fn new(n_embd: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_attn: linear(n_embd, 3 * n_embd, vb.pp("c_attn"))?,
            c_proj: linear(n_embd, n_embd, vb.pp("c_proj"))?,
            n_head,
            n_embd,
            head_dim: n_embd / n_head,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.shape().dims3()?;
        let qkv = self.c_attn.forward(xs)?;

        let q = qkv.narrow(D::Minus1, 0, self.n_embd)?;
        let k = qkv.narrow(D::Minus1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(D::Minus1, 2 * self.n_embd, self.n_embd)?;

        // [b, t, n_embd] -> [b, n_head, t, head_dim]
        let q = q
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        let mask = Tensor::tril2(t, DType::U32, xs.device())?;
        let neg_inf = Tensor::try_from(f32::NEG_INFINITY)?.to_device(xs.device())?;
        let mask = mask.broadcast_as(attn.shape())?;
        let attn = mask.where_cond(&attn, &neg_inf.broadcast_as(attn.shape())?)?;
        let attn = ops::softmax(&attn, D::Minus1)?;

        // [b, n_head, t, head_dim] -> [b, t, n_embd]
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_embd))?;
        self.c_proj.forward(&out)
    }
}

// ─── Transformer Block ──────────────────────────────────────────────────────

struct GPT2Block {
    ln_1: LayerNorm,
    attn: GPT2Attention,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}

impl GPT2Block {
    fn new(n_embd: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: LN_EPS,
            ..Default::default()
        };
        Ok(Self {
            ln_1: layer_norm(n_embd, ln_cfg, vb.pp("ln_1"))?,
            attn: GPT2Attention::new(n_embd, n_head, vb.pp("attn"))?,
            ln_2: layer_norm(n_embd, ln_cfg, vb.pp("ln_2"))?,
            mlp: GPT2MLP::new(n_embd, vb.pp("mlp"))?,
        })
    }
}

impl Module for GPT2Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.ln_1.forward(xs)?)?)?;
        let xs = (&xs + self.mlp.forward(&self.ln_2.forward(&xs)?)?)?;
        Ok(xs)
    }
}

// ─── GPT-2 Model ────────────────────────────────────────────────────────────

pub struct GPT2 {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<GPT2Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    pub var_map: VarMap,
    n_layer: usize,
    pub block_size: usize,
}

impl GPT2 {
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_layer: usize,
        n_head: usize,
        block_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let wte = embedding(vocab_size, n_embd, vb.pp("transformer.wte"))?;
        let wpe = embedding(block_size, n_embd, vb.pp("transformer.wpe"))?;

        let blocks = (0..n_layer)
            .map(|i| GPT2Block::new(n_embd, n_head, vb.pp(format!("transformer.h.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let ln_cfg = LayerNormConfig {
            eps: LN_EPS,
            ..Default::default()
        };
        let ln_f = layer_norm(n_embd, ln_cfg, vb.pp("transformer.ln_f"))?;
        let lm_head = linear_no_bias(n_embd, vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            var_map,
            n_layer,
            block_size,
        })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_, t) = idx.shape().dims2()?;
        let tok = self.wte.forward(idx)?;
        let pos_ids = Tensor::arange(0u32, t as u32, idx.device())?;
        let pos = self.wpe.forward(&pos_ids)?;
        let mut x = tok.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn loss(&self, idx: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward(idx)?;
        let (b, t, c) = logits.shape().dims3()?;
        loss::cross_entropy(
            &logits.reshape((b * t, c))?,
            &targets.reshape((b * t,))?,
        )
    }

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

    /// Load pretrained GPT-2 weights from safetensors, handling Conv1D transpose.
    /// Conv1D layers store weights as [in, out] but candle Linear expects [out, in].
    /// lm_head weight is copied from wte (weight tying).
    pub fn load_pretrained(&self, safetensors_path: &Path) -> anyhow::Result<()> {
        let tensors = candle_core::safetensors::load(safetensors_path, &Device::Cpu)?;

        // Conv1D weight tensor names that need transposing
        let mut conv1d_names = HashSet::new();
        for i in 0..self.n_layer {
            conv1d_names.insert(format!("transformer.h.{i}.attn.c_attn.weight"));
            conv1d_names.insert(format!("transformer.h.{i}.attn.c_proj.weight"));
            conv1d_names.insert(format!("transformer.h.{i}.mlp.c_fc.weight"));
            conv1d_names.insert(format!("transformer.h.{i}.mlp.c_proj.weight"));
        }

        let data = self.var_map.data().lock().unwrap();
        for (name, var) in data.iter() {
            if let Some(tensor) = tensors.get(name) {
                let tensor = if conv1d_names.contains(name) {
                    tensor.t()?
                } else {
                    tensor.clone()
                };
                let device = var.as_tensor().device().clone();
                var.set(&tensor.to_dtype(DType::F32)?.to_device(&device)?)?;
            } else if name == "lm_head.weight" {
                // Weight tying: copy wte embeddings into lm_head
                if let Some(wte) = tensors.get("transformer.wte.weight") {
                    let device = var.as_tensor().device().clone();
                    var.set(&wte.to_dtype(DType::F32)?.to_device(&device)?)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_forward_shape() {
        let device = Device::Cpu;
        // Use tiny dimensions for fast testing
        let model = GPT2::new(100, 32, 2, 2, 16, &device).unwrap();
        let input = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let logits = model.forward(&input).unwrap();
        let (b, t, c) = logits.shape().dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(t, 8);
        assert_eq!(c, 100);
    }

    #[test]
    fn test_gpt2_loss_computes() {
        let device = Device::Cpu;
        let model = GPT2::new(100, 32, 2, 2, 16, &device).unwrap();
        let input = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let targets = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let loss = model.loss(&input, &targets).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val > 0.0);
    }
}
