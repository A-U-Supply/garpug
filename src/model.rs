use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, loss, ops,
    Embedding, LayerNorm, LayerNormConfig, Linear, Module,
    VarBuilder, VarMap,
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
        assert!(val > 0.0);
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
        let mut model2 = GPT::new(10, 16, 1, 2, 8, 0.0, &device).unwrap();
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
