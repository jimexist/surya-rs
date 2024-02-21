//! Swin Transformer
//!
//! The Swin Transformer was proposed in Swin Transformer: Hierarchical Vision Transformer using
//! Shifted Windows by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
//! Baining Guo.
//!
//! https://huggingface.co/docs/transformers/model_doc/swin

use crate::tensor_roll::Roll;
use candle_core::{DType, Device, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn::{
    conv2d, layer_norm, linear, linear_no_bias, Activation, Conv2d, Conv2dConfig, LayerNorm,
    Linear, VarBuilder,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct SwinConfig {
    pub image_size: (usize, usize),
    pub patch_size: usize,
    pub num_channels: usize,
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub window_size: usize,
    pub mlp_ratio: f64,
    pub qkv_bias: bool,
    pub hidden_act: Activation,
    pub use_absolute_embeddings: bool,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
}

impl SwinConfig {
    fn tiny_patch4_window7_224() -> Self {
        Self {
            image_size: (224, 224),
            patch_size: 4,
            num_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            hidden_act: Activation::Gelu,
            use_absolute_embeddings: false,
            initializer_range: 0.02,
            layer_norm_eps: 1e-05,
        }
    }
}

impl Default for SwinConfig {
    /// this defaults to the surya swin encoder config
    fn default() -> Self {
        Self {
            image_size: (196, 896),
            patch_size: 4,
            num_channels: 3,
            embed_dim: 128,
            depths: vec![2, 2, 14, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            hidden_act: Activation::Gelu,
            use_absolute_embeddings: true,
            initializer_range: 0.02,
            layer_norm_eps: 1e-05,
        }
    }
}

#[derive(Debug, Clone)]
struct SwinPatchEmbeddings {
    projection: Conv2d,
    patch_size: usize,
    num_channels: usize,
    num_patches: usize,
}

impl SwinPatchEmbeddings {
    fn new(config: &SwinConfig, vb: VarBuilder) -> Result<Self> {
        let num_channels = config.num_channels as usize;
        let patch_size = config.patch_size as usize;
        let hidden_size = config.embed_dim as usize;
        let projection = conv2d(
            num_channels,
            hidden_size,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("projection"),
        )?;
        let num_patches = config.image_size.0 * config.image_size.1 / patch_size / patch_size;
        Ok(Self {
            projection,
            patch_size,
            num_patches,
            num_channels,
        })
    }

    fn maybe_pad(&self, tensor: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        debug_assert_eq!(
            4,
            tensor.dims().len(),
            "Input tensor must have 4 dimensions"
        );
        let tensor = if width % self.patch_size != 0 {
            let pad = self.patch_size - (width % self.patch_size);
            tensor.pad_with_zeros(3, 0, pad)?
        } else {
            tensor.clone()
        };
        let tensor = if height % self.patch_size != 0 {
            let pad = self.patch_size - (height % self.patch_size);
            tensor.pad_with_zeros(2, 0, pad)?
        } else {
            tensor.clone()
        };
        Ok(tensor)
    }
}

impl Module for SwinPatchEmbeddings {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        if c != self.num_channels {
            candle_core::bail!("Input channels must be equal to num_channels");
        }
        let x = self.maybe_pad(x, h, w)?;
        let embedding = self.projection.forward(&x)?;
        Ok(embedding)
    }
}

#[derive(Debug, Clone)]
struct SwinEmbeddings {
    patch_embeddings: SwinPatchEmbeddings,
    position_embeddings: Option<Tensor>,
    norm: LayerNorm,
}

impl SwinEmbeddings {
    fn new(config: &SwinConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embeddings = SwinPatchEmbeddings::new(config, vb.pp("patch_embeddings"))?;
        let norm = layer_norm(config.embed_dim, config.layer_norm_eps, vb.pp("norm"))?;
        let position_embeddings = if config.use_absolute_embeddings {
            let position_embedding = vb.get(
                (1, patch_embeddings.num_patches + 1, config.embed_dim),
                "position_embeddings",
            )?;
            Some(position_embedding)
        } else {
            None
        };
        Ok(Self {
            patch_embeddings,
            position_embeddings,
            norm,
        })
    }
}

impl Module for SwinEmbeddings {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.patch_embeddings.forward(x)?;
        let (b, c, h, w) = x.dims4()?;
        let x = {
            let x = x.flatten_from(2)?.permute((0, 2, 1))?;
            let x = self.norm.forward(&x)?;
            x.permute((0, 2, 1))?.reshape(&[b, c, h, w])?
        };
        let x = if let Some(position_embedding) = &self.position_embeddings {
            let seq_len = h * w;
            let position_embedding = position_embedding.i((.., ..seq_len))?;
            let x = x.flatten_from(2)?.permute((0, 2, 1))?;
            let x = x.broadcast_add(&position_embedding)?;
            x.permute((0, 2, 1))?.reshape(&[b, c, h, w])?
        } else {
            x.clone()
        };
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct SwinIntermediate {
    dense: Linear,
    intermediate_act_fn: Activation,
}

impl SwinIntermediate {
    fn new(config: &SwinConfig, dim: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(
            dim,
            (dim as f64 * config.mlp_ratio) as usize,
            vb.pp("dense"),
        )?;
        let intermediate_act_fn = config.hidden_act;
        Ok(Self {
            dense,
            intermediate_act_fn,
        })
    }
}

impl Module for SwinIntermediate {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense.forward(&x)?;
        let x = self.intermediate_act_fn.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct SwinSelfOutput {
    dense: Linear,
}

impl SwinSelfOutput {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(dim, dim, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl Module for SwinSelfOutput {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct SwinOutput {
    dense: Linear,
}

impl SwinOutput {
    fn new(config: &SwinConfig, dim: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(dim * config.mlp_ratio as usize, dim, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl Module for SwinOutput {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct SwinPatchMerging {
    reduction: Linear,
    norm: LayerNorm,
}

impl SwinPatchMerging {
    fn new(config: &SwinConfig, dim: usize, vb: VarBuilder) -> Result<Self> {
        let reduction = linear_no_bias(dim * 4, dim * 2, vb.pp("reduction"))?;
        let norm = layer_norm(4 * dim, config.layer_norm_eps, vb.pp("norm"))?;
        Ok(Self { reduction, norm })
    }

    fn maybe_pad(x: &Tensor) -> Result<Tensor> {
        let (_, h, w, _) = x.dims4()?;
        let x = if h % 2 == 1 {
            x.pad_with_zeros(1, 0, 1)?
        } else {
            x.clone()
        };
        let x = if w % 2 == 1 {
            x.pad_with_zeros(2, 0, 1)?
        } else {
            x.clone()
        };
        Ok(x)
    }
}

impl Module for SwinPatchMerging {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = Self::maybe_pad(x)?;
        let (b, h, w, c) = x.dims4()?;
        let input_feature = {
            let x = x.reshape((b, 2, h / 2, 2, w / 2, c))?;
            let input_feature_0 = x.i((.., 0, .., 0, .., ..))?.squeeze(1)?.squeeze(2)?;
            let input_feature_1 = x.i((.., 0, .., 1, .., ..))?.squeeze(1)?.squeeze(2)?;
            let input_feature_2 = x.i((.., 1, .., 0, .., ..))?.squeeze(1)?.squeeze(2)?;
            let input_feature_3 = x.i((.., 1, .., 1, .., ..))?.squeeze(1)?.squeeze(2)?;
            let x = Tensor::cat(
                &[
                    input_feature_0,
                    input_feature_1,
                    input_feature_2,
                    input_feature_3,
                ],
                D::Minus1,
            )?;
            x.reshape((b, (), 4 * c))?
        };
        let x = self.norm.forward(&input_feature)?;
        self.reduction.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct SwinSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    relative_position_bias: Tensor,
    query: Linear,
    key: Linear,
    value: Linear,
}

impl SwinSelfAttention {
    fn new(dim: usize, num_heads: usize, window_size: usize, vb: VarBuilder) -> Result<Self> {
        let num_attention_heads = num_heads;
        let attention_head_size = dim / num_attention_heads;
        // let all_head_size = num_attention_heads * attention_head_size;
        let query = linear(dim, dim, vb.pp("query"))?;
        let key = linear(dim, dim, vb.pp("key"))?;
        let value = linear(dim, dim, vb.pp("value"))?;
        let relative_position_bias_table = vb.get(
            (
                (2 * window_size - 1) * (2 * window_size - 1),
                num_attention_heads,
            ),
            "relative_position_bias_table",
        )?;
        let relative_position_index = Self::generate_relative_position_index(
            window_size,
            relative_position_bias_table.device(),
        )?
        .flatten_all()?;
        let relative_position_bias = relative_position_bias_table.i(&relative_position_index)?;
        let relative_position_bias = relative_position_bias
            .reshape((
                window_size * window_size,
                window_size * window_size,
                num_attention_heads,
            ))?
            .permute((2, 0, 1))?
            .contiguous()?
            .unsqueeze(0)?;
        Ok(Self {
            num_attention_heads,
            attention_head_size,
            relative_position_bias,
            query,
            key,
            value,
        })
    }

    fn generate_relative_position_index(window_size: usize, device: &Device) -> Result<Tensor> {
        debug_assert!(window_size > 1, "window_size must be greater than 1");
        let window_size = window_size as i64;
        let h = Tensor::arange(0, window_size, device)?;
        let w = Tensor::arange(0, window_size, device)?;
        let xy_indexing = false; // use ij indexing
        let grids = Tensor::meshgrid(&[h, w], xy_indexing)?;
        let grid = Tensor::stack(&grids, 0)?.flatten_from(1)?;
        let grid = {
            let (_, w) = grid.shape().dims2()?;
            let left = grid.unsqueeze(2)?.repeat(Shape::from_dims(&[1, 1, w]))?;
            let right = grid.unsqueeze(1)?.repeat(Shape::from_dims(&[1, w, 1]))?;
            (left - right)?
        };
        let relative_grid = {
            let bias = Tensor::full(window_size - 1, &grid.shape().clone(), device)?;
            let relative_grid = (grid + bias)?;
            let m1 = relative_grid.i(0)?;
            let m2 = relative_grid.i(1)?;
            let scalar = Tensor::full(2 * window_size - 1, &m1.shape().clone(), device)?;
            let m1 = (m1 * scalar)?;
            Tensor::stack(&[m1, m2], 2)?
        };
        relative_grid.sum(2)?.to_dtype(DType::U32)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.shape().dims3()?;
        x.reshape((b, n, self.num_attention_heads, self.attention_head_size))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }
}

impl Module for SwinSelfAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        debug_assert_eq!(3, x.dims().len(), "Input tensor must have 3 dimensions");
        let key_layer = self.transpose_for_scores(&self.key.forward(&x)?)?;
        let query_layer = self.transpose_for_scores(&self.query.forward(&x)?)?;
        let value_layer = self.transpose_for_scores(&self.value.forward(&x)?)?;
        let attention_scores = (query_layer.matmul(&key_layer.t()?))?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(&self.relative_position_bias)?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.permute((0, 2, 1, 3))?;
        context_layer.flatten_from(2)
    }
}

#[derive(Debug, Clone)]
struct SwinAttention {
    self_attention: SwinSelfAttention,
    output: SwinSelfOutput,
}

impl SwinAttention {
    fn new(dim: usize, num_heads: usize, window_size: usize, vb: VarBuilder) -> Result<Self> {
        let self_attention = SwinSelfAttention::new(dim, num_heads, window_size, vb.pp("self"))?;
        let output = SwinSelfOutput::new(dim, vb.pp("output"))?;
        Ok(Self {
            self_attention,
            output,
        })
    }
}

impl Module for SwinAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.self_attention.forward(x)?;
        self.output.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct SwinLayer {
    shift_size: usize,
    window_size: usize,
    layernorm_before: LayerNorm,
    attention: SwinAttention,
    layernorm_after: LayerNorm,
    intermediate: SwinIntermediate,
    output: SwinOutput,
}

impl SwinLayer {
    fn new(
        config: &SwinConfig,
        dim: usize,
        num_heads: usize,
        shift_size: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layer_norm_eps = config.layer_norm_eps;
        let layernorm_before = layer_norm(dim, layer_norm_eps, vb.pp("layernorm_before"))?;
        let attention = SwinAttention::new(dim, num_heads, config.window_size, vb.pp("attention"))?;
        let layernorm_after = layer_norm(dim, layer_norm_eps, vb.pp("layernorm_after"))?;
        let intermediate = SwinIntermediate::new(config, dim, vb.pp("intermediate"))?;
        let output = SwinOutput::new(config, dim, vb.pp("output"))?;
        Ok(Self {
            shift_size: shift_size.unwrap_or_default(),
            window_size: config.window_size,
            layernorm_before,
            attention,
            layernorm_after,
            intermediate,
            output,
        })
    }

    /// the x tensor should be in [b, h, w, c] format
    fn maybe_pad(
        x: &Tensor,
        window_size: usize,
        height: usize,
        width: usize,
    ) -> Result<(Tensor, bool)> {
        let mut padded = false;
        let x = if width % window_size != 0 {
            padded = true;
            let pad_right = window_size - (width % window_size);
            x.pad_with_zeros(2, 0, pad_right)?
        } else {
            x.clone()
        };
        let x = if height % window_size != 0 {
            padded = true;
            let pad_bottom = window_size - (height % window_size);
            x.pad_with_zeros(1, 0, pad_bottom)?
        } else {
            x.clone()
        };
        Ok((x, padded))
    }

    fn get_attn_mask(
        shift_size: usize,
        window_size: usize,
        height: usize,
        width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        if shift_size > 0 {
            let xs = [0, height - window_size, height - shift_size, height];
            let ys = [0, width - window_size, width - shift_size, width];
            let mut count = 0i64;
            let mut rows = vec![];
            for (xs, xe) in xs.iter().zip(&xs[1..]) {
                let mut cols = vec![];
                for (ys, ye) in ys.iter().zip(&ys[1..]) {
                    let shape = (xe - xs, ye - ys);
                    let tensor = Tensor::full(count, shape, device)?;
                    cols.push(tensor);
                    count += 1;
                }
                let row = Tensor::cat(&cols, 1)?;
                rows.push(row);
            }
            let mask = Tensor::cat(&rows, 0)?;
            debug_assert_eq!(
                mask.dims2()?,
                (height, width),
                "mask shape must match input shape"
            );
            let mask = mask.unsqueeze(0)?.unsqueeze(3)?.to_dtype(dtype)?;
            let mask = Self::window_partition(&mask, window_size)?;
            let mask = mask.reshape(((), window_size * window_size))?;
            println!("mask dim {:?}", mask);
            let mask = (mask.unsqueeze(1)?.broadcast_sub(&mask.unsqueeze(2)?))?;
            let mask = (mask.ne(0i64)?.to_dtype(dtype)? * -100.0f64)?;
            Ok(Some(mask))
        } else {
            Ok(None)
        }
    }

    fn window_partition(x: &Tensor, window_size: usize) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        debug_assert!(
            h % window_size == 0 && w % window_size == 0,
            "input resolution must be divisible by window size"
        );
        let x = x.reshape((
            b,
            h / window_size,
            window_size,
            w / window_size,
            window_size,
            c,
        ))?;
        let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
        x.reshape(((), window_size, window_size, c))
    }

    fn window_reverse(x: &Tensor, window_size: usize, h: usize, w: usize) -> Result<Tensor> {
        let (b, _, _, c) = x.dims4()?;
        debug_assert!(
            h % window_size == 0 && w % window_size == 0,
            "input resolution must be divisible by window size"
        );
        let b = b * window_size * window_size / h / w;
        let x = x.reshape((
            b,
            h / window_size,
            w / window_size,
            window_size,
            window_size,
            c,
        ))?;
        let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
        x.reshape((b, h, w, c))
    }

    /// if window size is larger than input resolution, we don't partition windows
    fn get_shift_and_window_size(&self, h: usize, w: usize) -> (usize, usize) {
        let min = h.min(w);
        if min <= self.window_size {
            (0, min)
        } else {
            (self.shift_size, self.window_size)
        }
    }
}

impl Module for SwinLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let (shift_size, window_size) = self.get_shift_and_window_size(h, w);
        let shortcut = x;
        let x = {
            let x = x.flatten(1, 2)?;
            let x = self.layernorm_before.forward(&x)?;
            // note no permutation
            x.reshape((b, h, w, c))?
        };

        let (x, was_padded) = Self::maybe_pad(&x, window_size, h, w)?;
        let (_, padded_h, padded_w, _) = x.shape().dims4()?;

        // shift
        let x = if shift_size > 0 {
            let x = x.roll(-(shift_size as i32), 1)?;
            x.roll(-(shift_size as i32), 2)?
        } else {
            x
        };

        // partition
        let x = Self::window_partition(&x, window_size)?;

        // attention
        let x = {
            let (b, w1, w2, c) = x.dims4()?;
            debug_assert_eq!(w1, w2, "window size must be square");
            debug_assert_eq!(w1, window_size, "window size must be equal to window_size");
            let x = x.reshape((b, (), c))?;

            if let Some(_) =
                Self::get_attn_mask(shift_size, window_size, w1, w2, x.dtype(), x.device())?
            {
                // TODO attention mask
                println!("TODO must apply attention mask!");
            }

            let x = self.attention.forward(&x)?;
            x.reshape((b, w1, w2, c))?
        };

        // un-partition
        let x = Self::window_reverse(&x, window_size, padded_h, padded_w)?;

        // un-shift
        let x = if shift_size > 0 {
            let x = x.roll(shift_size as i32, 1)?;
            x.roll(shift_size as i32, 2)?
        } else {
            x
        };

        let x = if was_padded {
            x.narrow(1, 0, h)?.narrow(2, 0, w)?
        } else {
            x
        };

        let hidden_states = (x + shortcut)?;

        let x = self.layernorm_after.forward(&hidden_states)?;
        let x = self.intermediate.forward(&x)?;
        let x = self.output.forward(&x)?;

        x + hidden_states
    }
}

#[derive(Debug, Clone)]
struct SwinStage {
    blocks: Vec<SwinLayer>,
    downsample: Option<SwinPatchMerging>,
}

impl SwinStage {
    fn new(
        config: &SwinConfig,
        dim: usize,
        depth: usize,
        num_heads: usize,
        downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let blocks = (0..depth)
            .map(|i| {
                let shift_size = if i % 2 == 0 {
                    None
                } else {
                    Some(config.window_size / 2)
                };
                SwinLayer::new(
                    config,
                    dim,
                    num_heads,
                    shift_size,
                    vb.pp(&format!("blocks.{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let downsample = if downsample {
            Some(SwinPatchMerging::new(config, dim, vb.pp("downsample"))?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }
}

impl Module for SwinStage {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(downsample) = &self.downsample {
            x = downsample.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct SwinEncoder {
    layers: Vec<SwinStage>,
}

impl SwinEncoder {
    fn new(config: &SwinConfig, vb: VarBuilder) -> Result<Self> {
        let layers = (0..config.depths.len())
            .map(|i| {
                let dim = config.embed_dim * 2_usize.pow(i as u32);
                let depth = config.depths[i];
                let num_heads = config.num_heads[i];
                let downsample = i < config.depths.len() - 1;
                SwinStage::new(
                    config,
                    dim as usize,
                    depth,
                    num_heads,
                    downsample,
                    vb.pp(&format!("layers.{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
}

impl Module for SwinEncoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SwinModel {
    embeddings: SwinEmbeddings,
    encoder: SwinEncoder,
}

impl SwinModel {
    pub(crate) fn new(config: &SwinConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = SwinEmbeddings::new(config, vb.pp("embeddings"))?;
        let encoder = SwinEncoder::new(config, vb.pp("encoder"))?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }
}

impl Module for SwinModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(x)?;
        let x = self.encoder.forward(&x)?;
        // this is the same as adaptive avg pool with output size 1
        x.mean(1)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use candle_nn::{embedding, var_builder::VarBuilderArgs};

    #[test]
    fn test_swin_config_from_json() {
        let config_raw = r#"{
            "_name_or_path": "",
            "add_cross_attention": false,
            "architectures": [
              "DonutSwinModel"
            ],
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": null,
            "begin_suppress_tokens": null,
            "bos_token_id": null,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": null,
            "decoder_start_token_id": null,
            "depths": [
              2,
              2,
              14,
              2
            ],
            "diversity_penalty": 0.0,
            "do_sample": false,
            "drop_path_rate": 0.1,
            "early_stopping": false,
            "embed_dim": 128,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": null,
            "exponential_decay_length_penalty": null,
            "finetuning_task": null,
            "forced_bos_token_id": null,
            "forced_eos_token_id": null,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 1024,
            "id2label": {
              "0": "LABEL_0",
              "1": "LABEL_1"
            },
            "image_size": [
              196,
              896
            ],
            "initializer_range": 0.02,
            "is_decoder": false,
            "is_encoder_decoder": false,
            "label2id": {
              "LABEL_0": 0,
              "LABEL_1": 1
            },
            "layer_norm_eps": 0.00001,
            "length_penalty": 1.0,
            "max_length": 20,
            "min_length": 0,
            "mlp_ratio": 4.0,
            "model_type": "donut-swin",
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_channels": 3,
            "num_heads": [
              4,
              8,
              16,
              32
            ],
            "num_layers": 4,
            "num_return_sequences": 1,
            "output_attentions": false,
            "output_hidden_states": false,
            "output_scores": false,
            "pad_token_id": null,
            "patch_size": 4,
            "path_norm": true,
            "prefix": null,
            "problem_type": null,
            "pruned_heads": {},
            "qkv_bias": true,
            "remove_invalid_values": false,
            "repetition_penalty": 1.0,
            "return_dict": true,
            "return_dict_in_generate": false,
            "sep_token_id": null,
            "suppress_tokens": null,
            "task_specific_params": null,
            "temperature": 1.0,
            "tf_legacy_loss": false,
            "tie_encoder_decoder": false,
            "tie_word_embeddings": true,
            "tokenizer_class": null,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": "float32",
            "torchscript": false,
            "typical_p": 1.0,
            "use_2d_embeddings": false,
            "use_absolute_embeddings": true,
            "use_bfloat16": false,
            "window_size": 7
          }"#;
        let config: SwinConfig = serde_json::from_str(config_raw).unwrap();
        let default_config = SwinConfig::default();
        assert_eq!(config.image_size, default_config.image_size);
        assert_eq!(config.patch_size, default_config.patch_size);
        assert_eq!(config.num_channels, default_config.num_channels);
        assert_eq!(config.embed_dim, default_config.embed_dim);
        assert_eq!(config.depths, default_config.depths);
        assert_eq!(config.num_heads, default_config.num_heads);
        assert_eq!(config.window_size, default_config.window_size);
        assert_eq!(config.mlp_ratio, default_config.mlp_ratio);
        assert_eq!(config.qkv_bias, default_config.qkv_bias);
        assert_eq!(config.hidden_act, default_config.hidden_act);
        assert_eq!(
            config.use_absolute_embeddings,
            default_config.use_absolute_embeddings
        );
        assert_eq!(config.initializer_range, default_config.initializer_range);
        assert_eq!(config.layer_norm_eps, default_config.layer_norm_eps);
    }

    #[test]
    fn test_swin_patch_embeddings() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilderArgs::zeros(DType::F32, &device);
        let config = SwinConfig::tiny_patch4_window7_224();
        let module = SwinPatchEmbeddings::new(&config, vb)?;
        let x = Tensor::zeros(&[1, 3, 224, 224], DType::F32, &device)?;
        let result = module.forward(&x)?;
        assert_eq!(result.dims(), &[1, config.embed_dim, 56, 56]);
        Ok(())
    }

    // this is expensive, run using `cargo t -- --ignored`
    #[test]
    #[ignore]
    fn test_embeddings_value_compare() -> anyhow::Result<()> {
        use crate::hf::HfModelInfo;
        use crate::recognition::Config;
        use float_cmp::approx_eq;

        let device = Device::Cpu;
        let dtype = DType::F16;
        let model_info = HfModelInfo {
            model_type: "test",
            repo: "vikp/surya_rec".into(),
            weights_file: "model.safetensors".into(),
            config_file: "config.json".into(),
        };
        let (config_file, model_file) = model_info.download_model_files()?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let config =
            serde_json::from_str::<Config>(&std::fs::read_to_string(config_file)?)?.encoder;
        let embedding = SwinEmbeddings::new(&config, vb.pp("encoder").pp("embeddings"))?;

        let x = Tensor::ones(
            &[
                1,
                config.num_channels,
                config.image_size.0,
                config.image_size.1,
            ],
            dtype,
            &device,
        )?;
        {
            let y = embedding.patch_embeddings.forward(&x)?;
            let y_sum: f32 = y.to_dtype(DType::F32)?.sum_all()?.to_scalar()?;
            assert!(approx_eq!(
                f32,
                y_sum,
                -112499.0938,
                epsilon = 112499.0938 * 5e-3
            ));
        }
        {
            let y = embedding.forward(&x)?;
            let y_sum: f32 = y.to_dtype(DType::F32)?.sum_all()?.to_scalar()?;
            assert!(approx_eq!(
                f32,
                y_sum,
                140062.19,
                epsilon = 140062.19 * 5e-3
            ));
        }
        Ok(())
    }

    #[test]
    fn test_swin_embeddings() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilderArgs::zeros(DType::F32, &device);
        let config = SwinConfig::tiny_patch4_window7_224();
        let module = SwinEmbeddings::new(&config, vb)?;
        let x = Tensor::zeros(&[1, 3, 224, 224], DType::F32, &device)?;
        let result = module.forward(&x)?;
        assert_eq!(result.dims(), &[1, config.embed_dim, 56, 56]);
        Ok(())
    }

    #[test]
    fn test_generate_relative_position_index() -> Result<()> {
        let device = Device::Cpu;
        let result = SwinSelfAttention::generate_relative_position_index(3, &device)?;
        assert_eq!(
            result.to_vec2::<u32>()?,
            [
                [12, 11, 10, 7, 6, 5, 2, 1, 0],
                [13, 12, 11, 8, 7, 6, 3, 2, 1],
                [14, 13, 12, 9, 8, 7, 4, 3, 2],
                [17, 16, 15, 12, 11, 10, 7, 6, 5],
                [18, 17, 16, 13, 12, 11, 8, 7, 6],
                [19, 18, 17, 14, 13, 12, 9, 8, 7],
                [22, 21, 20, 17, 16, 15, 12, 11, 10],
                [23, 22, 21, 18, 17, 16, 13, 12, 11],
                [24, 23, 22, 19, 18, 17, 14, 13, 12]
            ]
        );
        Ok(())
    }

    #[test]
    fn test_swin_self_attention() -> Result<()> {
        let device = Device::Cpu;
        let dim = 96;
        let window_size = 7;
        let vb = VarBuilderArgs::zeros(DType::F32, &device);
        let module: SwinSelfAttention = SwinSelfAttention::new(dim, 3, window_size, vb)?;
        let x = Tensor::zeros(&[1, window_size * window_size, dim], DType::F32, &device)?;
        let result = module.forward(&x)?;
        assert_eq!(result.dims(), &[1, window_size * window_size, dim]);
        Ok(())
    }

    #[test]
    fn test_window_partition() -> Result<()> {
        let device = Device::Cpu;
        let dim = 96;
        let x = Tensor::zeros(&[1, 56, 56, dim], DType::F32, &device)?;
        let result = SwinLayer::window_partition(&x, 7)?;
        assert_eq!(result.dims(), &[56 * 56 / 7 / 7, 7, 7, dim]);
        Ok(())
    }

    #[test]
    fn test_window_reverse() -> Result<()> {
        let device = Device::Cpu;
        let dim = 96;
        let x = Tensor::zeros(&[56 * 56 / 7 / 7, 7, 7, dim], DType::F32, &device)?;
        let result = SwinLayer::window_reverse(&x, 7, 56, 56)?;
        assert_eq!(result.dims(), &[1, 56, 56, dim]);
        Ok(())
    }

    #[test]
    fn test_window_full_cycle() -> Result<()> {
        let device = Device::Cpu;
        let dim = 96;
        let x = Tensor::zeros(&[1, 56, 56, dim], DType::F32, &device)?;
        let window_size = 7;
        let x = SwinLayer::window_partition(&x, window_size)?;
        let x = SwinLayer::window_reverse(&x, window_size, 56, 56)?;
        assert_eq!(x.dims(), &[1, 56, 56, dim]);
        Ok(())
    }

    #[test]
    fn test_swin_layer() -> Result<()> {
        let device = Device::Cpu;
        let config = SwinConfig::tiny_patch4_window7_224();
        let dim = 96;
        let vb = VarBuilderArgs::zeros(DType::F32, &device);
        let x = Tensor::zeros(&[1, 56, 56, 96], DType::F32, &device)?;
        let module = SwinLayer::new(&config, dim, 3, None, vb)?;
        let result = module.forward(&x)?;
        assert_eq!(result.dims(), &[1, 56, 56, 96]);
        Ok(())
    }

    #[test]
    fn test_get_attn_mask() -> Result<()> {
        let device = Device::Cpu;
        let result = SwinLayer::get_attn_mask(0, 7, 56, 56, DType::F32, &device)?;
        assert!(result.is_none());

        let result = SwinLayer::get_attn_mask(3, 7, 56, 56, DType::F32, &device)?;
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.dims(), &[64, 49, 49]);

        let result = SwinLayer::get_attn_mask(1, 2, 4, 4, DType::I64, &device)?;
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.dims(), &[4, 4, 4]);
        assert_eq!(
            result.to_vec3::<i64>()?,
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [
                    [0, -100, 0, -100],
                    [-100, 0, -100, 0],
                    [0, -100, 0, -100],
                    [-100, 0, -100, 0]
                ],
                [
                    [0, 0, -100, -100],
                    [0, 0, -100, -100],
                    [-100, -100, 0, 0],
                    [-100, -100, 0, 0]
                ],
                [
                    [0, -100, -100, -100],
                    [-100, 0, -100, -100],
                    [-100, -100, 0, -100],
                    [-100, -100, -100, 0]
                ]
            ]
        );
        Ok(())
    }

    #[test]
    fn test_swin_patch_merging() -> Result<()> {
        let device = Device::Cpu;
        let dim = 96;
        let vb = VarBuilderArgs::zeros(DType::F32, &device);
        let config = SwinConfig::tiny_patch4_window7_224();
        let module = SwinPatchMerging::new(&config, dim, vb)?;
        let x = Tensor::zeros(&[1, 7, 7, 96], DType::F32, &device)?;
        let result = module.forward(&x)?;
        assert_eq!(result.dims(), &[1, 4 * 4, 96 * 2]);
        Ok(())
    }

    // skip this test for now
    // #[test]
    fn test_swin_encoder() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilderArgs::zeros(DType::F32, &device);
        let config = SwinConfig::tiny_patch4_window7_224();
        let module = SwinEncoder::new(&config, vb)?;
        let x = Tensor::zeros(&[1, 56, 56, 96], DType::F32, &device)?;
        let result = module.forward(&x)?;
        assert_eq!(result.dims(), &[1, 49, 768]);
        Ok(())
    }
}
