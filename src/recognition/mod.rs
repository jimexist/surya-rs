//! The recognition module consists of donut encoder and an MBart decoder

mod mbart;
mod swin_transformer;

use crate::hf::HfModel;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use mbart::MBart;
use mbart::MBartConfig;
use std::path::PathBuf;
use swin_transformer::SwinConfig;
use swin_transformer::SwinModel;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    encoder: SwinConfig,
    decoder: MBartConfig,
}

#[derive(Debug, Clone)]
pub struct RecognitionModel {
    encoder: SwinModel,
    decoder: MBart,
}

impl RecognitionModel {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = SwinModel::new(&config.encoder, vb.pp("encoder"))?;
        let decoder = MBart::new(&config.decoder, vb.pp("decoder"))?;
        Ok(Self { encoder, decoder })
    }
}

impl Module for RecognitionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let encoded = self.encoder.forward(input)?;
        self.decoder.forward(&encoded)
    }
}

impl HfModel for RecognitionModel {
    fn from_hf_files(
        config: PathBuf,
        weights: PathBuf,
        device: &Device,
    ) -> crate::error::Result<Self> {
        let config = serde_json::from_str(&std::fs::read_to_string(config)?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], candle_core::DType::F16, device)?
        };
        Self::new(&config, vb).map_err(Into::into)
    }
}
