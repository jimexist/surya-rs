//! MBart with MOE
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

// TODO this is a placeholder

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct MBartConfig {}

#[derive(Debug, Clone)]
pub(crate) struct MBart {}

impl MBart {
    pub(crate) fn new(config: &MBartConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for MBart {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}
