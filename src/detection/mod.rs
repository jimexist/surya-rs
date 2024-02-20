//! Detection module, consisting of segformer implementation

mod segformer;

use crate::error::Result;
use crate::hf::HfModel;
use candle_core::Device;
use candle_nn::VarBuilder;
pub use segformer::Config;
pub use segformer::SemanticSegmentationModel;
use std::path::PathBuf;

impl HfModel for SemanticSegmentationModel {
    fn from_hf_files(config: PathBuf, weights: PathBuf, device: &Device) -> Result<Self> {
        let config = serde_json::from_str(&std::fs::read_to_string(config)?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], candle_core::DType::F32, device)?
        };
        Self::new(&config, 2, vb).map_err(Into::into)
    }
}
