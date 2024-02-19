//! HuggingFace API

use crate::error::Result;
use candle_core::Device;
use hf_hub::api::sync::ApiBuilder;
use log::debug;
use std::path::PathBuf;

pub struct HfModelInfo {
    pub model_type: &'static str,
    pub repo: String,
    pub weights_file: String,
    pub config_file: String,
}

pub trait HfModel {
    fn from_hf(info: HfModelInfo, device: &Device) -> Result<Self>
    where
        Self: Sized,
    {
        let api = ApiBuilder::new().with_progress(true).build()?;
        let repo = api.model(info.repo.clone());
        debug!(
            "using {} model from HuggingFace repo '{}'",
            info.model_type, info.repo,
        );
        let model_file = repo.get(&info.weights_file)?;
        debug!(
            "using {} weights file '{}'",
            info.model_type, info.weights_file
        );
        let config_file = repo.get(&info.config_file)?;
        debug!(
            "using {} config file '{}'",
            info.model_type, info.config_file
        );
        Self::from_hf_files(config_file, model_file, device)
    }

    fn from_hf_files(config: PathBuf, weights: PathBuf, device: &Device) -> Result<Self>
    where
        Self: Sized;
}
