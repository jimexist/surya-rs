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

impl HfModelInfo {
    pub fn download_model_files(&self) -> Result<(PathBuf, PathBuf)> {
        let api = ApiBuilder::new().with_progress(true).build()?;
        let repo = api.model(self.repo.clone());
        debug!(
            "using {} model from HuggingFace repo '{}'",
            self.model_type, self.repo,
        );
        let model_file = repo.get(&self.weights_file)?;
        debug!(
            "using {} weights file '{}'",
            self.model_type, self.weights_file
        );
        let config_file = repo.get(&self.config_file)?;
        debug!(
            "using {} config file '{}'",
            self.model_type, self.config_file
        );
        Ok((config_file, model_file))
    }
}

pub trait HfModel {
    fn from_hf(info: HfModelInfo, device: &Device) -> Result<Self>
    where
        Self: Sized,
    {
        let (config_file, model_file) = info.download_model_files()?;
        Self::from_hf_files(config_file, model_file, device)
    }

    fn from_hf_files(config: PathBuf, weights: PathBuf, device: &Device) -> Result<Self>
    where
        Self: Sized;
}
