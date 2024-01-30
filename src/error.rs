use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SuryaError {
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("OpenCV error: {0}")]
    OpenCVError(#[from] opencv::Error),
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Json deser error: {0}")]
    JsonDeserError(#[from] serde_json::Error),
    #[error("Hugging Face Hub error: {0}")]
    ApiError(#[from] hf_hub::api::sync::ApiError),
}

pub type Result<T> = std::result::Result<T, SuryaError>;
