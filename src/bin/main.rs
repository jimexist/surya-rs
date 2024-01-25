use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use log::info;
use surya::segformer::SemanticSegmentationModel;

#[derive(Debug, ValueEnum, Clone, Copy)]
enum Device {
    Cpu,
    Gpu,
    #[cfg(feature = "metal")]
    Metal,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, help = "path to image")]
    image: PathBuf,
    #[arg(
        long,
        default_value = "vikp/line_detector",
        help = "model's hugging face repo"
    )]
    model_repo: String,
    #[arg(
        long,
        default_value = "model.safetensors",
        help = "model's weights name"
    )]
    weights_name: String,
    #[arg(long, value_enum, default_value_t = Device::Cpu)]
    device_type: Device,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let api = Api::new()?;
    info!("downloading {} from hugging face", &args.model_repo);
    let repo = api.model(args.model_repo);
    let file = repo.get(&args.weights_name)?;
    info!("done");
    Ok(())
}
