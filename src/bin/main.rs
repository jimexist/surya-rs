use candle_core::{Device, IndexOp, Module};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;

use log::info;
use std::path::PathBuf;
use surya::preprocess::{heatmap_to_gray_image, load_image_tensor, read_resized_image};
use surya::segformer::SemanticSegmentationModel;

#[derive(Debug, ValueEnum, Clone, Copy)]
enum DeviceType {
    Cpu,
    Gpu,
    #[cfg(feature = "metal")]
    Metal,
}

impl TryInto<Device> for DeviceType {
    type Error = candle_core::Error;

    fn try_into(self) -> Result<Device, Self::Error> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Gpu => Device::new_cuda(0),
            #[cfg(feature = "metal")]
            Self::Metal => Device::new_metal(0),
        }
    }
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
    #[arg(long, value_enum, default_value_t = DeviceType::Cpu)]
    device_type: DeviceType,
}

impl Args {
    fn get_var_builder(&self, device: &Device) -> anyhow::Result<VarBuilder> {
        let api = Api::new()?;
        let repo = api.model(self.model_repo.clone());
        info!("downloading {} from hugging face", &self.model_repo);
        let model_file = repo.get(&self.weights_name)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], candle_core::DType::F32, device)?
        };
        Ok(vb)
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = args.device_type.try_into()?;
    let vb = args.get_var_builder(&device)?;
    let num_labels = 2;
    info!(
        "loaded model from {} with weights file {}",
        args.model_repo, args.weights_name
    );
    let (image, origin_size) = read_resized_image(&args.image)?;
    info!("image original size {origin_size:?}");
    let image = load_image_tensor(image, &device)?;
    let input = image.unsqueeze(0)?;
    let config = Default::default();
    let model = SemanticSegmentationModel::new(&config, num_labels, vb)?;
    let segmentation = model.forward(&input)?;
    let segmentation = segmentation.squeeze(0)?;
    {
        let heatmap = segmentation.i(0)?;
        let imgbuf = heatmap_to_gray_image(heatmap, origin_size)?;
        imgbuf.save("heatmap.png")?;
    }
    {
        let affinity_map = segmentation.i(1)?;
        let imgbuf = heatmap_to_gray_image(affinity_map, origin_size)?;
        imgbuf.save("affinity_map.png")?;
    }
    Ok(())
}
