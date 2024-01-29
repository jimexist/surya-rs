use candle_core::{Device, IndexOp, Module};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use log::info;
use std::path::PathBuf;
use std::time::Instant;
use surya::bbox::{draw_bboxes, generate_bbox};
use surya::convert::image_to_mat;
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
    #[arg(help = "path to image")]
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
        help = "model's weights file name"
    )]
    weights_file_name: String,
    #[arg(long, default_value = "config.json", help = "model's config file name")]
    config_file_name: String,
    #[arg(long, default_value_t = true, help = "whether to generate bbox image")]
    generate_bbox_image: bool,
    #[arg(long, default_value_t = true, help = "whether to generate heatmap")]
    generate_heatmap: bool,
    #[arg(
        long,
        default_value_t = true,
        help = "whether to generate affinity map"
    )]
    generate_affinity_map: bool,
    #[arg(
        long,
        default_value = "./surya_output",
        help = "output directory, each file will be generating a subdirectory under this directory"
    )]
    output_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = DeviceType::Cpu)]
    device_type: DeviceType,
}

impl Args {
    fn get_model(
        &self,
        device: &Device,
        num_labels: usize,
    ) -> anyhow::Result<SemanticSegmentationModel> {
        let api = Api::new()?;
        let repo = api.model(self.model_repo.clone());
        info!("using model from HuggingFace repo {0}", self.model_repo);
        let model_file = repo.get(&self.weights_file_name)?;
        info!("using weights file '{0}'", self.weights_file_name);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], candle_core::DType::F32, device)?
        };
        let config_file = repo.get(&self.config_file_name)?;
        info!("using config file '{0}'", self.config_file_name);
        let config = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
        info!("loaded config: {:?}, num_labels {}", config, num_labels);
        Ok(SemanticSegmentationModel::new(&config, num_labels, vb)?)
    }
}

const NUM_LABELS: usize = 2;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    let device = args.device_type.try_into()?;
    info!("using device {:?}", device);

    let (image, original_size) = read_resized_image(&args.image)?;
    info!("image original size (w, h)={original_size:?}");
    let image = load_image_tensor(image, &device)?;

    // join the output dir with the input image's base name
    let output_dir = args
        .image
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("failed to get file stem of {:?}", args.image))?;
    let output_dir = args.output_dir.join(output_dir);
    std::fs::DirBuilder::new()
        .recursive(true)
        .create(output_dir.clone())?;
    info!("generating output to {:?}", output_dir);

    let model = args.get_model(&device, NUM_LABELS)?;

    let input = image.unsqueeze(0)?;
    let now = Instant::now();
    let segmentation = model.forward(&input)?;
    info!("inference took {:.3}s", now.elapsed().as_secs_f32());
    let segmentation = segmentation.squeeze(0)?;

    let non_max_suppression_threshold = 0.35;
    let extract_text_threshold = 0.6;
    let bbox_area_threshold = 10;

    let bboxes = generate_bbox(
        original_size,
        segmentation.i(0)?.to_vec2::<f32>()?,
        non_max_suppression_threshold,
        extract_text_threshold,
        bbox_area_threshold,
    )?;

    if args.generate_bbox_image {
        info!("generating bbox image");
        let image = image::io::Reader::open(args.image)?.decode()?;
        let mut image = image_to_mat(image)?;
        let output_file = output_dir.join("bbox.png");
        let output_file = output_file.to_str().expect("failed to convert to str");
        draw_bboxes(&mut image, bboxes, output_file)?;
    }

    if args.generate_heatmap {
        info!("generating heatmap");
        let heatmap = segmentation.i(0)?;
        let imgbuf = heatmap_to_gray_image(heatmap, original_size)?;
        imgbuf.save(output_dir.join("heatmap.png"))?;
    }
    if args.generate_affinity_map {
        info!("generating affinity map");
        let affinity_map = segmentation.i(1)?;
        let imgbuf = heatmap_to_gray_image(affinity_map, original_size)?;
        imgbuf.save(output_dir.join("affinity_map.png"))?;
    }

    Ok(())
}
