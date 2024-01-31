use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use log::{debug, info};
use opencv::hub_prelude::MatTraitConst;
use std::path::PathBuf;
use std::time::Instant;
use surya::bbox::{draw_bboxes, generate_bbox};
use surya::postprocess::save_grayscale_image_with_resize;
use surya::preprocess::{image_to_tensor, read_chunked_resized_image, read_image};
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
    ) -> surya::Result<SemanticSegmentationModel> {
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

fn main() -> surya::Result<()> {
    env_logger::init();

    let args = Args::parse();

    let device = args.device_type.try_into()?;
    info!("using device {:?}", device);

    let image_chunks = read_chunked_resized_image(&args.image)?;

    // join the output dir with the input image's base name
    let output_dir = args.image.file_stem().expect("failed to get file stem");
    let output_dir = args.output_dir.join(output_dir);
    std::fs::DirBuilder::new()
        .recursive(true)
        .create(output_dir.clone())?;
    info!("generating output to {:?}", output_dir);

    let model = args.get_model(&device, NUM_LABELS)?;

    let batch_size = 2;
    let image_tensors: Vec<Tensor> = image_chunks
        .resized_chunks
        .iter()
        .map(|img| image_to_tensor(img, &device))
        .collect::<surya::Result<_>>()?;

    let mut heatmaps = Vec::new();
    let mut affinity_maps = Vec::new();
    for batch in image_tensors.chunks(batch_size) {
        let batch = Tensor::stack(batch, 0)?;
        let now = Instant::now();
        let segmentation = model.forward(&batch)?;
        info!("inference took {:.3}s", now.elapsed().as_secs_f32());
        for i in 0..batch_size {
            let heatmap: Tensor = segmentation.i(i)?.squeeze(0)?.i(0)?;
            let affinity_map: Tensor = segmentation.i(i)?.squeeze(0)?.i(1)?;
            heatmaps.push(heatmap);
            affinity_maps.push(affinity_map);
        }
    }

    let heatmap = image_chunks.stitch_image_tensors(heatmaps)?;
    let affinity_map = image_chunks.stitch_image_tensors(affinity_maps)?;

    debug!("heatmap {:?}", heatmap);
    debug!("affinity_map {:?}", affinity_map);

    let non_max_suppression_threshold = 0.35;
    let extract_text_threshold = 0.6;
    let bbox_area_threshold = 10;

    let bboxes = generate_bbox(
        &heatmap,
        non_max_suppression_threshold,
        extract_text_threshold,
        bbox_area_threshold,
    )?;

    if args.generate_bbox_image {
        let mut image = read_image(args.image)?;
        let output_file = output_dir.join("bbox.png");
        draw_bboxes(
            &mut image,
            heatmap.size()?,
            image_chunks.original_size_with_padding,
            bboxes,
            &output_file,
        )?;
        info!("bbox image {:?} generated", output_file);
    }

    if args.generate_heatmap {
        let output_file = output_dir.join("heatmap.png");
        save_grayscale_image_with_resize(&heatmap, image_chunks.original_size, &output_file)?;
        info!("heatmap image {:?} generated", output_file);
    }

    if args.generate_affinity_map {
        let output_file = output_dir.join("affinity_map.png");
        save_grayscale_image_with_resize(&affinity_map, image_chunks.original_size, &output_file)?;
        info!("affinity map image {:?} generated", output_file);
    }

    Ok(())
}
