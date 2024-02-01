#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use env_logger::Env;
use hf_hub::api::sync::ApiBuilder;
use log::{debug, info};
use opencv::hub_prelude::MatTraitConst;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use surya::bbox::{draw_bboxes, generate_bbox};
use surya::postprocess::save_image;
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
struct Cli {
    #[arg(help = "path to image")]
    image: PathBuf,

    #[arg(
        long,
        help = "detection batch size, if not supplied defaults to 2 on CPU and 16 on GPU"
    )]
    batch_size: Option<usize>,

    #[arg(
        long,
        default_value = "vikp/line_detector",
        help = "detection model's hugging face repo"
    )]
    model_repo: String,

    #[arg(
        long,
        default_value = "model.safetensors",
        help = "detection model's weights file name"
    )]
    weights_file_name: String,

    #[arg(
        long,
        default_value = "config.json",
        help = "detection model's config file name"
    )]
    config_file_name: String,

    #[arg(
        long,
        default_value_t = 0.35,
        help = "a value between 0.0 and 1.0 to filter low density part of heatmap"
    )]
    non_max_suppression_threshold: f64,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "a value between 0.0 and 1.0 to filter out bbox with low heatmap density"
    )]
    extract_text_threshold: f64,

    #[arg(
        long,
        default_value_t = 10,
        help = "a pixel threshold to filter out small area bbox"
    )]
    bbox_area_threshold: usize,

    #[arg(
        long = "polygons",
        default_value_t = true,
        help = "whether to output polygons json file"
    )]
    output_polygons: bool,

    #[arg(
        long = "image",
        default_value_t = true,
        help = "whether to generate bbox image"
    )]
    generate_bbox_image: bool,

    #[arg(
        long = "heatmap",
        default_value_t = true,
        help = "whether to generate heatmap"
    )]
    generate_heatmap: bool,

    #[arg(
        long = "affinity-map",
        default_value_t = true,
        help = "whether to generate affinity map"
    )]
    generate_affinity_map: bool,

    #[arg(
        long,
        default_value = "./surya_output",
        help = "output directory, under which the input image will be generating a subdirectory"
    )]
    output_dir: PathBuf,

    #[arg(long = "device", value_enum, default_value_t = DeviceType::Cpu)]
    device_type: DeviceType,

    #[arg(long, help = "whether to enable verbose mode")]
    verbose: bool,
}

impl Cli {
    fn get_model(
        &self,
        device: &Device,
        num_labels: usize,
    ) -> surya::Result<SemanticSegmentationModel> {
        let api = ApiBuilder::new().with_progress(true).build()?;
        let repo = api.model(self.model_repo.clone());
        debug!("using model from HuggingFace repo {0}", self.model_repo);
        let model_file = repo.get(&self.weights_file_name)?;
        debug!("using weights file '{0}'", self.weights_file_name);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], candle_core::DType::F32, device)?
        };
        let config_file = repo.get(&self.config_file_name)?;
        debug!("using config file '{0}'", self.config_file_name);
        let config = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
        debug!("loaded config: {:?}, num_labels {}", config, num_labels);
        Ok(SemanticSegmentationModel::new(&config, num_labels, vb)?)
    }
}

const NUM_LABELS: usize = 2;

fn main() -> surya::Result<()> {
    let args = Cli::parse();
    let env = Env::new().filter_or("SURYA_LOG", if args.verbose { "debug" } else { "info" });
    env_logger::init_from_env(env);

    assert!(
        0.0 <= args.non_max_suppression_threshold && args.non_max_suppression_threshold <= 1.0,
        "non-max-suppression-threshold must be between 0.0 and 1.0"
    );
    assert!(
        0.0 <= args.extract_text_threshold && args.extract_text_threshold <= 1.0,
        "extract-text-threshold must be between 0.0 and 1.0"
    );
    assert!(
        args.bbox_area_threshold > 0,
        "bbox-area-threshold must be > 0"
    );

    let device = args.device_type.try_into()?;
    debug!("using device {:?}", device);

    let image_chunks = read_chunked_resized_image(&args.image)?;

    // join the output dir with the input image's base name
    let output_dir = args.image.file_stem().expect("failed to get file stem");
    let output_dir = args.output_dir.join(output_dir);
    std::fs::DirBuilder::new()
        .recursive(true)
        .create(output_dir.clone())?;
    info!("generating output to {:?}", output_dir);

    let model = args.get_model(&device, NUM_LABELS)?;

    let batch_size = args.batch_size.unwrap_or_else(|| match device {
        Device::Cpu => 2,
        Device::Cuda(_) | Device::Metal(_) => 16,
    });
    let image_tensors: Vec<Tensor> = image_chunks
        .resized_chunks
        .iter()
        .map(|img| image_to_tensor(img, &device))
        .collect::<surya::Result<_>>()?;

    let mut heatmaps = Vec::new();
    let mut affinity_maps = Vec::new();
    for batch in image_tensors.chunks(batch_size) {
        let batch_size = batch.len();
        let batch = Tensor::stack(batch, 0)?;
        info!(
            "starting segformer inference with batch size {}...",
            batch_size,
        );
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

    let bboxes = generate_bbox(
        &heatmap,
        args.non_max_suppression_threshold,
        args.extract_text_threshold,
        args.bbox_area_threshold as i32,
    )?;

    if args.output_polygons {
        let output_file = output_dir.join("polygons.jsonl");
        let mut buf_writer = BufWriter::new(File::create(&output_file)?);
        for bbox in &bboxes {
            let polygons: Vec<(f32, f32)> = bbox
                .polygon
                .iter()
                .map(|p| {
                    let precision = 1.0e3;
                    let x = (p.x * precision).round() / precision;
                    let y = (p.y * precision).round() / precision;
                    (x, y)
                })
                .collect();
            serde_json::to_writer(&mut buf_writer, &polygons)?;
            writeln!(&mut buf_writer)?;
        }
        buf_writer.flush()?;
        info!("polygons json file {:?} generated", output_file);
    }

    if args.generate_bbox_image {
        let mut image = read_image(args.image)?;
        let output_file = output_dir.join("bbox.png");
        draw_bboxes(
            &mut image,
            heatmap.size()?,
            image_chunks.original_size_with_padding,
            &bboxes,
            &output_file,
        )?;
        info!("bbox image {:?} generated", output_file);
    }

    if args.generate_heatmap {
        let output_file = output_dir.join("heatmap.png");
        let image = image_chunks.resize_heatmap_to_image(heatmap)?;
        save_image(&image, &output_file)?;
        info!("heatmap image {:?} generated", output_file);
    }

    if args.generate_affinity_map {
        let output_file = output_dir.join("affinity_map.png");
        let image = image_chunks.resize_heatmap_to_image(affinity_map)?;
        save_image(&image, &output_file)?;
        info!("affinity map image {:?} generated", output_file);
    }

    Ok(())
}
