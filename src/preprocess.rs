use candle_core::{Device, Result as CandleResult, Tensor};
use image::{io::Reader, DynamicImage, GrayImage, ImageResult, Luma};
use log::debug;
use std::path::Path;

const IMAGE_SIZE: u32 = 224;

/// load image from path and resize it to IMAGE_SIZE
pub fn read_resized_image<P: AsRef<Path>>(image_path: P) -> ImageResult<DynamicImage> {
    let image = Reader::open(image_path)?.decode()?.resize_to_fill(
        IMAGE_SIZE,
        IMAGE_SIZE,
        image::imageops::FilterType::Triangle,
    );
    Ok(image)
}

/// load dynamic image into a device tensor
pub fn load_image_tensor(image: DynamicImage, device: &Device) -> CandleResult<Tensor> {
    let image = image.to_rgb8();
    let data = image.into_raw();
    let data = Tensor::from_vec(data, (224, 224, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], device)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

/// convert heatmap tensor to gray image, assuming 2 dimensional tensor
pub fn heatmap_to_gray_image(heatmap: Tensor) -> CandleResult<DynamicImage> {
    debug!("heatmap {:?}", heatmap.to_vec2::<f32>()?);
    let heatmap = (heatmap * 255.0)?;
    let heatmap = heatmap.to_dtype(candle_core::DType::U8)?;
    let heatmap = heatmap.to_vec2()?;
    let mut imgbuf = GrayImage::new(IMAGE_SIZE, IMAGE_SIZE);
    for (x, row) in heatmap.iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            imgbuf.put_pixel(y as u32, x as u32, Luma([value]));
        }
    }
    Ok(imgbuf.into())
}
