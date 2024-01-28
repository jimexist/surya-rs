use candle_core::{Device, Result as CandleResult, Tensor};
use image::{io::Reader, DynamicImage, GenericImageView, GrayImage, ImageResult, Luma};
use std::path::Path;

const INPUT_IMAGE_SIZE: u32 = 896;

/// load image from path and resize it to [INPUT_IMAGE_SIZE] and return the resized image and
/// its original size
pub fn read_resized_image<P: AsRef<Path>>(
    image_path: P,
) -> ImageResult<(DynamicImage, (u32, u32))> {
    let image = Reader::open(image_path)?.decode()?;
    let dimensions = image.dimensions();
    let image = image.resize_exact(
        INPUT_IMAGE_SIZE,
        INPUT_IMAGE_SIZE,
        image::imageops::FilterType::Triangle,
    );
    Ok((image, dimensions))
}

/// load dynamic image into a device tensor
pub fn load_image_tensor(image: DynamicImage, device: &Device) -> CandleResult<Tensor> {
    let image = image.to_rgb8();
    let (width, height) = image.dimensions();
    let data = image.into_raw();
    let data =
        Tensor::from_vec(data, (height as usize, width as usize, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], device)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

/// convert heatmap tensor to gray image, assuming 2 dimensional tensor
pub fn heatmap_to_gray_image(
    heatmap: Tensor,
    original_size: (u32, u32),
) -> CandleResult<DynamicImage> {
    let (height, width) = heatmap.dims2()?;
    assert_eq!(height, width, "original heatmap must be square");
    let heatmap = (heatmap * 255.0)?;
    let heatmap = heatmap.to_dtype(candle_core::DType::U8)?;
    let heatmap = heatmap.to_vec2()?;
    let mut imgbuf = GrayImage::new(width as u32, height as u32);
    for (x, row) in heatmap.iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            imgbuf.put_pixel(y as u32, x as u32, Luma([value]));
        }
    }
    let image: DynamicImage = imgbuf.into();
    Ok(image.resize_exact(
        original_size.0,
        original_size.1,
        image::imageops::FilterType::Triangle,
    ))
}
