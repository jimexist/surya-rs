use crate::postprocess::ImageChunks;
use candle_core::{Device, Tensor};
use log::debug;
use opencv::{
    core::{self},
    imgcodecs::{self, IMREAD_COLOR},
    imgproc,
    prelude::*,
};
use std::path::Path;

const INPUT_IMAGE_SIZE: i32 = 896;
const IMAGE_CHUNK_HEIGHT: i32 = 1200;

/// load image from path and resize it to [INPUT_IMAGE_SIZE] and return the resized image and
/// its original size
pub fn read_chunked_resized_image<P: AsRef<Path>>(image_path: P) -> crate::Result<ImageChunks> {
    let image = read_image(image_path)?;
    let original_size = core::Size::new(image.cols(), image.rows());

    let num_chunks = (original_size.height as f32 / IMAGE_CHUNK_HEIGHT as f32).ceil() as usize;
    debug_assert!(num_chunks > 0, "image must have at least one chunk");

    // pad the image with black pixels to make it divisible by chunk_height
    let mut padding: i32 = original_size.height % IMAGE_CHUNK_HEIGHT;
    if padding > 0 {
        padding = IMAGE_CHUNK_HEIGHT - padding;
        debug_assert!(padding > 0, "padding must be (still) greater than 0");
    }
    debug!(
        "image size is (w, h)=({}, {}), padding with {}",
        original_size.width, original_size.height, padding
    );

    let image = if padding > 0 {
        let mut padded_image = Mat::default();
        core::copy_make_border(
            &image,
            &mut padded_image,
            0,
            padding,
            0,
            0,
            core::BORDER_CONSTANT,
            core::Scalar::all(0.),
        )?;
        padded_image
    } else {
        image
    };
    debug_assert_eq!(
        image.rows() % IMAGE_CHUNK_HEIGHT,
        0,
        "image height must be divisible by {}",
        IMAGE_CHUNK_HEIGHT
    );

    let resized_chunks = (0..num_chunks)
        .map(|i| {
            let start = (i as i32) * IMAGE_CHUNK_HEIGHT;
            let roi: core::Rect_<i32> = core::Rect::new(0, start, image.cols(), IMAGE_CHUNK_HEIGHT);
            let chunk = Mat::roi(&image, roi)?;
            let size = core::Size::new(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
            resize(chunk, size)
        })
        .collect::<crate::Result<Vec<_>>>()?;

    Ok(ImageChunks {
        resized_chunks,
        padding,
        original_size,
        original_size_with_padding: core::Size::new(
            original_size.width,
            original_size.height + padding,
        ),
    })
}

/// read image into a matrix
pub fn read_image<P: AsRef<Path>>(image_path: P) -> crate::Result<Mat> {
    let image = imgcodecs::imread(image_path.as_ref().to_str().unwrap(), IMREAD_COLOR)?;
    Ok(image)
}

/// load dynamic image into a device tensor
pub fn image_to_tensor(input: &Mat, device: &Device) -> crate::Result<Tensor> {
    let mut image = Mat::default();
    // Convert the image to RGB (OpenCV reads images in BGR format by default)
    imgproc::cvt_color(input, &mut image, imgproc::COLOR_BGR2RGB, 0)?;
    // Get the dimensions of the image
    let size = image.size()?;
    let width = size.width;
    let height = size.height;
    // Convert the Mat to a slice of u8 and then to a Tensor and reshape it
    let data = Tensor::from_slice(
        image.data_bytes()?,
        (height as usize, width as usize, 3),
        device,
    )?
    .permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], device)?.reshape((3, 1, 1))?;
    Ok((data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?)
}

fn resize(image: Mat, new_size: core::Size) -> crate::Result<Mat> {
    let mut resized_image = Mat::default();
    imgproc::resize(
        &image,
        &mut resized_image,
        new_size,
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    Ok(resized_image)
}
