use candle_core::{DType, Device, Tensor};
use opencv::{
    core,
    imgcodecs::{self, IMREAD_COLOR},
    imgproc,
    prelude::*,
};
use std::path::Path;

const INPUT_IMAGE_SIZE: i32 = 896;

/// load image from path and resize it to [INPUT_IMAGE_SIZE] and return the resized image and
/// its original size
pub fn read_resized_image<P: AsRef<Path>>(image_path: P) -> crate::Result<(Mat, core::Size)> {
    let image = read_image(image_path)?;
    let original_size = core::Size::new(image.cols(), image.rows());
    let mut resized_image = Mat::default();
    imgproc::resize(
        &image,
        &mut resized_image,
        core::Size::new(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    Ok((resized_image, original_size))
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

/// convert heatmap tensor to gray image, assuming 2 dimensional tensor
pub fn heatmap_to_gray_image(heatmap: Tensor, original_size: core::Size) -> crate::Result<Mat> {
    let (height, width) = heatmap.dims2()?;
    assert_eq!(height, width, "original heatmap must be square");
    // Scale the heatmap values to 0-255 and convert to u8
    // Scale the heatmap values to 0-255 and convert to u8
    let heatmap: Vec<Vec<u8>> = (heatmap * 255.0)?.to_dtype(DType::U8)?.to_vec2()?;
    // Create a new Mat for the grayscale image
    let mut img =
        unsafe { Mat::new_size(core::Size::new(width as i32, height as i32), core::CV_8UC1)? };
    // Fill the Mat with heatmap values
    for (x, row) in heatmap.iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            *(img.at_2d_mut::<u8>(x as i32, y as i32)?) = value;
        }
    }
    let mut resized_image = unsafe {
        Mat::new_size(
            core::Size::new(original_size.width, original_size.height),
            core::CV_8UC1,
        )?
    };
    imgproc::resize(
        &img,
        &mut resized_image,
        original_size,
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    Ok(resized_image)
}
