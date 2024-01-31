use opencv::core::{self, Mat, Size};
use opencv::prelude::MatTraitConst;
use opencv::{imgcodecs, imgproc};
use std::path::Path;

/// convert an image from map to gray scale image and save it to output_path
pub fn save_grayscale_image_with_resize<P: AsRef<Path>>(
    image: &Mat,
    size: Size,
    output_path: P,
) -> crate::Result<()> {
    // convert image [0,1) to 255 grayscale image
    let mut gray_scale_image = Mat::default();
    image.convert_to(&mut gray_scale_image, core::CV_8UC1, 255.0, 0.0)?;
    // resize image
    let mut resized_image = Mat::default();
    imgproc::resize(
        &gray_scale_image,
        &mut resized_image,
        size,
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    imgcodecs::imwrite(
        output_path.as_ref().as_os_str().to_str().unwrap(),
        &resized_image,
        &core::Vector::new(),
    )?;
    Ok(())
}
