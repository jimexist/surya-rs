use opencv::core::{self, Mat};
use opencv::imgcodecs;
use opencv::prelude::MatTraitConst;
use std::path::PathBuf;

/// save a grayscale image to an image
pub fn save_grayscale_image(image: &Mat, output_path: PathBuf) -> crate::Result<()> {
    // convert image 0..1 to 255 grayscale image
    let mut gray_scale_image = Mat::default();
    image.convert_to(&mut gray_scale_image, core::CV_8UC1, 255.0, 0.0)?;
    imgcodecs::imwrite(
        output_path.as_os_str().to_str().unwrap(),
        &gray_scale_image,
        &core::Vector::new(),
    )?;
    Ok(())
}
