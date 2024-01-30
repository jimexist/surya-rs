use std::path::PathBuf;

use opencv::core::{self, Mat};
use opencv::imgcodecs;
use opencv::imgproc;

/// convert a rgb image to mat
pub fn save_image(image: &Mat, output_path: PathBuf) -> crate::Result<()> {
    let mut output = Mat::default();
    // Convert the image to BGR (OpenCV writes images in BGR format by default)
    imgproc::cvt_color(&image, &mut output, imgproc::COLOR_RGB2BGR, 0)?;
    imgcodecs::imwrite(
        output_path.as_os_str().to_str().unwrap(),
        &output,
        &core::Vector::new(),
    )?;
    Ok(())
}
