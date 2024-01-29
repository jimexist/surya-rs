use image::DynamicImage;
use image::GenericImageView;
use opencv::core::{self, Mat};

/// convert a rgb image to mat
pub fn image_to_mat(image: DynamicImage) -> opencv::Result<Mat> {
    let (width, height) = image.dimensions();
    // for each box generate a red rectangle
    let image = image.clone().to_rgb8();
    let mut raw_pixels = image.into_raw();
    // Create an OpenCV Mat from the raw bytes
    let image = unsafe {
        Mat::new_rows_cols_with_data(
            height as i32,
            width as i32,
            core::CV_8UC3,
            raw_pixels.as_mut_ptr() as *mut std::ffi::c_void,
            core::Mat_AUTO_STEP,
        )?
    };
    Ok(image)
}
