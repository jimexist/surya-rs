use std::path::Path;

use candle_core::Tensor;
use opencv::core::{self, Mat};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use opencv::types::VectorOfMat;

pub struct ImageChunks {
    pub resized_chunks: Vec<Mat>,
    pub padding: i32,
    pub original_size: core::Size,
    pub original_size_with_padding: core::Size,
}

impl ImageChunks {
    pub fn stitch_image_tensors(&self, images: Vec<Tensor>) -> crate::Result<Mat> {
        let image_chunks = images
            .into_iter()
            .map(heatmap_tensor_to_mat)
            .collect::<crate::Result<Vec<_>>>()?;
        let mut image = Mat::default();
        let image_chunks = VectorOfMat::from_iter(image_chunks);
        core::vconcat(&image_chunks, &mut image)?;
        Ok(image)
    }

    pub fn resize_heatmap_to_image(&self, heatmap: Mat) -> crate::Result<Mat> {
        // convert image [0,1) to 255 grayscale image
        let mut gray_scale_image = Mat::default();
        heatmap.convert_to(&mut gray_scale_image, core::CV_8UC1, 255.0, 0.0)?;
        // resize image
        let mut resized_image = Mat::default();
        imgproc::resize(
            &gray_scale_image,
            &mut resized_image,
            self.original_size_with_padding,
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;
        let result = Mat::roi(
            &resized_image,
            core::Rect::new(0, 0, self.original_size.width, self.original_size.height),
        )?;
        Ok(result)
    }
}

fn heatmap_tensor_to_mat(heatmap: Tensor) -> crate::Result<Mat> {
    let (height, width) = heatmap.dims2()?;
    debug_assert_eq!(height, width, "original heatmap must be square");
    let heatmap: Vec<Vec<f32>> = heatmap.to_vec2()?;
    let mut img =
        unsafe { Mat::new_size(core::Size::new(width as i32, height as i32), core::CV_32F)? };
    for (x, row) in heatmap.iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            *(img.at_2d_mut::<f32>(x as i32, y as i32)?) = value;
        }
    }
    Ok(img)
}

/// convert an image from map to gray scale image and save it to output_path
pub fn save_image<P: AsRef<Path>>(image: &Mat, output_path: P) -> crate::Result<()> {
    imgcodecs::imwrite(
        output_path.as_ref().as_os_str().to_str().unwrap(),
        image,
        &core::Vector::new(),
    )?;
    Ok(())
}
