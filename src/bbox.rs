use log::debug;
use opencv::core::{
    self, max_mat_f64, min_mat_f64, Mat, Point, Point2f, Rect, Scalar, Size, Vector, CV_32S,
};
use opencv::prelude::*;
use opencv::types::VectorOfi32;
use opencv::{imgcodecs, imgproc};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct BBox {
    pub polygon: [Point2f; 4],
}

impl BBox {
    fn rescale(&self, heatmap_size: Size, image_with_padding_size: Size) -> crate::Result<Self> {
        let (h_scaler, w_scaler) = (
            image_with_padding_size.height as f32 / heatmap_size.height as f32,
            image_with_padding_size.width as f32 / heatmap_size.width as f32,
        );
        let mut polygon = [Point2f::default(); 4];
        for (i, point) in self.polygon.iter().enumerate() {
            let x = point.x * w_scaler;
            let y = point.y * h_scaler;
            polygon[i] = Point2f::new(x, y);
        }
        Ok(Self { polygon })
    }

    fn draw_on_image(&self, image: &mut Mat) -> crate::Result<()> {
        let points: Vector<Point> = self
            .polygon
            .iter()
            .map(|point| {
                let x = point.x as i32;
                let y = point.y as i32;
                Point::new(x, y)
            })
            .collect();
        imgproc::polylines(
            image,
            &points,
            true,
            Scalar::new(0., 0., 255., 0.),
            1,
            opencv::imgproc::LINE_8,
            0,
        )?;
        Ok(())
    }
}

/// https://docs.rs/opencv/0.88.8/opencv/imgproc/fn.threshold.html
fn image_threshold(mat: &Mat, non_max_suppression_threshold: f64) -> crate::Result<Mat> {
    let mut r = Mat::default();
    let max_val = 1.0;
    imgproc::threshold(
        &mat,
        &mut r,
        non_max_suppression_threshold,
        max_val,
        imgproc::THRESH_BINARY,
    )?;
    let r = min_mat_f64(&r, 1.0)?.to_mat()?;
    let r = max_mat_f64(&r, 0.0)?.to_mat()?;
    Ok(r)
}

/// https://docs.rs/opencv/0.88.8/opencv/prelude/trait.MatTraitConst.html#method.convert_to
fn image_f32_to_u8(mat: Mat) -> crate::Result<Mat> {
    let mut r = Mat::default();
    let alpha = 255.0;
    let beta = 0.0;
    mat.convert_to(&mut r, core::CV_8UC1, alpha, beta)?;
    Ok(r)
}

/// https://docs.rs/opencv/0.88.8/opencv/imgproc/fn.connected_components.html
fn image_to_connected_components(mat: Mat) -> crate::Result<(Mat, Mat, Mat)> {
    let mut labels: Mat = Default::default();
    let mut stats: Mat = Default::default();
    let mut centroids: Mat = Default::default();
    imgproc::connected_components_with_stats(
        &mat,
        &mut labels,
        &mut stats,
        &mut centroids,
        4,
        CV_32S,
    )?;
    Ok((labels, stats, centroids))
}

fn heatmap_label_max(heatmap: &Mat, labels: &Mat, label: i32) -> crate::Result<f64> {
    let mut mask = Mat::default();
    core::compare(labels, &(label as f64), &mut mask, opencv::core::CMP_EQ)?;
    let mut max_value = 0.0;
    core::min_max_loc(heatmap, None, Some(&mut max_value), None, None, &mask)?;
    Ok(max_value)
}

fn get_dilation_matrix(segmap: &mut Mat, stats_row: &[i32]) -> crate::Result<Mat> {
    let (x, y, w, h, area) = (
        stats_row[imgproc::CC_STAT_LEFT as usize],
        stats_row[imgproc::CC_STAT_TOP as usize],
        stats_row[imgproc::CC_STAT_WIDTH as usize],
        stats_row[imgproc::CC_STAT_HEIGHT as usize],
        stats_row[imgproc::CC_STAT_AREA as usize],
    );
    let niter = {
        let niter = (area * w.min(h)) as f64 / (w * h) as f64;
        (niter.sqrt() * 2.0) as i32
    };
    let roi = {
        let sx = (x - niter).max(0);
        let sy = (y - niter).max(0);
        let ex = (x + w + niter + 1).min(segmap.cols());
        let ey = (y + h + niter + 1).min(segmap.rows());
        Rect::new(sx, sy, ex - sx, ey - sy)
    };
    let mut roi = Mat::roi(segmap, roi)?;
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(1 + niter, 1 + niter),
        Point::new(-1, -1),
    )?;
    imgproc::dilate(
        segmap,
        &mut roi,
        &kernel,
        Point::new(-1, -1), // default anchor
        1,
        core::BORDER_CONSTANT, // border type
        Scalar::default(),     // border value
    )?;
    Ok(roi)
}

fn connected_area_to_bbox(
    labels: &Mat,
    stats_row: &[i32],
    label: i32,
) -> crate::Result<[Point2f; 4]> {
    let mut segmap = Mat::default();
    core::compare(&labels, &(label as f64), &mut segmap, opencv::core::CMP_EQ)?;

    let dilated_roi = get_dilation_matrix(&mut segmap, stats_row)?;
    dilated_roi.copy_to(&mut segmap)?;

    let mut non_zero = Mat::default();
    core::find_non_zero(&segmap, &mut non_zero)?;
    let rotated_rect = imgproc::min_area_rect(&non_zero)?;
    let mut points = [Point2f::default(); 4];
    rotated_rect.points(&mut points)?;
    Ok(points)
}

pub fn draw_bboxes<P: AsRef<Path>>(
    image: &mut Mat,
    heatmap_size: Size,
    image_with_padding_size: Size,
    bboxes: &[BBox],
    output_file: P,
) -> crate::Result<()> {
    debug!(
        "image size={:?}, heatmap_size={:?}, image_with_padding_size={:?}",
        image.size()?,
        heatmap_size,
        image_with_padding_size
    );
    for bbox in bboxes {
        bbox.rescale(heatmap_size, image_with_padding_size)?
            .draw_on_image(image)?;
    }
    let params = VectorOfi32::new();
    imgcodecs::imwrite(
        output_file.as_ref().as_os_str().to_str().unwrap(),
        image,
        &params,
    )?;
    Ok(())
}

/// generate bbox from heatmap which are rescaled to original size
pub fn generate_bbox(
    heatmap: &Mat,
    non_max_suppression_threshold: f64,
    extract_text_threshold: f64,
    bbox_area_threshold: i32,
) -> crate::Result<Vec<BBox>> {
    let labels = image_threshold(heatmap, non_max_suppression_threshold)?;
    let labels = image_f32_to_u8(labels)?;
    let (labels, stats, centroids) = image_to_connected_components(labels)?;
    debug!("labels {:?}", labels);
    debug!("stats {:?}", stats);
    debug!("centroids {:?}", centroids);

    debug_assert_eq!(
        centroids.rows(),
        stats.rows(),
        "centroids and stats rows must be equal"
    );
    debug_assert_eq!(5, stats.cols(), "stats must have 5 columns");
    debug_assert_eq!(2, centroids.cols(), "centroids must have 2 columns");

    let mut bboxes = Vec::new();
    // 0 is background so skip it
    for label in 1..stats.rows() {
        let stats_row = stats.at_row::<i32>(label)?;
        let area = stats_row[imgproc::CC_STAT_AREA as usize];
        if area < bbox_area_threshold {
            continue;
        }
        let max_value = heatmap_label_max(heatmap, &labels, label)?;
        if max_value < extract_text_threshold {
            continue;
        }
        let polygon = connected_area_to_bbox(&labels, stats_row, label)?;
        bboxes.push(BBox { polygon });
    }
    debug!(
        "bbox filtering, before={}, after={} bboxes",
        stats.rows(),
        bboxes.len()
    );
    Ok(bboxes)
}
