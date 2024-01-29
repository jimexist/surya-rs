use log::{debug, info};
use opencv::core::{
    max_mat_f64, min_mat_f64, Mat, Point, Point2f, Rect, RotatedRect, Scalar, Size,
};
use opencv::prelude::*;
use opencv::{core, imgproc};

#[derive(Debug, Clone)]
pub struct BBox {
    pub rect: RotatedRect,
}

impl BBox {
    fn scale_to_rect(&self, original_size: (u32, u32), heatmap: &Mat) -> opencv::Result<Rect> {
        let (width, height) = original_size;
        let w_scaler = width as f32 / heatmap.cols() as f32;
        let h_scaler = height as f32 / heatmap.rows() as f32;
        let mut point_2fs = [Point2f::default(); 4];
        self.rect.points(&mut point_2fs)?;
        let mut points = [Point::default(); 4];
        for (i, point_2f) in point_2fs.iter().enumerate() {
            points[i] = Point::new(
                (point_2f.x * w_scaler) as i32,
                (point_2f.y * h_scaler) as i32,
            );
        }
        Ok(Rect::from_points(points[0], points[2]))
    }
}

/// https://docs.rs/opencv/0.88.8/opencv/imgproc/fn.threshold.html
fn image_threshold(mat: Mat, non_max_suppression_threshold: f64) -> opencv::Result<Mat> {
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
fn image_f32_to_u8(mat: Mat) -> opencv::Result<Mat> {
    let mut r = Mat::default();
    let alpha = 255.0;
    let beta = 0.0;
    mat.convert_to(&mut r, opencv::core::CV_8U, alpha, beta)?;
    Ok(r)
}

/// https://docs.rs/opencv/0.88.8/opencv/imgproc/fn.connected_components.html
fn image_to_connected_components(mat: Mat) -> opencv::Result<(Mat, Mat, Mat)> {
    let mut labels: Mat = Default::default();
    let mut stats: Mat = Default::default();
    let mut centroids: Mat = Default::default();
    imgproc::connected_components_with_stats(
        &mat,
        &mut labels,
        &mut stats,
        &mut centroids,
        4,
        core::CV_32S,
    )?;
    Ok((labels, stats, centroids))
}

fn heatmap_label_max(heatmap: &Mat, labels: &Mat, label: i32) -> opencv::Result<f64> {
    let mut mask = Mat::default();
    core::compare(labels, &(label as f64), &mut mask, opencv::core::CMP_EQ)?;
    let mut max_value = 0.0;
    core::min_max_loc(heatmap, None, Some(&mut max_value), None, None, &mask)?;
    Ok(max_value)
}

fn connected_area_to_bbox(
    labels: &Mat,
    stats_row: &[i32],
    label: i32,
) -> opencv::Result<RotatedRect> {
    let (w, h, area) = (
        // stats_row[imgproc::CC_STAT_LEFT as usize],
        // stats_row[imgproc::CC_STAT_TOP as usize],
        stats_row[imgproc::CC_STAT_WIDTH as usize],
        stats_row[imgproc::CC_STAT_HEIGHT as usize],
        stats_row[imgproc::CC_STAT_AREA as usize],
    );

    let mut segmap = Mat::default();
    core::compare(&labels, &(label as f64), &mut segmap, opencv::core::CMP_EQ)?;
    let niter = (f64::sqrt((area * i32::min(w, h)) as f64 / (w * h) as f64) * 2.0) as i32;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(1 + niter, 1 + niter),
        Point::new(-1, -1),
    )?;

    let mut dilated: Mat = Mat::default();
    imgproc::dilate(
        &segmap,
        &mut dilated,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    let mut non_zero = Mat::default();
    core::find_non_zero(&dilated, &mut non_zero)?;
    imgproc::min_area_rect(&non_zero)
}

pub fn generate_bbox(
    original_size: (u32, u32),
    heatmap: Vec<Vec<f32>>,
    non_max_suppression_threshold: f64,
    text_threshold: f64,
    bbox_area_threshold: i32,
) -> opencv::Result<Vec<Rect>> {
    let heatmap = Mat::from_slice_2d(&heatmap)?;
    let labels = image_threshold(heatmap.clone(), non_max_suppression_threshold)?;
    let labels = image_f32_to_u8(labels)?;
    let (labels, stats, centroids) = image_to_connected_components(labels)?;
    debug!("labels {:?}", labels);
    debug!("stats {:?}", stats);
    debug!("centroids {:?}", centroids);

    assert_eq!(
        centroids.rows(),
        stats.rows(),
        "centroids and stats rows must be equal"
    );
    assert_eq!(5, stats.cols(), "stats must have 5 columns");
    assert_eq!(2, centroids.cols(), "centroids must have 2 columns");

    let mut bboxes = Vec::new();
    for i in 1..stats.rows() {
        let stats_row = stats.at_row::<i32>(i)?;
        let area = stats_row[opencv::imgproc::CC_STAT_AREA as usize];
        if area < bbox_area_threshold {
            continue;
        }
        let max_value = heatmap_label_max(&heatmap, &labels, i)?;
        if max_value < text_threshold {
            continue;
        }
        let rect = connected_area_to_bbox(&labels, stats_row, i)?;
        bboxes.push(BBox { rect }.scale_to_rect(original_size, &heatmap)?);
    }
    Ok(bboxes)
}
