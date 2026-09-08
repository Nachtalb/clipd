//! CLIP vision encoder: preprocessed pixels -> L2-normalized 512-d embedding.

use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

use crate::preprocess::SIZE;

pub const DIM: usize = 512;

pub struct Vision {
    session: Session,
}

impl Vision {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(path)?;
        Ok(Self { session })
    }

    /// `pixels` is a flat NCHW buffer from `preprocess` (1x3x224x224).
    pub fn embed(&mut self, pixels: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let s = SIZE as usize;
        let arr = Array4::from_shape_vec((1, 3, s, s), pixels.to_vec())?;
        let input = Tensor::from_array(arr)?;

        let outputs = self.session.run(ort::inputs!["pixel_values" => input])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(l2_normalize(&data[..DIM]))
    }
}

pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::preprocess;
    use image::{DynamicImage, Rgb, RgbImage};

    fn model_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models/vision.onnx")
    }

    fn solid(rgb: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(256, 256, Rgb(rgb)))
    }

    #[test]
    fn normalize_gives_unit_length() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn embeddings_are_deterministic_and_discriminative() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: run scripts/fetch-models.sh first");
            return;
        }
        let mut v = Vision::load(&path).expect("load vision model");

        let red = preprocess(&solid([220, 30, 30]));
        let blue = preprocess(&solid([30, 30, 220]));

        let a = v.embed(&red).unwrap();
        let b = v.embed(&red).unwrap();
        let c = v.embed(&blue).unwrap();

        assert_eq!(a.len(), DIM);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-4, "not unit length");
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-5, "same image differs");
        assert!(cosine(&a, &c) < 0.99, "different images too close");
    }
}
