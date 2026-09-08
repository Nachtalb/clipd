//! CLIP text encoder. Session is constructed, used, and dropped by the caller —
//! never held resident.

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::vision::{l2_normalize, DIM};

/// CLIP context length.
const CTX: usize = 77;

pub struct Text {
    session: Session,
    tokenizer: Tokenizer,
}

impl Text {
    pub fn load(model: &Path, tokenizer: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| e.to_string())?;
        Ok(Self { session, tokenizer })
    }

    /// Embed several prompts in one pass. Returns L2-normalized 512-d vectors,
    /// in the same order as the input.
    pub fn embed_batch(
        &mut self,
        prompts: &[String],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let n = prompts.len();
        let mut ids = vec![0i64; n * CTX];

        for (row, prompt) in prompts.iter().enumerate() {
            let enc = self
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| e.to_string())?;
            for (col, &id) in enc.get_ids().iter().take(CTX).enumerate() {
                ids[row * CTX + col] = id as i64;
            }
        }

        let ids = Tensor::from_array(Array2::from_shape_vec((n, CTX), ids)?)?;

        let outputs = self.session.run(ort::inputs!["input_ids" => ids])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok((0..n)
            .map(|i| l2_normalize(&data[i * DIM..(i + 1) * DIM]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::preprocess;
    use crate::vision::{cosine, Vision};
    use image::{DynamicImage, Rgb, RgbImage};

    fn models() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
    }

    #[test]
    fn text_matches_the_right_image() {
        let dir = models();
        if !dir.join("text.onnx").exists() || !dir.join("vision.onnx").exists() {
            eprintln!("skipping: run scripts/fetch-models.sh first");
            return;
        }

        let t0 = std::time::Instant::now();
        let mut text = Text::load(&dir.join("text.onnx"), &dir.join("tokenizer.json"))
            .expect("load text model");
        eprintln!("text session load: {:?}", t0.elapsed());

        let vecs = text
            .embed_batch(&[
                "a photo of a red square".to_string(),
                "a photo of a blue square".to_string(),
            ])
            .unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), DIM);
        assert!((cosine(&vecs[0], &vecs[0]) - 1.0).abs() < 1e-4);

        let mut vision = Vision::load(&dir.join("vision.onnx")).unwrap();
        let red = DynamicImage::ImageRgb8(RgbImage::from_pixel(256, 256, Rgb([230, 20, 20])));
        let img = vision.embed(&preprocess(&red)).unwrap();

        let to_red = cosine(&img, &vecs[0]);
        let to_blue = cosine(&img, &vecs[1]);
        eprintln!("red image -> 'red' {to_red:.4}, 'blue' {to_blue:.4}");
        assert!(to_red > to_blue, "red image scored closer to 'blue'");
    }

    #[test]
    fn empty_batch_is_empty() {
        let dir = models();
        if !dir.join("text.onnx").exists() {
            return;
        }
        let mut text =
            Text::load(&dir.join("text.onnx"), &dir.join("tokenizer.json")).unwrap();
        assert!(text.embed_batch(&[]).unwrap().is_empty());
    }
}
