//! Image preprocessing for CLIP: resize, center crop, normalize to NCHW f32.

use image::{imageops::FilterType, DynamicImage};

pub const SIZE: u32 = 224;

/// CLIP ViT-B/32 normalization constants (OpenAI).
const MEAN: [f32; 3] = [0.481_454_67, 0.457_827_5, 0.408_210_72];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Resize shortest side to 224 (bicubic), center crop 224x224, normalize.
/// Returns a flat NCHW buffer of length 1*3*224*224.
pub fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let scale = SIZE as f32 / w.min(h) as f32;
    let (nw, nh) = (
        (w as f32 * scale).round().max(SIZE as f32) as u32,
        (h as f32 * scale).round().max(SIZE as f32) as u32,
    );

    let resized = img.resize_exact(nw, nh, FilterType::CatmullRom);
    let (x, y) = ((nw - SIZE) / 2, (nh - SIZE) / 2);
    let rgb = resized.crop_imm(x, y, SIZE, SIZE).to_rgb8();

    let n = (SIZE * SIZE) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb(rgb)))
    }

    #[test]
    fn shape_is_nchw_224() {
        let out = preprocess(&solid(256, 256, [255, 0, 0]));
        assert_eq!(out.len(), 3 * 224 * 224);
    }

    #[test]
    fn solid_red_matches_hand_computed_normalization() {
        let out = preprocess(&solid(256, 256, [255, 0, 0]));
        let n = (SIZE * SIZE) as usize;
        let expect = [
            (1.0 - MEAN[0]) / STD[0],
            (0.0 - MEAN[1]) / STD[1],
            (0.0 - MEAN[2]) / STD[2],
        ];
        for c in 0..3 {
            let mean: f32 = out[c * n..(c + 1) * n].iter().sum::<f32>() / n as f32;
            assert!(
                (mean - expect[c]).abs() < 1e-3,
                "channel {c}: got {mean}, want {}",
                expect[c]
            );
        }
    }

    #[test]
    fn non_square_input_still_yields_224_square() {
        for (w, h) in [(640, 480), (100, 900), (224, 224), (50, 50)] {
            let out = preprocess(&solid(w, h, [10, 200, 30]));
            assert_eq!(out.len(), 3 * 224 * 224, "failed for {w}x{h}");
        }
    }

    #[test]
    fn center_crop_keeps_center_content() {
        // Left half black, right half white; after center crop the mean should
        // sit between the two extremes rather than at either end.
        let mut img = RgbImage::new(448, 224);
        for (x, _y, px) in img.enumerate_pixels_mut() {
            *px = if x < 224 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            };
        }
        let out = preprocess(&DynamicImage::ImageRgb8(img));
        let n = (SIZE * SIZE) as usize;
        let mean: f32 = out[..n].iter().sum::<f32>() / n as f32;
        let lo = (0.0 - MEAN[0]) / STD[0];
        let hi = (1.0 - MEAN[0]) / STD[0];
        assert!(
            mean > lo + 0.1 && mean < hi - 0.1,
            "mean {mean} not centered"
        );
    }
}
