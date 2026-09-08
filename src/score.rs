//! Scoring: cosine similarity and softmax over a label set.

use serde::Serialize;
use std::collections::HashMap;

use crate::vision::cosine;

/// CLIP's learned temperature.
const LOGIT_SCALE: f32 = 100.0;

#[derive(Debug, Serialize, PartialEq)]
pub struct Score {
    pub cosine: f32,
    pub softmax: f32,
}

#[derive(Debug, Serialize)]
pub struct Ranked {
    pub scores: HashMap<String, Score>,
    pub top: String,
}

/// Rank one image embedding against labels (id -> vector).
/// Returns per-label cosine and softmax, plus the winning label id.
pub fn rank(image: &[f32], labels: &[(String, Vec<f32>)]) -> Ranked {
    let cosines: Vec<f32> = labels.iter().map(|(_, v)| cosine(image, v)).collect();
    let probs = softmax(&cosines);

    let top = cosines
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| labels[i].0.clone())
        .unwrap_or_default();

    let scores = labels
        .iter()
        .enumerate()
        .map(|(i, (id, _))| {
            (id.clone(), Score { cosine: cosines[i], softmax: probs[i] })
        })
        .collect();

    Ranked { scores, top }
}

/// Numerically stable softmax at CLIP's temperature.
fn softmax(cosines: &[f32]) -> Vec<f32> {
    if cosines.is_empty() {
        return Vec::new();
    }
    let logits: Vec<f32> = cosines.iter().map(|c| c * LOGIT_SCALE).collect();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(i: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        v[i] = 1.0;
        v
    }

    #[test]
    fn identical_vectors_score_one() {
        let a = unit(0, 4);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_score_zero() {
        assert!(cosine(&unit(0, 4), &unit(1, 4)).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[0.31, 0.24, 0.11]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_is_stable_for_large_gaps() {
        let p = softmax(&[1.0, -1.0]);
        assert!(p.iter().all(|x| x.is_finite()), "produced NaN/inf");
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[0] > p[1]);
    }

    #[test]
    fn top_is_the_highest_cosine() {
        let image = unit(0, 4);
        let labels = vec![
            ("blue".to_string(), unit(1, 4)),
            ("red".to_string(), unit(0, 4)),
            ("green".to_string(), unit(2, 4)),
        ];
        let r = rank(&image, &labels);
        assert_eq!(r.top, "red");
        assert!((r.scores["red"].cosine - 1.0).abs() < 1e-6);
        assert!(r.scores["blue"].cosine.abs() < 1e-6);
        assert!(r.scores["red"].softmax > r.scores["blue"].softmax);
    }

    #[test]
    fn every_label_appears_in_scores() {
        let labels = vec![
            ("a".to_string(), unit(0, 4)),
            ("b".to_string(), unit(1, 4)),
        ];
        let r = rank(&unit(0, 4), &labels);
        assert_eq!(r.scores.len(), 2);
        let total: f32 = r.scores.values().map(|s| s.softmax).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_labels_do_not_panic() {
        let r = rank(&unit(0, 4), &[]);
        assert!(r.scores.is_empty());
        assert_eq!(r.top, "");
    }
}
