use ndarray::Array1;

pub(crate) fn normalize_vector(vec: &Array1<f32>) -> Array1<f32> {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        vec / norm
    } else {
        Array1::zeros(vec.len())
    }
}

pub(crate) fn average_vectors(vectors: &[Array1<f32>], embedding_size: usize) -> Array1<f32> {
    if vectors.is_empty() {
        return Array1::zeros(embedding_size);
    }
    let sum = vectors.iter().fold(Array1::zeros(vectors[0].len()), |acc, v| acc + v);
    sum / vectors.len() as f32
} 