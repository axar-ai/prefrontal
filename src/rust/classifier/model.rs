use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use ort::session::Session;
use ndarray::Array1;
use tokenizers::Tokenizer;
use super::error::ClassifierError;
use super::embedding::TextEmbedding;
use crate::ModelCharacteristics;

/// A thread-safe text classifier using ONNX models for embedding and classification.
/// 
/// # Thread Safety
/// 
/// This type is automatically `Send + Sync` because all of its fields are thread-safe:
/// - `String` and `ModelCharacteristics` are `Send + Sync`
/// - `Arc<T>` provides thread-safe shared ownership
/// - `Tokenizer`, `Session`, and `HashMap` are wrapped in `Arc`
/// 
/// Single-thread usage:
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use prefrontal::{Classifier, BuiltinModel, ClassDefinition};
/// 
/// let classifier = Classifier::builder()
///     .with_model(BuiltinModel::MiniLM)?
///     .add_class(
///         ClassDefinition::new("example", "Example class")
///             .with_examples(vec!["sample text"])
///     )?
///     .build()?;
/// 
/// classifier.predict("test text")?;
/// # Ok(())
/// # }
/// ```
/// 
/// Multi-thread usage:
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use prefrontal::{Classifier, BuiltinModel, ClassDefinition};
/// use std::sync::Arc;
/// use std::thread;
/// 
/// let classifier = Arc::new(Classifier::builder()
///     .with_model(BuiltinModel::MiniLM)?
///     .add_class(
///         ClassDefinition::new("example", "Example class")
///             .with_examples(vec!["sample text"])
///     )?
///     .build()?);
/// 
/// let classifier_clone = Arc::clone(&classifier);
/// thread::spawn(move || {
///     classifier_clone.predict("test text").unwrap();
/// });
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Classifier {
    pub model_path: String,
    pub tokenizer_path: String,
    pub tokenizer: Arc<Tokenizer>,
    pub session: Arc<Session>,
    pub embedded_prototypes: Arc<HashMap<String, Array1<f32>>>,
    pub class_descriptions: Arc<HashMap<String, String>>,
    pub model_characteristics: ModelCharacteristics,
}

// Compile-time verification of thread-safety
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn verify_thread_safety() {
        assert_send_sync::<Classifier>();
    }
};

impl TextEmbedding for Classifier {
    fn tokenizer(&self) -> Option<&Tokenizer> {
        Some(&self.tokenizer)
    }
    
    fn session(&self) -> Option<&Session> {
        Some(&self.session)
    }
}

impl Classifier {
    /// Creates a new ClassifierBuilder for fluent construction
    pub fn builder() -> super::builder::ClassifierBuilder {
        super::builder::ClassifierBuilder::new()
    }

    /// Returns information about the classifier's current state
    pub fn info(&self) -> super::ClassifierInfo {
        super::ClassifierInfo {
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            num_classes: self.embedded_prototypes.len(),
            class_labels: self.embedded_prototypes.keys().cloned().collect(),
            class_descriptions: Arc::clone(&self.class_descriptions),
            embedding_size: self.model_characteristics.embedding_size,
        }
    }

    /// Makes a prediction for the given text
    pub fn predict(&self, text: &str) -> Result<(String, HashMap<String, f32>), ClassifierError> {
        if text.is_empty() {
            return Err(ClassifierError::ValidationError("Input text cannot be empty".into()));
        }
        
        let input_vector = self.embed_text(text)?;
        
        let mut scores = HashMap::new();
        for (label, prototype) in self.embedded_prototypes.as_ref() {
            let similarity = Self::cosine_similarity(&input_vector, prototype);
            scores.insert(label.clone(), similarity);
        }
        
        let best_class = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(class, _)| class.clone())
            .unwrap_or_else(|| "unknown".to_string());
        
        Ok((best_class, scores))
    }

    fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.dot(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelManager, BuiltinModel, ClassDefinition};

    async fn setup_test_classifier() -> Result<Classifier, Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model_info = BuiltinModel::MiniLM.get_model_info();
        manager.ensure_model_downloaded(&model_info).await?;

        let classifier = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)?
            .add_class(
                ClassDefinition::new("test", "Test class")
                    .with_examples(vec!["example text"])
            )?
            .build()?;

        Ok(classifier)
    }

    #[tokio::test]
    async fn test_class_info() -> Result<(), Box<dyn std::error::Error>> {
        let classifier = setup_test_classifier().await?;
        let info = classifier.info();
        assert_eq!(info.num_classes, 1);
        assert!(info.class_descriptions.contains_key("test"));
        Ok(())
    }
} 