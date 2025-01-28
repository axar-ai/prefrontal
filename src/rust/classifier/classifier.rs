use std::sync::Arc;
use std::collections::HashMap;
use ort::session::Session;
use tokenizers::Tokenizer;
use ndarray::Array1;
use anyhow::Result;

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

    /// Predicts the class of the input text and returns class scores.
    /// 
    /// # Arguments
    /// * `text` - The text to classify
    /// 
    /// # Returns
    /// A tuple containing:
    /// * The predicted class label (String)
    /// * A HashMap of class labels to their similarity scores (0.0 to 1.0)
    /// 
    /// # Example
    /// ```rust
    /// # use prefrontal::{Classifier, BuiltinModel, ClassDefinition};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let classifier = Classifier::builder()
    /// #     .with_model(BuiltinModel::MiniLM)?
    /// #     .add_class(ClassDefinition::new("positive", "Positive")
    /// #         .with_examples(vec!["great"]))?
    /// #     .build()?;
    /// let (label, scores) = classifier.predict("This is great!")?;
    /// println!("Predicted class: {}", label);
    /// for (class, score) in scores {
    ///     println!("{}: {:.2}", class, score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
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
    use std::fs;
    use crate::{ModelManager, BuiltinModel, ClassDefinition};

    async fn setup_test_classifier() -> Result<Classifier, Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model = BuiltinModel::MiniLM;
        
        if !manager.is_model_downloaded(model) {
            manager.download_model(model).await?;
        }
        assert!(manager.is_model_downloaded(model));

        let classifier = Classifier::builder()
            .with_model(model)?
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

    #[tokio::test]
    async fn test_model_loading() -> Result<(), Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model = BuiltinModel::MiniLM;
        
        if !manager.is_model_downloaded(model) {
            manager.download_model(model).await?;
        }
        assert!(manager.is_model_downloaded(model));

        let classifier = Classifier::builder()
            .with_model(model)?
            .add_class(
                ClassDefinition::new("test", "Test class")
                    .with_examples(vec!["test example"])
            )?
            .build()?;

        assert!(classifier.predict("test input").is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_model_verification() -> Result<(), Box<dyn std::error::Error>> {
        let manager = ModelManager::new("/tmp/test-prefrontal/models").unwrap();
        let model = BuiltinModel::MiniLM;

        // Clean up any existing files
        let model_path = manager.get_model_path(model);
        let tokenizer_path = manager.get_tokenizer_path(model);
        if model_path.exists() {
            fs::remove_file(&model_path)?;
        }
        if tokenizer_path.exists() {
            fs::remove_file(&tokenizer_path)?;
        }

        // Test verification of non-existent model
        assert!(!manager.verify_model(model)?);

        // Download and verify
        manager.download_model(model).await?;
        assert!(manager.verify_model(model)?);

        // Corrupt file and verify
        fs::write(&model_path, "corrupted data")?;
        assert!(!manager.verify_model(model)?);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_download() -> Result<(), Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model = BuiltinModel::MiniLM;

        // Ensure model is downloaded
        if !manager.is_model_downloaded(model) {
            manager.download_model(model).await?;
        }

        // Test paths
        let model_path = manager.get_model_path(model);
        let tokenizer_path = manager.get_tokenizer_path(model);
        assert!(model_path.exists());
        assert!(tokenizer_path.exists());

        Ok(())
    }

    #[tokio::test]
    async fn test_model_setup() -> Result<(), Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model = BuiltinModel::MiniLM;

        // Ensure model is downloaded
        if !manager.is_model_downloaded(model) {
            manager.download_model(model).await?;
        }
        assert!(manager.is_model_downloaded(model));

        // Test model setup
        let classifier = Classifier::builder()
            .with_model(model)?
            .add_class(
                ClassDefinition::new("test", "Test class")
                    .with_examples(vec!["test example"])
            )?
            .build()?;

        assert!(classifier.predict("test input").is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_model_characteristics() -> Result<(), Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model = BuiltinModel::MiniLM;

        // Ensure model is downloaded
        if !manager.is_model_downloaded(model) {
            manager.download_model(model).await?;
        }

        // Test model characteristics
        let characteristics = model.characteristics();
        assert_eq!(characteristics.embedding_size, 384);
        assert_eq!(characteristics.max_sequence_length, 256);
        assert_eq!(characteristics.model_size_mb, 85);

        Ok(())
    }
} 