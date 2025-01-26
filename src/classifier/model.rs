use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use ort::Session;
use ndarray::Array1;

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
/// use text_classifier::{Classifier, BuiltinModel, ClassDefinition};
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
/// use text_classifier::{Classifier, BuiltinModel, ClassDefinition};
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
        Some(&*self.tokenizer)
    }
    
    fn session(&self) -> Option<&Session> {
        Some(&*self.session)
    }

    fn max_sequence_length(&self) -> Option<usize> {
        Some(self.model_characteristics.max_sequence_length)
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
            class_descriptions: self.class_descriptions.as_ref().clone(),
            embedding_size: self.model_characteristics.embedding_size,
        }
    }

    /// Returns the number of tokens in the input text
    /// This is useful to check if text will fit within the model's max_sequence_length
    /// before attempting classification
    pub fn count_tokens(&self, text: &str) -> Result<usize, ClassifierError> {
        TextEmbedding::count_tokens(self, text)
    }

    /// Makes a prediction for the given text
    pub fn predict(&self, text: &str) -> Result<(String, HashMap<String, f32>), ClassifierError> {
        if text.is_empty() {
            return Err(ClassifierError::ValidationError("Input text cannot be empty".into()));
        }
        
        // First check token count to provide a more specific error
        let token_count = self.count_tokens(text)?;
            
        if token_count > self.model_characteristics.max_sequence_length {
            return Err(ClassifierError::ValidationError(
                format!(
                    "Input text is too long ({} tokens, max is {}). Consider splitting the text into smaller chunks.",
                    token_count,
                    self.model_characteristics.max_sequence_length
                )
            ));
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