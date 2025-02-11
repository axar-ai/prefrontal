use std::collections::HashMap;
use std::sync::Arc;

mod error;
mod embedding;
mod classifier;
pub mod builder;
mod utils;

pub use error::ClassifierError;
pub use classifier::Classifier;
pub use builder::{ClassifierBuilder, ClassDefinition};

/// Information about the current state and configuration of a classifier
#[derive(Debug, Clone)]
pub struct ClassifierInfo {
    /// Path to the ONNX model file
    pub model_path: String,
    /// Path to the tokenizer file
    pub tokenizer_path: String,
    /// Number of classes the classifier is trained on
    pub num_classes: usize,
    /// Labels of the classes
    pub class_labels: Vec<String>,
    /// Descriptions of the classes, wrapped in Arc for efficient cloning
    pub class_descriptions: Arc<HashMap<String, String>>,
    /// Size of the embedding vectors
    pub embedding_size: usize,
} 