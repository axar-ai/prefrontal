use std::collections::HashMap;

mod error;
mod embedding;
mod model;
pub mod builder;
mod utils;

pub use error::ClassifierError;
pub use model::Classifier;
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
    /// Descriptions of the classes
    pub class_descriptions: HashMap<String, String>,
    /// Size of the embedding vectors
    pub embedding_size: usize,
} 