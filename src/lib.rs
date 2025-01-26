//! A thread-safe text classifier library using ONNX models for embedding and classification.
//! 
//! # Basic Usage
//! 
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use text_classifier::{Classifier, BuiltinModel};
//! 
//! let classifier = Classifier::builder()
//!     .with_model(BuiltinModel::MiniLM)?
//!     .add_class("positive", vec!["great", "awesome", "excellent"])?
//!     .add_class("negative", vec!["bad", "terrible", "awful"])?
//!     .build()?;
//! 
//! let (label, scores) = classifier.predict("This is a great movie!")?;
//! println!("Predicted class: {}", label);
//! # Ok(())
//! # }
//! ```
//! 
//! # Thread Safety
//! 
//! The classifier is thread-safe and can be shared across threads using `Arc`:
//! 
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use text_classifier::{Classifier, BuiltinModel};
//! use std::sync::Arc;
//! use std::thread;
//! 
//! let classifier = Arc::new(Classifier::builder()
//!     .with_model(BuiltinModel::MiniLM)?
//!     .add_class("example", vec!["sample text"])?
//!     .build()?);
//! 
//! let mut handles = vec![];
//! for _ in 0..3 {
//!     let classifier = Arc::clone(&classifier);
//!     handles.push(thread::spawn(move || {
//!         classifier.predict("test text").unwrap();
//!     }));
//! }
//! 
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//! # Ok(())
//! # }
//! ```

pub mod classifier;
mod runtime;

pub use classifier::{Classifier, ClassifierBuilder, ClassifierError, ClassifierInfo};
pub use runtime::{RuntimeConfig, create_session_builder};

/// Built-in models that can be used with the classifier
#[derive(Debug, Clone, Copy)]
pub enum BuiltinModel {
    /// Small and efficient model based on MiniLM architecture
    /// 
    /// Characteristics:
    /// - Embedding size: 384
    /// - Max sequence length: 256
    /// - Size: ~85MB
    /// - Good balance of speed and accuracy
    MiniLM,
}

/// Characteristics of a model, including its capabilities and resource requirements
#[derive(Debug, Clone)]
pub struct ModelCharacteristics {
    /// Size of the embedding vectors produced by the model
    pub embedding_size: usize,
    /// Maximum sequence length the model can handle
    pub max_sequence_length: usize,
    /// Approximate size of the model in memory
    pub model_size_mb: usize,
}

impl BuiltinModel {
    /// Returns the characteristics of the model
    pub fn characteristics(&self) -> ModelCharacteristics {
        match self {
            BuiltinModel::MiniLM => ModelCharacteristics {
                embedding_size: 384,
                max_sequence_length: 256,
                model_size_mb: 85,
            }
        }
    }

    /// Returns the paths to the model and tokenizer files
    pub fn get_paths(&self) -> (&'static str, &'static str) {
        match self {
            BuiltinModel::MiniLM => (
                "models/onnx-minilm/model.onnx",
                "models/onnx-minilm/tokenizer.json"
            )
        }
    }
}

pub fn init_logger() {
    env_logger::init();
}
