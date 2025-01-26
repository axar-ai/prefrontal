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

mod classifier;
mod runtime;

pub use classifier::{Classifier, ClassifierBuilder, ClassifierInfo, ClassifierError};
pub use runtime::{RuntimeConfig, get_env};

/// Represents the available built-in models in the library
#[derive(Debug, Clone)]
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

/// Characteristics of a model including its capabilities and requirements
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
    /// Get the characteristics of the model
    pub fn characteristics(&self) -> ModelCharacteristics {
        match self {
            Self::MiniLM => ModelCharacteristics {
                embedding_size: 384,
                max_sequence_length: 256,
                model_size_mb: 85,
            },
        }
    }

    /// Returns the paths to the model and tokenizer files
    pub fn get_paths(&self) -> (&'static str, &'static str) {
        match self {
            BuiltinModel::MiniLM => (
                "models/onnx-minilm/model.onnx",
                "models/onnx-minilm/tokenizer.json"
            ),
        }
    }
}

pub fn init_logger() {
    env_logger::init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_classifier_thread_safety() {
        let classifier = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap()
            .add_class("test", vec!["example text"])
            .unwrap()
            .build()
            .unwrap();
        
        let classifier = Arc::new(classifier);
        let mut handles = vec![];
        
        // Spawn multiple threads that use the classifier concurrently
        for i in 0..3 {
            let classifier = Arc::clone(&classifier);
            let handle = thread::spawn(move || {
                let text = match i {
                    0 => "test example",
                    1 => "another test",
                    _ => "final test",
                };
                classifier.predict(text).unwrap()
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_classifier_clone_and_send() {
        let classifier = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap()
            .add_class("test", vec!["example text"])
            .unwrap()
            .build()
            .unwrap();
        
        // Test that classifier can be sent to another thread
        thread::spawn(move || {
            classifier.predict("test").unwrap();
        }).join().unwrap();
    }
}
