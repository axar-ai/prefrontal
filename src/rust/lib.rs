//! A thread-safe text classifier library using ONNX models for embedding and classification.
//! 
//! # Basic Usage
//! 
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use prefrontal::{Classifier, BuiltinModel, ClassDefinition};
//! 
//! let classifier = Classifier::builder()
//!     .with_model(BuiltinModel::MiniLM)?
//!     .add_class(
//!         ClassDefinition::new("positive", "Content with positive sentiment")
//!             .with_examples(vec!["great", "awesome", "excellent"])
//!     )?
//!     .add_class(
//!         ClassDefinition::new("negative", "Content with negative sentiment")
//!             .with_examples(vec!["bad", "terrible", "awful"])
//!     )?
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
//! use prefrontal::{Classifier, BuiltinModel, ClassDefinition};
//! use std::sync::Arc;
//! use std::thread;
//! 
//! let classifier = Arc::new(Classifier::builder()
//!     .with_model(BuiltinModel::MiniLM)?
//!     .add_class(
//!         ClassDefinition::new("example", "Example class")
//!             .with_examples(vec!["sample text"])
//!     )?
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
pub mod model_manager;
pub mod models;

pub use classifier::{Classifier, ClassifierBuilder, ClassifierError, ClassifierInfo, ClassDefinition};
pub use runtime::{RuntimeConfig, create_session_builder};
pub use model_manager::{ModelManager, ModelError};
pub use models::{BuiltinModel, ModelCharacteristics, ModelInfo};

pub fn init_logger() {
    env_logger::init();
}
