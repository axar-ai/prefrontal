pub mod classifier;

pub use classifier::Classifier;
pub use classifier::ClassifierError;

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

    /// Get the paths to the model and tokenizer files
    pub(crate) fn get_paths(&self) -> (&'static str, &'static str) {
        match self {
            Self::MiniLM => (
                "models/onnx-minilm/model.onnx",
                "models/onnx-minilm/tokenizer.json",
            ),
        }
    }
}

pub fn init_logger() {
    env_logger::init();
}
