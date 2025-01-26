use ort::Error as OrtError;
use std::fmt;

/// Represents the different types of errors that can occur in the text classifier.
#[derive(Debug)]
pub enum ClassifierError {
    /// Error occurred while loading or using the tokenizer
    TokenizerError(String),
    /// Error occurred while loading or running the ONNX model
    ModelError(String),
    /// Error occurred during the build phase
    BuildError(String),
    /// Error occurred while making predictions
    PredictionError(String),
    /// Error occurred due to invalid input parameters
    ValidationError(String),
}

impl fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
            Self::ModelError(msg) => write!(f, "Model error: {}", msg),
            Self::BuildError(msg) => write!(f, "Build error: {}", msg),
            Self::PredictionError(msg) => write!(f, "Prediction error: {}", msg),
            Self::ValidationError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for ClassifierError {}

impl From<OrtError> for ClassifierError {
    fn from(err: OrtError) -> Self {
        ClassifierError::BuildError(err.to_string())
    }
} 