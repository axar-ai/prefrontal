use std::error::Error;
use std::fmt;
use crate::ModelError;

/// Represents the different types of errors that can occur in the text classifier.
#[derive(Debug)]
pub enum ClassifierError {
    /// Error during model loading or inference
    ModelError(String),
    /// Error during tokenization
    TokenizerError(String),
    /// Error during classifier construction
    BuildError(String),
    /// Error during input validation
    ValidationError(String),
    /// Error during model management
    ModelManagementError(ModelError),
}

impl fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassifierError::ModelError(msg) => write!(f, "Model error: {}", msg),
            ClassifierError::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
            ClassifierError::BuildError(msg) => write!(f, "Build error: {}", msg),
            ClassifierError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            ClassifierError::ModelManagementError(e) => write!(f, "Model management error: {}", e),
        }
    }
}

impl Error for ClassifierError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ClassifierError::ModelManagementError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ModelError> for ClassifierError {
    fn from(error: ModelError) -> Self {
        ClassifierError::ModelManagementError(error)
    }
}

impl From<ort::Error> for ClassifierError {
    fn from(error: ort::Error) -> Self {
        ClassifierError::ModelError(error.to_string())
    }
}

impl From<tokenizers::Error> for ClassifierError {
    fn from(error: tokenizers::Error) -> Self {
        ClassifierError::TokenizerError(error.to_string())
    }
} 