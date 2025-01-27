use tokenizers::Tokenizer;
use ort::session::Session;
use ndarray::{Array1, Array2};
use ort::value::Tensor;
use std::convert::TryFrom;
use std::collections::HashMap;

use super::error::ClassifierError;
use super::utils::normalize_vector;

/// Provides text embedding functionality using ONNX models.
/// 
/// This trait handles the conversion of text into numerical embeddings through:
/// 1. Tokenization of input text
/// 2. Token counting and validation
/// 3. Running the ONNX model to generate embeddings
/// 4. Post-processing of embeddings (normalization)
/// 
/// The ONNX model is expected to:
/// - Accept two inputs: input_ids and attention_mask (both shape [batch_size, sequence_length])
/// - Output embeddings of shape [batch_size, sequence_length, embedding_size]
/// - Use the first token's embedding as the sequence embedding
pub(crate) trait TextEmbedding {
    /// Returns the initialized tokenizer if available
    fn tokenizer(&self) -> Option<&Tokenizer>;
    
    /// Returns the initialized ONNX session if available
    fn session(&self) -> Option<&Session>;
    
    /// Returns the maximum sequence length the model can handle
    fn max_sequence_length(&self) -> Option<usize>;
    
    /// Counts the number of tokens in the text without performing the full embedding.
    /// 
    /// This is useful for:
    /// - Checking if text needs to be chunked before processing
    /// - Validating input length without the overhead of embedding
    /// 
    /// # Errors
    /// - `TokenizerError` if the tokenizer is not initialized
    /// - `TokenizerError` if the text cannot be encoded
    fn count_tokens(&self, text: &str) -> Result<usize, ClassifierError> {
        let tokenizer = self.tokenizer()
            .ok_or_else(|| ClassifierError::TokenizerError("Tokenizer not initialized".into()))?;
            
        tokenizer.encode(text, false)
            .map_err(|e| ClassifierError::TokenizerError(e.to_string()))
            .map(|encoding| encoding.get_ids().len())
    }
    
    /// Converts text into token IDs suitable for model input.
    /// 
    /// This method:
    /// 1. Tokenizes the input text
    /// 2. Validates the token length against max_sequence_length
    /// 3. Ensures safe conversion of token IDs to u32
    /// 
    /// # Errors
    /// - `TokenizerError` if the tokenizer is not initialized
    /// - `TokenizerError` if the text cannot be encoded
    /// - `ValidationError` if the token length exceeds max_sequence_length
    /// - `ValidationError` if token length exceeds system limits
    /// - `TokenizerError` if any token ID is invalid
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, ClassifierError> {
        let tokenizer = self.tokenizer()
            .ok_or_else(|| ClassifierError::TokenizerError("Tokenizer not initialized".into()))?;
        let max_length = self.max_sequence_length()
            .ok_or_else(|| ClassifierError::TokenizerError("Max sequence length not set".into()))?;
        
        let encoding = tokenizer.encode(text, false)
            .map_err(|e| ClassifierError::TokenizerError(e.to_string()))?;
        let token_ids = encoding.get_ids();
        
        // Safe length check
        let token_len = usize::try_from(token_ids.len())
            .map_err(|_| ClassifierError::ValidationError("Token length exceeds system limits".into()))?;
            
        if token_len > max_length {
            return Err(ClassifierError::ValidationError(
                format!(
                    "Input text too long: {} tokens (max: {}). Consider splitting the text into smaller chunks.",
                    token_len, max_length
                )
            ));
        }
        
        // Safe conversion of token IDs
        let safe_tokens: Result<Vec<u32>, _> = token_ids.iter()
            .map(|&id| u32::try_from(id))
            .collect();
            
        safe_tokens.map_err(|_| ClassifierError::TokenizerError("Invalid token ID encountered".into()))
    }

    /// Converts text into a normalized embedding vector.
    /// 
    /// This is the main entry point for text embedding, which:
    /// 1. Tokenizes the input text
    /// 2. Generates embeddings using the model
    /// 3. Returns a normalized embedding vector
    /// 
    /// # Errors
    /// - Forwards all errors from `tokenize()` and `get_embedding()`
    fn embed_text(&self, text: &str) -> Result<Array1<f32>, ClassifierError> {
        let tokens = self.tokenize(text)?;
        self.get_embedding(&tokens)
    }

    /// Generates embeddings from token IDs using the ONNX model.
    /// 
    /// The process involves:
    /// 1. Creating input tensors (input_ids and attention_mask)
    /// 2. Running the ONNX model
    /// 3. Extracting and normalizing the embedding
    /// 
    /// # Model Input Format
    /// - input_ids: Token IDs [batch_size=1, sequence_length]
    /// - attention_mask: 1 for real tokens, 0 for padding [batch_size=1, sequence_length]
    /// 
    /// # Model Output Format
    /// - Shape: [batch_size=1, sequence_length, embedding_size]
    /// - Uses first token's embedding ([0,0,:]) as sequence embedding
    /// 
    /// # Errors
    /// - `ModelError` if the session is not initialized
    /// - `ModelError` if tensor creation fails
    /// - `ModelError` if model execution fails
    /// - `ModelError` if output extraction fails
    fn get_embedding(&self, tokens: &[u32]) -> Result<Array1<f32>, ClassifierError> {
        let session = self.session()
            .ok_or_else(|| ClassifierError::ModelError("Session not initialized".into()))?;

        let input_array = Array2::from_shape_vec((1, tokens.len()), 
            tokens.iter().map(|&x| x as i64).collect())
            .map_err(|e| ClassifierError::ModelError(format!("Failed to create input array: {}", e)))?;
        let input_dyn = input_array.into_dyn();
        let input_ids = input_dyn.as_standard_layout();
        
        let mask_array = Array2::from_shape_vec((1, tokens.len()),
            tokens.iter().map(|&x| if x == 0 { 0i64 } else { 1i64 }).collect())
            .map_err(|e| ClassifierError::ModelError(format!("Failed to create mask array: {}", e)))?;
        let mask_dyn = mask_array.into_dyn();
        let attention_mask = mask_dyn.as_standard_layout();
        
        let mut input_tensors = HashMap::new();
        input_tensors.insert("input_ids", Tensor::from_array(&input_ids)
            .map_err(|e| ClassifierError::ModelError(format!("Failed to create input tensor: {}", e)))?);
        input_tensors.insert("attention_mask", Tensor::from_array(&attention_mask)
            .map_err(|e| ClassifierError::ModelError(format!("Failed to create mask tensor: {}", e)))?);

        let outputs = session.run(input_tensors)
            .map_err(|e| ClassifierError::ModelError(format!("Failed to run model: {}", e)))?;
        let output_tensor = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| ClassifierError::ModelError(format!("Failed to extract output tensor: {}", e)))?;
        
        let mut embedding = Array1::zeros(output_tensor.shape()[2]);
        let embedding_slice = output_tensor.slice(ndarray::s![0, 0, ..]);
        embedding.assign(&Array1::from_iter(embedding_slice.iter().cloned()));

        Ok(normalize_vector(&embedding))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Classifier, BuiltinModel, ClassDefinition};

    fn setup_test_classifier() -> Classifier {
        Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap()
            .add_class(
                ClassDefinition::new("test", "Test class")
                    .with_examples(vec!["example text"])
            )
            .unwrap()
            .build()
            .expect("Failed to create classifier")
    }

    #[test]
    fn test_token_counting() {
        let classifier = setup_test_classifier();
        let result = classifier.count_tokens("test text");
        assert!(result.is_ok());
        assert!(result.unwrap() > 0);
    }
} 