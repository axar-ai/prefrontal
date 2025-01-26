use tokenizers::Tokenizer;
use ort::Session;
use ndarray::{Array1, Array2};
use ort::tensor::OrtOwnedTensor;
use ort::Value;
use std::convert::TryFrom;

use super::error::ClassifierError;
use super::utils::normalize_vector;

/// Private trait for embedding functionality
pub(crate) trait TextEmbedding {
    fn tokenizer(&self) -> Option<&Tokenizer>;
    fn session(&self) -> Option<&Session>;
    fn max_sequence_length(&self) -> Option<usize>;
    
    /// Returns the number of tokens in the text without actually performing the embedding
    fn count_tokens(&self, text: &str) -> Result<usize, ClassifierError> {
        let tokenizer = self.tokenizer()
            .ok_or_else(|| ClassifierError::TokenizerError("Tokenizer not initialized".into()))?;
            
        tokenizer.encode(text, false)
            .map_err(|e| ClassifierError::TokenizerError(e.to_string()))
            .map(|encoding| encoding.get_ids().len())
    }
    
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

    fn embed_text(&self, text: &str) -> Result<Array1<f32>, ClassifierError> {
        let tokens = self.tokenize(text)?;
        self.get_embedding(&tokens)
    }

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
        
        let input_tensors = vec![
            Value::from_array(session.allocator(), &input_ids)
                .map_err(|e| ClassifierError::ModelError(format!("Failed to create input tensor: {}", e)))?,
            Value::from_array(session.allocator(), &attention_mask)
                .map_err(|e| ClassifierError::ModelError(format!("Failed to create mask tensor: {}", e)))?,
        ];

        let outputs = session.run(input_tensors)
            .map_err(|e| ClassifierError::ModelError(format!("Failed to run model: {}", e)))?;
        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract()
            .map_err(|e| ClassifierError::ModelError(format!("Failed to extract output tensor: {}", e)))?;
        let array = output_tensor.view();
        
        let mut embedding = Array1::zeros(array.shape()[2]);
        let embedding_slice = array.slice(ndarray::s![0, 0, ..]);
        embedding.assign(&Array1::from_iter(embedding_slice.iter().cloned()));

        Ok(normalize_vector(&embedding))
    }
} 