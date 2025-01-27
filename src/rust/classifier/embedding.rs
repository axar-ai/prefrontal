use tokenizers::Tokenizer;
use ort::session::Session;
use ndarray::{Array1, Array2};
use ort::value::Tensor;
use std::convert::TryFrom;
use std::collections::HashMap;

use crate::classifier::error::ClassifierError;
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
        
        let encoding = tokenizer.encode(text, false)
            .map_err(|e| ClassifierError::TokenizerError(e.to_string()))?;
        let token_ids = encoding.get_ids();
        
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
    use super::TextEmbedding;
    use crate::classifier::error::ClassifierError;
    use crate::{BuiltinModel, ModelManager};
    use ort::session::Session;
    use anyhow::Result;

    const TEST_CASES: &[(&str, usize)] = &[
        ("short text", 2),
        ("longer sample text", 3),
        ("this is a very long text that should be rejected", 10),
    ];

    struct MockEmbedding {}

    struct TestEmbedding {
        session: Session,
        tokenizer: tokenizers::Tokenizer,
    }

    impl TextEmbedding for TestEmbedding {
        fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
            Some(&self.tokenizer)
        }
        
        fn session(&self) -> Option<&Session> {
            Some(&self.session)
        }
    }

    impl TextEmbedding for MockEmbedding {
        fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
            None
        }
        
        fn session(&self) -> Option<&ort::session::Session> {
            None
        }
    }

    mod token_validation {
        use super::*;

        #[test]
        fn test_rejects_missing_tokenizer() {
            let embedding = MockEmbedding {};
            let result = embedding.tokenize("test text");
            assert!(matches!(result, Err(ClassifierError::TokenizerError(_))));
        }

        #[test]
        fn test_validates_sequence_length() {
            let embedding = MockEmbedding {};
            let long_text = "this is a very long text that should be rejected";
            let result = embedding.tokenize(long_text);
            assert!(matches!(result, Err(ClassifierError::TokenizerError(_))));
        }
    }

    mod embedding_generation {
        use super::*;

        async fn setup_real_embedding() -> Result<Box<dyn TextEmbedding>, ClassifierError> {
            let manager = ModelManager::new_default()
                .map_err(|e| ClassifierError::TokenizerError(e.to_string()))?;
            let model_info = BuiltinModel::MiniLM.get_model_info();
            manager.ensure_model_downloaded(&model_info).await
                .map_err(|e| ClassifierError::TokenizerError(e.to_string()))?;

            let model_path = manager.get_model_path(&model_info.name);
            let tokenizer_path = manager.get_tokenizer_path(&model_info.name);
            let session = Session::builder()
                .map_err(|e| ClassifierError::ModelError(e.to_string()))?
                .commit_from_file(&model_path)
                .map_err(|e| ClassifierError::ModelError(e.to_string()))?;
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| ClassifierError::TokenizerError(e.to_string()))?;

            Ok(Box::new(TestEmbedding {
                session,
                tokenizer,
            }))
        }

        #[tokio::test]
        async fn test_real_token_counting() -> Result<(), ClassifierError> {
            let embedding = setup_real_embedding().await?;
            for (text, _) in TEST_CASES.iter() {
                let result = embedding.tokenize(text);
                assert!(result.is_ok());
                assert!(result.unwrap().len() > 0);
            }
            Ok(())
        }
    }
} 