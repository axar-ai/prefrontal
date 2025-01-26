use std::collections::HashMap;
use tokenizers::Tokenizer;
use ort::{Environment, Session, SessionBuilder, Value, tensor::OrtOwnedTensor, OrtError};
use ndarray::{Array1, Array2};
use log::{info, error};
use std::sync::Arc;
use crate::ClassifierError;
use crate::BuiltinModel;
use crate::ModelCharacteristics;
use std::convert::TryFrom;

/// A builder for constructing a Classifier with a fluent interface.
#[derive(Default, Debug)]
pub struct ClassifierBuilder {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    class_examples: HashMap<String, Vec<String>>,
    model_characteristics: Option<ModelCharacteristics>,
}

/// Private trait for embedding functionality
trait TextEmbedding {
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

        Ok(Classifier::normalize_vector(&embedding))
    }
}

impl TextEmbedding for ClassifierBuilder {
    fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
    
    fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }

    fn max_sequence_length(&self) -> Option<usize> {
        self.model_characteristics.as_ref().map(|c| c.max_sequence_length)
    }
}

impl TextEmbedding for Classifier {
    fn tokenizer(&self) -> Option<&Tokenizer> {
        Some(&*self.tokenizer)
    }
    
    fn session(&self) -> Option<&Session> {
        Some(&*self.session)
    }

    fn max_sequence_length(&self) -> Option<usize> {
        Some(self.model_characteristics.max_sequence_length)
    }
}

impl From<OrtError> for ClassifierError {
    fn from(err: OrtError) -> Self {
        ClassifierError::BuildError(err.to_string())
    }
}

impl ClassifierBuilder {
    /// Creates a new ClassifierBuilder
    pub fn new() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            tokenizer: None,
            session: None,
            class_examples: HashMap::new(),
            model_characteristics: None,
        }
    }

    /// Validates that the model has the expected input/output structure
    fn validate_model(session: &Session) -> Result<(), ClassifierError> {
        // Check inputs
        let inputs = &session.inputs;
        if inputs.len() < 2 {
            return Err(ClassifierError::ModelError(
                format!("Model must have at least 2 inputs (input_ids and attention_mask), found {}", inputs.len())
            ));
        }

        // Check outputs
        let outputs = &session.outputs;
        if outputs.is_empty() {
            return Err(ClassifierError::ModelError(
                "Model must have at least 1 output for embeddings".to_string()
            ));
        }

        Ok(())
    }

    /// Sets the model to use for classification
    pub fn with_model(mut self, model: BuiltinModel) -> Result<Self, ClassifierError> {
        if self.model_path.is_some() || self.tokenizer_path.is_some() {
            return Err(ClassifierError::BuildError("Model and tokenizer paths already set".to_string()));
        }
        let (model_path, tokenizer_path) = model.get_paths();
        
        // Store model characteristics
        self.model_characteristics = Some(model.characteristics());
        
        // Validate paths exist
        if !std::path::Path::new(model_path).exists() {
            return Err(ClassifierError::BuildError(format!("Model file not found: {}", model_path)));
        }
        if !std::path::Path::new(tokenizer_path).exists() {
            return Err(ClassifierError::BuildError(format!("Tokenizer file not found: {}", tokenizer_path)));
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| {
                error!("Failed to load tokenizer: {}", e);
                ClassifierError::BuildError(format!("Failed to load tokenizer: {}", e))
            })?;
        info!("Tokenizer loaded successfully");

        // Initialize ONNX Runtime and load model
        let env = Arc::new(Environment::builder()
            .with_name("text_classifier")
            .build()
            .map_err(|e| ClassifierError::BuildError(format!("Failed to create environment: {}", e)))?);
        
        let session = SessionBuilder::new(&env)?
            .with_model_from_file(model_path)?;

        // Validate model structure
        Self::validate_model(&session)?;
        info!("Model structure validated successfully");
        
        self.model_path = Some(model_path.to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string());
        self.tokenizer = Some(tokenizer);
        self.session = Some(session);
        Ok(self)
    }

    /// Sets a custom model and tokenizer path with configurable sequence length
    pub fn with_custom_model(
        mut self,
        model_path: &str,
        tokenizer_path: &str,
        max_sequence_length: Option<usize>,
    ) -> Result<Self, ClassifierError> {
        if model_path.is_empty() || tokenizer_path.is_empty() {
            return Err(ClassifierError::BuildError("Model and tokenizer paths cannot be empty".to_string()));
        }
        if self.model_path.is_some() || self.tokenizer_path.is_some() {
            return Err(ClassifierError::BuildError("Model and tokenizer paths already set".to_string()));
        }
        
        // Validate paths exist
        if !std::path::Path::new(model_path).exists() {
            return Err(ClassifierError::BuildError(format!("Model file not found: {}", model_path)));
        }
        if !std::path::Path::new(tokenizer_path).exists() {
            return Err(ClassifierError::BuildError(format!("Tokenizer file not found: {}", tokenizer_path)));
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| {
                error!("Failed to load tokenizer: {}", e);
                ClassifierError::BuildError(format!("Failed to load tokenizer: {}", e))
            })?;
        info!("Tokenizer loaded successfully");

        // Initialize ONNX Runtime and load model
        let env = Arc::new(Environment::builder()
            .with_name("text_classifier")
            .build()
            .map_err(|e| ClassifierError::BuildError(format!("Failed to create environment: {}", e)))?);
        
        let session = SessionBuilder::new(&env)?
            .with_model_from_file(model_path)?;

        // Validate model structure
        Self::validate_model(&session)?;
        info!("Model structure validated successfully");
        
        // Store session and tokenizer temporarily
        self.tokenizer = Some(tokenizer);
        self.session = Some(session);
        
        // Infer embedding size by running a test input
        let test_text = "Test input to infer embedding size";
        let embedding = self.embed_text(test_text)?;
        
        let embedding_size = embedding.len();
        info!("Inferred embedding size from model: {}", embedding_size);
        
        // Set model characteristics with provided or default sequence length
        self.model_characteristics = Some(ModelCharacteristics {
            embedding_size,
            max_sequence_length: max_sequence_length.unwrap_or(512), // More reasonable default
            model_size_mb: 0, // Not critical for functionality
        });
        
        self.model_path = Some(model_path.to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string());
        Ok(self)
    }

    // Private helper to validate class data
    fn validate_class_data(label: &str, examples: &[impl AsRef<str>]) -> Result<(), ClassifierError> {
        if label.is_empty() {
            return Err(ClassifierError::ValidationError("Class label cannot be empty".into()));
        }
        if examples.is_empty() {
            return Err(ClassifierError::ValidationError(format!("Class '{}' has no examples", label)));
        }
        for (i, example) in examples.iter().enumerate() {
            if example.as_ref().is_empty() {
                return Err(ClassifierError::ValidationError(
                    format!("Empty example {} in class '{}'", i + 1, label)
                ));
            }
        }
        Ok(())
    }

    /// Adds a class with example texts
    pub fn add_class(mut self, label: &str, examples: Vec<&str>) -> Result<Self, ClassifierError> {
        if label.is_empty() {
            return Err(ClassifierError::ValidationError("Class label cannot be empty".to_string()));
        }
        if examples.is_empty() {
            return Err(ClassifierError::ValidationError("Examples list cannot be empty".to_string()));
        }
        if examples.iter().any(|e| e.is_empty()) {
            return Err(ClassifierError::ValidationError("Example text cannot be empty".to_string()));
        }
        if self.class_examples.contains_key(label) {
            return Err(ClassifierError::ValidationError(format!("Class '{}' already exists", label)));
        }
        self.class_examples.insert(label.to_string(), examples.iter().map(|s| s.to_string()).collect());
        Ok(self)
    }

    /// Builds the classifier, consuming the builder
    pub fn build(mut self) -> Result<Classifier, ClassifierError> {
        if self.model_path.is_none() || self.tokenizer_path.is_none() {
            return Err(ClassifierError::BuildError("Model and tokenizer paths must be set".to_string()));
        }
        if self.class_examples.is_empty() {
            return Err(ClassifierError::BuildError("At least one class must be added".to_string()));
        }

        let model_characteristics = self.model_characteristics
            .clone()
            .ok_or_else(|| ClassifierError::BuildError("Model characteristics not set".to_string()))?;

        // Validate all class data
        for (label, examples) in &self.class_examples {
            Self::validate_class_data(label, examples)?;
        }

        let mut embedded_prototypes = HashMap::new();
        
        // Process all examples before moving tokenizer and session
        let mut class_embeddings: Vec<(String, Vec<Array1<f32>>)> = Vec::new();
        for (label, examples) in &self.class_examples {
            info!("\nProcessing class '{}':", label);
            
            let embedded_examples: Vec<Array1<f32>> = examples.iter()
                .enumerate()
                .filter_map(|(i, text)| {
                    match self.embed_text(text) {
                        Ok(embedding) => Some(embedding),
                        Err(e) => {
                            error!("Failed to embed example {} for class '{}': {}", i + 1, label, e);
                            None
                        }
                    }
                })
                .collect();
            
            if embedded_examples.is_empty() {
                return Err(ClassifierError::BuildError(
                    format!("No valid embeddings generated for class '{}'", label)
                ));
            }
            
            class_embeddings.push((label.clone(), embedded_examples));
        }

        // Now we can safely take ownership of tokenizer and session
        let tokenizer = Arc::new(self.tokenizer.take()
            .ok_or_else(|| ClassifierError::BuildError("No tokenizer loaded".into()))?);
        let session = Arc::new(self.session.take()
            .ok_or_else(|| ClassifierError::BuildError("No ONNX model loaded".into()))?);

        // Process the embeddings into prototypes
        for (label, embedded_examples) in class_embeddings {
            let avg_vector = Classifier::average_vectors(&embedded_examples, model_characteristics.embedding_size);
            let prototype = Classifier::normalize_vector(&avg_vector);
            embedded_prototypes.insert(label, prototype);
        }
        
        Ok(Classifier {
            model_path: self.model_path.take().unwrap(),
            tokenizer_path: self.tokenizer_path.take().unwrap(),
            tokenizer,
            session,
            embedded_prototypes: Arc::new(embedded_prototypes),
            model_characteristics: self.model_characteristics.take().unwrap(),
        })
    }
}
/// A thread-safe text classifier using ONNX models for embedding and classification.
/// 
/// # Thread Safety
/// 
/// Single-thread usage:
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use text_classifier::{Classifier, BuiltinModel};
/// 
/// let classifier = Classifier::builder()
///     .with_model(BuiltinModel::MiniLM)?
///     .add_class("example", vec!["sample text"])?
///     .build()?;
/// 
/// classifier.predict("test text")?;
/// # Ok(())
/// # }
/// ```
/// 
/// Multi-thread usage:
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use text_classifier::{Classifier, BuiltinModel};
/// use std::sync::Arc;
/// use std::thread;
/// 
/// let classifier = Arc::new(Classifier::builder()
///     .with_model(BuiltinModel::MiniLM)?
///     .add_class("example", vec!["sample text"])?
///     .build()?);
/// 
/// let classifier_clone = Arc::clone(&classifier);
/// thread::spawn(move || {
///     classifier_clone.predict("test text").unwrap();
/// });
/// # Ok(())
/// # }
/// ```
pub struct Classifier {
    model_path: String,
    tokenizer_path: String,
    tokenizer: Arc<Tokenizer>,
    session: Arc<Session>,
    embedded_prototypes: Arc<HashMap<String, Array1<f32>>>,
    model_characteristics: ModelCharacteristics,
}

// Explicitly implement Send + Sync
unsafe impl Send for Classifier {}
unsafe impl Sync for Classifier {}

impl Classifier {
    /// Creates a new ClassifierBuilder for fluent construction
    pub fn builder() -> ClassifierBuilder {
        ClassifierBuilder::new()
    }

    /// Returns comprehensive information about the current state of the classifier,
    /// including model paths, number of classes, class labels, and embedding dimensions.
    pub fn info(&self) -> ClassifierInfo {
        ClassifierInfo {
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            num_classes: self.embedded_prototypes.len(),
            class_labels: self.embedded_prototypes.keys().cloned().collect(),
            embedding_size: self.model_characteristics.embedding_size,
        }
    }

    // Static helper methods
    fn normalize_vector(vec: &Array1<f32>) -> Array1<f32> {
        let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            vec / norm
        } else {
            Array1::zeros(vec.len())
        }
    }

    fn average_vectors(vectors: &[Array1<f32>], embedding_size: usize) -> Array1<f32> {
        if vectors.is_empty() {
            return Array1::zeros(embedding_size);
        }
        let sum = vectors.iter().fold(Array1::zeros(vectors[0].len()), |acc, v| acc + v);
        sum / vectors.len() as f32
    }

    fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.dot(b)
    }

    /// Returns the number of tokens in the input text
    /// This is useful to check if text will fit within the model's max_sequence_length
    /// before attempting classification
    pub fn count_tokens(&self, text: &str) -> Result<usize, ClassifierError> {
        TextEmbedding::count_tokens(self, text)
    }

    /// Makes a prediction for the given text
    pub fn predict(&self, text: &str) -> Result<(String, HashMap<String, f32>), ClassifierError> {
        if text.is_empty() {
            return Err(ClassifierError::ValidationError("Input text cannot be empty".into()));
        }
        
        // First check token count to provide a more specific error
        let token_count = self.count_tokens(text)?;
            
        if token_count > self.model_characteristics.max_sequence_length {
            return Err(ClassifierError::ValidationError(
                format!(
                    "Input text is too long ({} tokens, max is {}). Consider splitting the text into smaller chunks.",
                    token_count,
                    self.model_characteristics.max_sequence_length
                )
            ));
        }
        
        let input_vector = self.embed_text(text)?;
        
        let mut scores = HashMap::new();
        for (label, prototype) in self.embedded_prototypes.as_ref() {
            let similarity = Self::cosine_similarity(&input_vector, prototype);
            scores.insert(label.clone(), similarity);
        }
        
        let best_class = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(class, _)| class.clone())
            .unwrap_or_else(|| "unknown".to_string());
        
        Ok((best_class, scores))
    }
}

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
    /// Size of the embedding vectors
    pub embedding_size: usize,
} 