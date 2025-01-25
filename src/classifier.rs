use ndarray::Array1;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use ort::{Environment, Session, SessionBuilder, Value, tensor::OrtOwnedTensor};
use ndarray::{Array2};
use std::sync::Arc;
use log::{info, error};
use crate::ClassifierError;
use crate::BuiltinModel;

/// A builder for constructing a Classifier with a fluent interface.
#[derive(Default, Debug)]
pub struct ClassifierBuilder {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    class_examples: HashMap<String, Vec<String>>,
}

/// Private trait for embedding functionality
trait TextEmbedding {
    fn tokenizer(&self) -> Option<&Tokenizer>;
    fn session(&self) -> Option<&Session>;
    
    fn tokenize(&self, text: &str) -> Option<Vec<u32>> {
        self.tokenizer()?.encode(text, false).ok()
            .map(|encoding| encoding.get_ids().to_vec())
    }

    fn embed_text(&self, text: &str) -> Option<Array1<f32>> {
        if let Some(tokens) = self.tokenize(text) {
            self.get_embedding(&tokens)
        } else {
            None
        }
    }

    fn get_embedding(&self, tokens: &[u32]) -> Option<Array1<f32>> {
        let session = self.session()?;

        let input_array = Array2::from_shape_vec((1, tokens.len()), 
            tokens.iter().map(|&x| x as i64).collect()).ok()?;
        let input_dyn = input_array.into_dyn();
        let input_ids = input_dyn.as_standard_layout();
        
        let mask_array = Array2::from_shape_vec((1, tokens.len()),
            tokens.iter().map(|&x| if x == 0 { 0i64 } else { 1i64 }).collect()).ok()?;
        let mask_dyn = mask_array.into_dyn();
        let attention_mask = mask_dyn.as_standard_layout();
        
        let input_tensors = vec![
            Value::from_array(session.allocator(), &input_ids).ok()?,
            Value::from_array(session.allocator(), &attention_mask).ok()?,
        ];

        let outputs = session.run(input_tensors).ok()?;
        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().ok()?;
        let array = output_tensor.view();
        
        let mut embedding = Array1::zeros(array.shape()[2]);
        let embedding_slice = array.slice(ndarray::s![0, 0, ..]);
        embedding.assign(&Array1::from_iter(embedding_slice.iter().cloned()));

        Some(Classifier::normalize_vector(&embedding))
    }
}

impl TextEmbedding for ClassifierBuilder {
    fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
    
    fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }
}

impl TextEmbedding for Classifier {
    fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
    
    fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }
}

impl ClassifierBuilder {
    /// Creates a new ClassifierBuilder
    pub fn new() -> Self {
        Self::default()
    }

    // Private helper to load model and tokenizer
    fn load_model(&mut self, model_path: &str, tokenizer_path: &str) -> Result<(), ClassifierError> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| {
                error!("Failed to load tokenizer: {}", e);
                ClassifierError::BuildError(format!("Failed to load tokenizer: {}", e))
            })?;
        info!("Tokenizer loaded successfully");

        // Initialize ONNX Runtime
        let session = Environment::builder()
            .with_name("text_classifier")
            .build()
            .map_err(|e| ClassifierError::BuildError(format!("Failed to create environment: {}", e)))?;
        
        let session = Arc::new(session);
        let session = SessionBuilder::new(&session)
            .and_then(|builder| builder.with_model_from_file(model_path))
            .map_err(|e| {
                error!("Failed to load ONNX model: {}", e);
                ClassifierError::BuildError(format!("Failed to load ONNX model: {}", e))
            })?;
        info!("ONNX model loaded successfully");

        self.model_path = Some(model_path.to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string());
        self.tokenizer = Some(tokenizer);
        self.session = Some(session);
        Ok(())
    }

    /// Sets the model to use for classification
    pub fn with_model(mut self, model: BuiltinModel) -> Result<Self, ClassifierError> {
        let (model_path, tokenizer_path) = model.get_paths();
        self.load_model(model_path, tokenizer_path)?;
        Ok(self)
    }

    /// Sets a custom model and tokenizer path
    pub fn with_custom_model(mut self, model_path: &str, tokenizer_path: &str) -> Result<Self, ClassifierError> {
        self.load_model(model_path, tokenizer_path)?;
        Ok(self)
    }

    /// Adds a class with example texts
    pub fn add_class(mut self, label: &str, examples: Vec<&str>) -> Result<Self, ClassifierError> {
        // Validate inputs first
        if label.is_empty() {
            return Err(ClassifierError::ValidationError("Class label cannot be empty".into()));
        }
        if examples.is_empty() {
            return Err(ClassifierError::ValidationError(format!("Class '{}' has no examples", label)));
        }
        for (i, example) in examples.iter().enumerate() {
            if example.is_empty() {
                return Err(ClassifierError::ValidationError(
                    format!("Empty example {} in class '{}'", i + 1, label)
                ));
            }
        }

        let examples: Vec<String> = examples.into_iter()
            .map(String::from)
            .collect();
            
        self.class_examples.insert(label.to_string(), examples);
        Ok(self)
    }

    /// Builds the classifier, consuming the builder
    pub fn build(self) -> Result<Classifier, ClassifierError> {
        // Validate required fields
        if self.model_path.is_none() || self.tokenizer_path.is_none() {
            return Err(ClassifierError::BuildError("Model not set. Call with_model() first".into()));
        }
        if self.tokenizer.is_none() {
            return Err(ClassifierError::BuildError("No tokenizer loaded".into()));
        }
        if self.session.is_none() {
            return Err(ClassifierError::BuildError("No ONNX model loaded".into()));
        }
        if self.class_examples.is_empty() {
            return Err(ClassifierError::ValidationError("No classes added. Add at least one class with examples".into()));
        }

        // Validate class data
        for (label, examples) in &self.class_examples {
            if label.is_empty() {
                return Err(ClassifierError::ValidationError("Class label cannot be empty".into()));
            }
            if examples.is_empty() {
                return Err(ClassifierError::ValidationError(format!("Class '{}' has no examples", label)));
            }
            for (i, example) in examples.iter().enumerate() {
                if example.is_empty() {
                    return Err(ClassifierError::ValidationError(
                        format!("Empty example {} in class '{}'", i + 1, label)
                    ));
                }
            }
        }

        let mut embedded_prototypes = HashMap::new();
        
        for (label, examples) in &self.class_examples {
            info!("\nProcessing class '{}':", label);
            
            let embedded_examples: Vec<Array1<f32>> = examples.iter()
                .enumerate()
                .filter_map(|(i, text)| {
                    match self.embed_text(text) {
                        Some(embedding) => Some(embedding),
                        None => {
                            error!("Failed to embed example {} for class '{}'", i + 1, label);
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
            
            let avg_vector = Classifier::average_vectors(&embedded_examples);
            let prototype = Classifier::normalize_vector(&avg_vector);
            embedded_prototypes.insert(label.clone(), prototype);
        }
        
        Ok(Classifier {
            model_path: self.model_path.unwrap(),
            tokenizer_path: self.tokenizer_path.unwrap(),
            tokenizer: self.tokenizer,
            session: self.session,
            embedded_prototypes,
        })
    }
}

/// A text classifier that uses ONNX models for embedding and classification.
pub struct Classifier {
    model_path: String,
    tokenizer_path: String,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    embedded_prototypes: HashMap<String, Array1<f32>>,
}

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
            embedding_size: self.embedded_prototypes
                .values()
                .next()
                .map(|v| v.len())
                .unwrap_or(0),
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

    fn average_vectors(vectors: &[Array1<f32>]) -> Array1<f32> {
        if vectors.is_empty() {
            return Array1::zeros(384);
        }
        let sum = vectors.iter().fold(Array1::zeros(vectors[0].len()), |acc, v| acc + v);
        sum / vectors.len() as f32
    }

    fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.dot(b)
    }

    /// Makes a prediction for the given text
    pub fn predict(&self, text: &str) -> Result<(String, HashMap<String, f32>), ClassifierError> {
        if text.is_empty() {
            return Err(ClassifierError::ValidationError("Input text cannot be empty".into()));
        }
        
        let input_vector = self.embed_text(text)
            .ok_or_else(|| ClassifierError::PredictionError("Failed to generate embedding".into()))?;
        
        let mut scores = HashMap::new();
        for (label, prototype) in &self.embedded_prototypes {
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