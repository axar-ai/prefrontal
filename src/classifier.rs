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
#[derive(Default)]
pub struct ClassifierBuilder {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    class_examples: HashMap<String, Vec<String>>,
}

impl ClassifierBuilder {
    /// Creates a new ClassifierBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model to use for classification
    pub fn with_model(mut self, model: BuiltinModel) -> Self {
        let (model_path, tokenizer_path) = model.get_paths();
        
        // Load tokenizer
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(tok) => {
                info!("Tokenizer loaded successfully");
                Some(tok)
            },
            Err(e) => {
                error!("Failed to load tokenizer: {}", e);
                None
            }
        };

        // Initialize ONNX Runtime
        let session = match Environment::builder()
            .with_name("text_classifier")
            .build()
            .map(|env| Arc::new(env))
            .and_then(|env| SessionBuilder::new(&env).unwrap().with_model_from_file(model_path)) {
                Ok(sess) => {
                    info!("ONNX model loaded successfully");
                    Some(sess)
                },
                Err(e) => {
                    error!("Failed to load ONNX model: {}", e);
                    None
                }
        };

        self.model_path = Some(model_path.to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string());
        self.tokenizer = tokenizer;
        self.session = session;
        
        self
    }

    /// Sets a custom model and tokenizer path
    pub fn with_custom_model(mut self, model_path: &str, tokenizer_path: &str) -> Self {
        // Load tokenizer
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(tok) => {
                info!("Tokenizer loaded successfully");
                Some(tok)
            },
            Err(e) => {
                error!("Failed to load tokenizer: {}", e);
                None
            }
        };

        // Initialize ONNX Runtime
        let session = match Environment::builder()
            .with_name("text_classifier")
            .build()
            .map(|env| Arc::new(env))
            .and_then(|env| SessionBuilder::new(&env).unwrap().with_model_from_file(model_path)) {
                Ok(sess) => {
                    info!("ONNX model loaded successfully");
                    Some(sess)
                },
                Err(e) => {
                    error!("Failed to load ONNX model: {}", e);
                    None
                }
        };

        self.model_path = Some(model_path.to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string());
        self.tokenizer = tokenizer;
        self.session = session;
        
        self
    }

    /// Adds a class with example texts
    pub fn add_class(mut self, label: &str, examples: Vec<&str>) -> Self {
        let examples: Vec<String> = examples.into_iter()
            .map(String::from)
            .collect();
            
        self.class_examples.insert(label.to_string(), examples);
        self
    }

    /// Builds the classifier, consuming the builder
    pub fn build(self) -> Result<Classifier, ClassifierError> {
        // Validate required fields
        if self.model_path.is_none() || self.tokenizer_path.is_none() {
            return Err(ClassifierError::BuildError("Model not set. Call with_model() first".into()));
        }
        if self.tokenizer.is_none() {
            return Err(ClassifierError::BuildError("Failed to load tokenizer".into()));
        }
        if self.session.is_none() {
            return Err(ClassifierError::BuildError("Failed to load ONNX model".into()));
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

    // Helper methods for embedding during build
    fn embed_text(&self, text: &str) -> Option<Array1<f32>> {
        if let Some(tokens) = self.tokenize(text) {
            self.get_embedding(&tokens)
        } else {
            None
        }
    }

    fn tokenize(&self, text: &str) -> Option<Vec<u32>> {
        self.tokenizer.as_ref()?.encode(text, false).ok()
            .map(|encoding| encoding.get_ids().to_vec())
    }

    fn get_embedding(&self, tokens: &[u32]) -> Option<Array1<f32>> {
        let session = self.session.as_ref()?;

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

    // Instance methods for prediction
    fn tokenize(&self, text: &str) -> Option<Vec<u32>> {
        self.tokenizer.as_ref()?.encode(text, false).ok()
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
        let session = self.session.as_ref()?;

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

        Some(Self::normalize_vector(&embedding))
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