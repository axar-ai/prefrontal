use ndarray::Array1;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use ort::{Environment, Session, SessionBuilder, Value, tensor::OrtOwnedTensor};
use ndarray::{Array2};
use std::sync::Arc;
use log::{info, error, debug, trace};

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

impl std::fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// A text classifier that uses ONNX models for embedding and classification.
/// 
/// This classifier works by:
/// 1. Converting text into embeddings using a pre-trained ONNX model
/// 2. Creating class prototypes by averaging embeddings of example texts
/// 3. Classifying new text by comparing its embedding to class prototypes
/// 
/// # Example
/// ```no_run
/// use text_classifier::Classifier;
/// 
/// let mut classifier = Classifier::new(
///     "path/to/model.onnx",
///     "path/to/tokenizer.json"
/// )?;
/// 
/// // Add examples for each class
/// classifier.add_class("sports", vec!["football game", "basketball match"])?;
/// classifier.add_class("tech", vec!["computer program", "software code"])?;
/// 
/// // Build the classifier
/// classifier.build()?;
/// 
/// // Make predictions
/// let (class, scores) = classifier.predict("new soccer match")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub struct Classifier {
    model_path: String,
    tokenizer_path: String,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    class_prototypes: HashMap<String, Vec<String>>,
    embedded_prototypes: HashMap<String, Array1<f32>>,
}

impl Classifier {
    /// Creates a new text classifier with the specified model and tokenizer.
    /// 
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the tokenizer JSON file
    /// 
    /// # Returns
    /// * `Ok(Classifier)` - A new classifier instance if initialization succeeds
    /// * `Err(ClassifierError)` - If model/tokenizer loading fails or paths are invalid
    /// 
    /// # Errors
    /// Returns `ClassifierError::ValidationError` if:
    /// * `model_path` is empty
    /// * `tokenizer_path` is empty
    /// 
    /// Returns `ClassifierError::TokenizerError` if:
    /// * Tokenizer file cannot be loaded
    /// * Tokenizer format is invalid
    /// 
    /// Returns `ClassifierError::ModelError` if:
    /// * ONNX model file cannot be loaded
    /// * Model format is invalid
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, ClassifierError> {
        debug!("Creating new classifier");
        
        if model_path.is_empty() {
            return Err(ClassifierError::ValidationError("Model path cannot be empty".to_string()));
        }
        if tokenizer_path.is_empty() {
            return Err(ClassifierError::ValidationError("Tokenizer path cannot be empty".to_string()));
        }
        
        info!("Model path: {}", model_path);
        info!("Tokenizer path: {}", tokenizer_path);
        
        // Load tokenizer
        debug!("Loading tokenizer...");
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(tok) => {
                info!("Tokenizer loaded successfully");
                Some(tok)
            },
            Err(e) => {
                error!("Failed to load tokenizer: {}", e);
                return Err(ClassifierError::TokenizerError(e.to_string()));
            }
        };

        // Initialize ONNX Runtime with Arc
        debug!("Initializing ONNX Runtime...");
        let session = match Environment::builder()
            .with_name("text_classifier")
            .build()
            .map(|env| Arc::new(env))
            .and_then(|env| SessionBuilder::new(&env)?.with_model_from_file(model_path)) {
                Ok(sess) => {
                    info!("ONNX model loaded successfully");
                    Some(sess)
                },
                Err(e) => {
                    error!("Failed to load ONNX model: {}", e);
                    return Err(ClassifierError::ModelError(e.to_string()));
                }
        };
        
        info!("Classifier created successfully");
        Ok(Self {
            model_path: model_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            tokenizer,
            session,
            class_prototypes: HashMap::new(),
            embedded_prototypes: HashMap::new(),
        })
    }

    /// Adds a new class with example texts to the classifier.
    /// 
    /// # Arguments
    /// * `label` - The name/label of the class to add
    /// * `examples` - A list of example texts that represent this class
    /// 
    /// # Returns
    /// * `Ok(())` - If the class was added successfully
    /// * `Err(ClassifierError)` - If validation fails
    /// 
    /// # Errors
    /// Returns `ClassifierError::ValidationError` if:
    /// * `label` is empty
    /// * `examples` is empty
    /// * Any example text is empty
    pub fn add_class(&mut self, label: &str, examples: Vec<&str>) -> Result<(), ClassifierError> {
        info!("\n=== Adding Class ===");
        
        if label.is_empty() {
            return Err(ClassifierError::ValidationError("Label cannot be empty".to_string()));
        }
        if examples.is_empty() {
            return Err(ClassifierError::ValidationError("Must provide at least one example".to_string()));
        }
        
        info!("Label: {}", label);
        info!("Number of examples: {}", examples.len());
        info!("Examples:");
        for (i, ex) in examples.iter().enumerate() {
            if ex.is_empty() {
                return Err(ClassifierError::ValidationError(format!("Example {} cannot be empty", i + 1)));
            }
            info!("  {}: {}", i + 1, ex);
        }
        
        let examples: Vec<String> = examples.into_iter()
            .map(String::from)
            .collect();
            
        info!("Converting examples to String type...");
        self.class_prototypes.insert(label.to_string(), examples);
        info!("Class added successfully");
        
        // Debug: Print current state
        info!("\nCurrent state:");
        info!("Number of classes: {}", self.class_prototypes.len());
        for (l, exs) in &self.class_prototypes {
            info!("Class '{}': {} examples", l, exs.len());
        }
        
        Ok(())
    }

    fn normalize_vector(vec: &Array1<f32>) -> Array1<f32> {
        info!("Normalizing vector...");
        let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        info!("Vector norm: {:.6}", norm);
        
        if norm > 1e-10 {
            let normalized = vec / norm;
            let check_norm: f32 = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
            info!("Normalized vector norm: {:.6}", check_norm);
            normalized
        } else {
            info!("Warning: Vector has near-zero norm, returning zeros");
            Array1::zeros(vec.len())
        }
    }

    fn get_embedding(&self, tokens: &[u32]) -> Option<Array1<f32>> {
        let session = self.session.as_ref()?;

        // Debug input shapes and types
        info!("Input tokens length: {}", tokens.len());
        
        // Create input arrays with proper lifetime management
        let input_array = Array2::from_shape_vec((1, tokens.len()), 
            tokens.iter().map(|&x| x as i64).collect())
            .ok()?;
        let input_dyn = input_array.into_dyn();
        let input_ids = input_dyn.as_standard_layout();
        info!("Input IDs shape: {:?}, type: i64", input_ids.shape());
        
        let mask_array = Array2::from_shape_vec((1, tokens.len()),
            tokens.iter().map(|&x| if x == 0 { 0i64 } else { 1i64 }).collect())
            .ok()?;
        let mask_dyn = mask_array.into_dyn();
        let attention_mask = mask_dyn.as_standard_layout();
        info!("Attention mask shape: {:?}, type: i64", attention_mask.shape());

        info!("Arrays converted to CowRepr");
        
        // Create input tensors
        let input_tensors = vec![
            Value::from_array(session.allocator(), &input_ids).ok()?,
            Value::from_array(session.allocator(), &attention_mask).ok()?,
        ];

        // Run inference and debug output
        info!("Running ONNX inference...");
        let outputs = session.run(input_tensors).ok()?;
        info!("Got {} outputs", outputs.len());
        
        // Extract and debug output tensor
        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().ok()?;
        info!("Output tensor type: {:?}", std::any::type_name_of_val(&output_tensor));
        
        // Convert output tensor to ndarray
        let array = output_tensor.view();
        let shape = array.shape();
        info!("Output array shape: {:?}", shape);
        
        // Get the [CLS] token embedding (first token)
        let mut embedding = Array1::zeros(shape[2]);
        
        // Copy the first token's embedding
        let embedding_slice = array.slice(ndarray::s![0, 0, ..]);
        embedding.assign(&Array1::from_iter(embedding_slice.iter().cloned()));

        info!("Embedding extracted successfully");
        Some(Self::normalize_vector(&embedding))
    }

    pub fn tokenize(&self, text: &str) -> Option<Vec<u32>> {
        if let Some(tokenizer) = &self.tokenizer {
            match tokenizer.encode(text, false) {
                Ok(encoding) => {
                    info!("Text tokenized successfully");
                    info!("Tokens: {:?}", encoding.get_ids());
                    Some(encoding.get_ids().to_vec())
                },
                Err(e) => {
                    error!("Failed to tokenize text: {}", e);
                    None
                }
            }
        } else {
            error!("No tokenizer available");
            None
        }
    }

    pub fn embed_text(&self, text: &str) -> Option<Array1<f32>> {
        if let Some(tokens) = self.tokenize(text) {
            self.get_embedding(&tokens)
        } else {
            error!("Failed to tokenize text");
            None
        }
    }

    /// Builds the classifier by computing embeddings for all example texts.
    /// 
    /// This method must be called after adding all classes and before making predictions.
    /// It computes embeddings for all example texts and creates class prototypes
    /// by averaging the embeddings for each class.
    /// 
    /// # Returns
    /// * `Ok(())` - If build succeeds
    /// * `Err(Box<dyn Error>)` - If build fails
    /// 
    /// # Errors
    /// * If tokenizer is not initialized
    /// * If ONNX session is not initialized
    /// * If embedding computation fails for any example
    pub fn build(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("\n=== Building Classifier ===");
        
        // Fail early if no tokenizer or session
        if self.tokenizer.is_none() {
            return Err(ClassifierError::BuildError("No tokenizer available".into()).into());
        }
        if self.session.is_none() {
            return Err(ClassifierError::BuildError("No ONNX session available".into()).into());
        }

        // Validate we have at least one class with examples
        if self.class_prototypes.is_empty() {
            return Err(ClassifierError::ValidationError("No classes added. Add at least one class with examples before building".into()).into());
        }

        info!("Initial state:");
        info!("Number of classes: {}", self.class_prototypes.len());
        
        self.embedded_prototypes.clear();
        info!("Cleared existing prototypes");
        
        for (label, examples) in &self.class_prototypes {
            info!("\nProcessing class '{}':", label);
            info!("Number of examples: {}", examples.len());
            
            let embedded_examples: Vec<Array1<f32>> = examples.iter()
                .enumerate()
                .map(|(i, text)| {
                    info!("Processing example {} for: {}", i + 1, text);
                    self.embed_text(text).ok_or_else(|| ClassifierError::BuildError(
                        format!("Failed to embed example {} for class '{}'", i + 1, label)
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            
            if embedded_examples.is_empty() {
                return Err(ClassifierError::BuildError(
                    format!("No valid embeddings generated for class '{}'", label)
                ).into());
            }
            
            // Average the embeddings
            info!("Computing average vector...");
            let avg_vector = Self::average_vectors(&embedded_examples);
            info!("Average vector created with shape: {:?}", avg_vector.shape());
            
            // Normalize the average
            let prototype = Self::normalize_vector(&avg_vector);
            info!("Final prototype shape: {:?}", prototype.shape());
            
            self.embedded_prototypes.insert(label.clone(), prototype);
            info!("Stored prototype for class '{}'", label);
        }
        
        info!("=== Build Complete ===\n");
        Ok(())
    }

    fn average_vectors(vectors: &[Array1<f32>]) -> Array1<f32> {
        if vectors.is_empty() {
            info!("Warning: Attempting to average empty vector list");
            return Array1::zeros(384);
        }
        
        let sum = vectors.iter().fold(Array1::zeros(vectors[0].len()), |acc, v| acc + v);
        sum / vectors.len() as f32
    }

    fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        // Since vectors are already normalized, dot product equals cosine similarity
        let sim = a.dot(b);
        info!("Cosine similarity: {:.6}", sim);
        
        // Add checks
        debug_assert!(sim <= 1.0 + 1e-6, "Similarity > 1: {}", sim);
        debug_assert!(sim >= -1.0 - 1e-6, "Similarity < -1: {}", sim);
        
        sim
    }

    /// Makes a prediction for the given text.
    /// 
    /// Computes the embedding for the input text and compares it to all class
    /// prototypes using cosine similarity. Returns the best matching class
    /// and similarity scores for all classes.
    /// 
    /// # Arguments
    /// * `text` - The text to classify
    /// 
    /// # Returns
    /// * `Ok((String, HashMap<String, f32>))` - The predicted class and similarity scores
    /// * `Err(Box<dyn Error>)` - If prediction fails
    /// 
    /// # Errors
    /// * If tokenization fails
    /// * If embedding computation fails
    /// * If no class prototypes are available (build() not called)
    pub fn predict(&self, text: &str) -> Result<(String, HashMap<String, f32>), Box<dyn std::error::Error>> {
        debug!("Making prediction for text: {}", text);
        
        // Validate input text
        if text.is_empty() {
            return Err(ClassifierError::ValidationError("Input text cannot be empty".into()).into());
        }
        
        // Validate that build() was called
        if self.embedded_prototypes.is_empty() {
            return Err(ClassifierError::ValidationError(
                "No class prototypes available. Call build() before making predictions".into()
            ).into());
        }
        
        if let Some(tokens) = self.tokenize(text) {
            debug!("Tokenization successful: {} tokens", tokens.len());
            trace!("Tokens: {:?}", tokens);
        } else {
            return Err(ClassifierError::PredictionError("Failed to tokenize input text".into()).into());
        }
        
        let input_vector = self.embed_text(text)
            .ok_or_else(|| ClassifierError::PredictionError("Failed to generate embedding".into()))?;
        
        info!("Input vector created with shape: {:?}", input_vector.shape());
        
        // Verify input vector is normalized
        let input_norm: f32 = input_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        info!("Input vector norm: {:.6}", input_norm);
        if (input_norm - 1.0).abs() >= 1e-6 {
            return Err(ClassifierError::PredictionError(
                format!("Input vector not properly normalized: {}", input_norm)
            ).into());
        }
        
        let mut scores = HashMap::new();
        
        info!("\nCalculating scores:");
        for (label, prototype) in &self.embedded_prototypes {
            info!("\nClass: {}", label);
            
            // Verify prototype is normalized
            let proto_norm: f32 = prototype.iter().map(|&x| x * x).sum::<f32>().sqrt();
            info!("Prototype norm: {:.6}", proto_norm);
            if (proto_norm - 1.0).abs() >= 1e-6 {
                return Err(ClassifierError::PredictionError(
                    format!("Prototype for class '{}' not properly normalized: {}", label, proto_norm)
                ).into());
            }
            
            info!("Computing cosine similarity...");
            let similarity = Self::cosine_similarity(&input_vector, prototype);
            
            scores.insert(label.clone(), similarity);
        }
        
        // Find best class
        info!("\nFinal scores: {:?}", scores);
        let best_class = if scores.is_empty() {
            info!("No scores available, using 'unknown'");
            "unknown".to_string()
        } else {
            match scores.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)) {
                Some((class, score)) => {
                    info!("Best class found: {} (score: {})", class, score);
                    class.clone()
                },
                None => {
                    return Err(ClassifierError::PredictionError("Failed to compare similarity scores".into()).into());
                }
            }
        };
        
        info!("\nFinal decision:");
        info!("Best class: {}", best_class);
        info!("All scores: {:?}", scores);
        info!("=== Prediction Complete ===\n");
        
        Ok((best_class, scores))
    }
}

