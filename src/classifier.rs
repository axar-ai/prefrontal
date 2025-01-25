use ndarray::Array1;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use ort::{Environment, Session, SessionBuilder, Value, tensor::OrtOwnedTensor};
use ndarray::{Array2};
use std::sync::Arc;
use log::{info, warn, error, debug, trace};

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
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        debug!("Creating new classifier");
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
                None
            }
        };

        // Initialize ONNX Runtime with Arc
        debug!("Initializing ONNX Runtime...");
        let session = match Environment::builder()
            .with_name("text_classifier")
            .build()
            .map(|env| Arc::new(env))  // Wrap in Arc
            .and_then(|env| SessionBuilder::new(&env)?.with_model_from_file(model_path)) {
                Ok(sess) => {
                    info!("ONNX model loaded successfully");
                    Some(sess)
                },
                Err(e) => {
                    error!("Failed to load ONNX model: {}", e);
                    None
                }
        };
        
        let classifier = Self {
            model_path: model_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            tokenizer,
            session,
            class_prototypes: HashMap::new(),
            embedded_prototypes: HashMap::new(),
        };
        
        info!("Classifier created successfully");
        classifier
    }

    pub fn add_class(&mut self, label: &str, examples: Vec<&str>) {
        info!("\n=== Adding Class ===");
        info!("Label: {}", label);
        info!("Number of examples: {}", examples.len());
        info!("Examples:");
        for (i, ex) in examples.iter().enumerate() {
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

    pub fn build(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("\n=== Building Classifier ===");
        
        // Fail early if no tokenizer or session
        if self.tokenizer.is_none() {
            return Err("No tokenizer available".into());
        }
        if self.session.is_none() {
            return Err("No ONNX session available".into());
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
                    self.embed_text(text).unwrap()
                })
                .collect();
            
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

    pub fn predict(&self, text: &str) -> Result<(String, HashMap<String, f32>), Box<dyn std::error::Error>> {
        debug!("Making prediction for text: {}", text);
        
        if let Some(tokens) = self.tokenize(text) {
            debug!("Tokenization successful: {} tokens", tokens.len());
            trace!("Tokens: {:?}", tokens);
        } else {
            warn!("Tokenization failed, falling back to mock embedding");
        }
        
        let input_vector = self.embed_text(text)
            .ok_or("Failed to generate embedding")?;
        
        info!("Input vector created with shape: {:?}", input_vector.shape());
        
        // Verify input vector is normalized
        let input_norm: f32 = input_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        info!("Input vector norm: {:.6}", input_norm);
        debug_assert!((input_norm - 1.0).abs() < 1e-6, "Input vector not normalized: {}", input_norm);
        
        let mut scores = HashMap::new();
        
        info!("\nCalculating scores:");
        for (label, prototype) in &self.embedded_prototypes {
            info!("\nClass: {}", label);
            
            // Verify prototype is normalized
            let proto_norm: f32 = prototype.iter().map(|&x| x * x).sum::<f32>().sqrt();
            info!("Prototype norm: {:.6}", proto_norm);
            debug_assert!((proto_norm - 1.0).abs() < 1e-6, "Prototype not normalized: {}", proto_norm);
            
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
                    info!("Could not determine best class, using 'unknown'");
                    "unknown".to_string()
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

