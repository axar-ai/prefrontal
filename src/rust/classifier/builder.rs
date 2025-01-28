use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use ort::session::Session;
use ndarray::Array1;
use log::{info, error};

use super::error::ClassifierError;
use super::embedding::TextEmbedding;
use super::classifier::Classifier;
use super::utils::{normalize_vector, average_vectors};
use crate::{BuiltinModel, ModelCharacteristics, runtime::{RuntimeConfig, create_session_builder}, ModelManager};

/// Represents a class definition with required label, description and optional examples
#[derive(Debug, Clone)]
pub struct ClassDefinition {
    /// The unique identifier for the class
    pub label: String,
    /// A detailed description of what this class represents.
    /// This helps document the class's purpose and can be used for:
    /// - Understanding the classifier's capabilities
    /// - Future extensibility (e.g., zero-shot classification)
    /// - API documentation
    pub description: String,
    /// Optional examples of text that belong to this class.
    /// When provided, these examples are used to compute the class prototype
    /// for similarity-based classification.
    pub examples: Option<Vec<String>>,
}

impl ClassDefinition {
    /// Creates a new class definition with required label and description
    /// 
    /// # Arguments
    /// * `label` - A unique identifier for the class
    /// * `description` - A detailed description of what this class represents
    /// 
    /// # Example
    /// ```
    /// use prefrontal::ClassDefinition;
    /// 
    /// let class = ClassDefinition::new(
    ///     "tech",
    ///     "Technology-related content including programming and software"
    /// );
    /// ```
    pub fn new(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
            examples: None,
        }
    }

    /// Adds examples to the class definition
    /// 
    /// # Arguments
    /// * `examples` - A vector of example texts that belong to this class
    /// 
    /// # Example
    /// ```
    /// use prefrontal::ClassDefinition;
    /// 
    /// let class = ClassDefinition::new(
    ///     "tech",
    ///     "Technology-related content"
    /// ).with_examples(vec![
    ///     "computer programming",
    ///     "software development"
    /// ]);
    /// ```
    pub fn with_examples(mut self, examples: Vec<impl Into<String>>) -> Self {
        self.examples = Some(examples.into_iter().map(Into::into).collect());
        self
    }
}

/// A builder for constructing a Classifier with a fluent interface.
#[derive(Default, Debug)]
pub struct ClassifierBuilder {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    tokenizer: Option<Tokenizer>,
    session: Option<Session>,
    class_examples: HashMap<String, Vec<String>>,
    class_descriptions: HashMap<String, String>,  // Added to store descriptions
    model_characteristics: Option<ModelCharacteristics>,
    runtime_config: RuntimeConfig,
}

impl TextEmbedding for ClassifierBuilder {
    /// Returns a reference to the tokenizer if it exists
    fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
    
    /// Returns a reference to the ONNX session if it exists
    fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }
}

impl ClassifierBuilder {
    /// Creates a new empty ClassifierBuilder instance with default configuration
    /// 
    /// # Example
    /// ```
    /// use prefrontal::ClassifierBuilder;
    /// 
    /// let builder = ClassifierBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            tokenizer: None,
            session: None,
            class_examples: HashMap::new(),
            class_descriptions: HashMap::new(),
            model_characteristics: None,
            runtime_config: RuntimeConfig::default(),
        }
    }

    /// Sets the runtime configuration for ONNX model execution
    /// 
    /// # Arguments
    /// * `config` - The RuntimeConfig containing settings for ONNX Runtime execution
    /// 
    /// # Example
    /// ```
    /// use prefrontal::{ClassifierBuilder, RuntimeConfig};
    /// 
    /// let config = RuntimeConfig::default();
    /// let builder = ClassifierBuilder::new()
    ///     .with_runtime_config(config);
    /// ```
    pub fn with_runtime_config(mut self, config: RuntimeConfig) -> Self {
        self.runtime_config = config;
        self
    }

    /// Sets the model to use for classification using a built-in model type
    /// 
    /// # Arguments
    /// * `model` - The BuiltinModel variant to use (e.g., MiniLM)
    /// 
    /// # Returns
    /// * `Result<Self, ClassifierError>` - The builder instance if successful, or an error if:
    ///   - The model paths are already set
    ///   - The model is not downloaded
    ///   - The model or tokenizer failed to load
    ///   - The model structure is invalid
    /// 
    /// # Example
    /// ```
    /// use prefrontal::{ClassifierBuilder, BuiltinModel};
    /// 
    /// let builder = ClassifierBuilder::new()
    ///     .with_model(BuiltinModel::MiniLM);
    /// ```
    pub fn with_model(mut self, model: BuiltinModel) -> Result<Self, ClassifierError> {
        if self.model_path.is_some() || self.tokenizer_path.is_some() {
            return Err(ClassifierError::BuildError("Model and tokenizer paths already set".to_string()));
        }

        // Initialize model manager with default location
        let manager = ModelManager::new_default()
            .map_err(|e| ClassifierError::BuildError(format!("Failed to create model manager: {}", e)))?;

        // Check if model is downloaded
        if !manager.is_model_downloaded(model) {
            return Err(ClassifierError::BuildError(format!(
                "Model '{:?}' is not downloaded. Please download it first using ModelManager::download_model()",
                model
            )));
        }

        // Get paths
        let model_path = manager.get_model_path(model);
        let tokenizer_path = manager.get_tokenizer_path(model);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| {
                error!("Failed to load tokenizer: {}", e);
                ClassifierError::BuildError(format!("Failed to load tokenizer: {}", e))
            })?;
        
        info!("Tokenizer loaded successfully");

        // Create session using the singleton environment
        let session = create_session_builder(&self.runtime_config)?
            .commit_from_file(&model_path)?;

        // Validate model structure
        Self::validate_model(&session)?;
        info!("Model structure validated successfully");
        
        // Store model characteristics
        self.model_characteristics = Some(model.characteristics());
        
        self.model_path = Some(model_path.to_string_lossy().to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string_lossy().to_string());
        self.tokenizer = Some(tokenizer);
        self.session = Some(session);
        Ok(self)
    }

    /// Sets a custom model and tokenizer path for the classifier
    /// 
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the tokenizer file
    /// * `max_sequence_length` - Optional maximum sequence length for the model. If not provided,
    ///   defaults to 256 tokens. This determines the maximum length of text that can be processed
    ///   before truncation.
    /// 
    /// # Returns
    /// * `Result<Self, ClassifierError>` - The builder instance if successful, or an error if:
    ///   - The model or tokenizer paths are empty
    ///   - The paths are already set
    ///   - The files don't exist
    ///   - The model or tokenizer failed to load
    ///   - The model structure is invalid
    /// 
    /// # Example
    /// ```
    /// use prefrontal::ClassifierBuilder;
    /// 
    /// let builder = ClassifierBuilder::new()
    ///     .with_custom_model(
    ///         "path/to/model.onnx",
    ///         "path/to/tokenizer.json",
    ///         Some(512)  // Custom sequence length
    ///     );
    /// ```
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

        // Create session using the singleton environment
        let session = create_session_builder(&self.runtime_config)?
            .commit_from_file(model_path)?;

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
            max_sequence_length: max_sequence_length.unwrap_or(256), // Default to MiniLM's limit
            model_size_mb: 0, // Not critical for functionality
        });
        
        self.model_path = Some(model_path.to_string());
        self.tokenizer_path = Some(tokenizer_path.to_string());
        Ok(self)
    }

    /// Validates class data according to the following rules:
    /// - Label must not be empty
    /// - Description must not be empty and must not exceed 1000 characters
    /// - Must have at least one example
    /// - No example text can be empty
    /// - Each example must be valid UTF-8
    /// 
    /// # Arguments
    /// * `label` - The class label to validate
    /// * `description` - The class description to validate
    /// * `examples` - The examples to validate
    /// 
    /// # Returns
    /// * `Ok(())` if validation passes
    /// * `Err(ClassifierError::ValidationError)` with a descriptive message if validation fails
    fn validate_class_data(
        label: &str,
        description: &str,
        examples: &[impl AsRef<str>]
    ) -> Result<(), ClassifierError> {
        const MAX_DESCRIPTION_LENGTH: usize = 1000;

        if label.is_empty() {
            return Err(ClassifierError::ValidationError("Class label cannot be empty".into()));
        }
        if description.is_empty() {
            return Err(ClassifierError::ValidationError("Class description cannot be empty".into()));
        }
        if description.len() > MAX_DESCRIPTION_LENGTH {
            return Err(ClassifierError::ValidationError(
                format!("Class description is too long ({} chars, max is {})",
                    description.len(), MAX_DESCRIPTION_LENGTH)
            ));
        }
        if examples.is_empty() {
            return Err(ClassifierError::ValidationError(
                format!("Class '{}' must have at least one example", label)
            ));
        }
        if let Some(pos) = examples.iter().position(|e| e.as_ref().is_empty()) {
            return Err(ClassifierError::ValidationError(
                format!("Example {} cannot be empty", pos + 1)
            ));
        }
        Ok(())
    }

    /// Adds a class to the classifier with its definition including label, description, and optional examples
    /// 
    /// # Arguments
    /// * `class` - The ClassDefinition containing the class label, description, and optional examples
    /// 
    /// # Returns
    /// * `Result<Self, ClassifierError>` - The builder instance if successful, or an error if:
    ///   - The class label is empty
    ///   - The description is empty or too long
    ///   - No examples are provided
    ///   - Any example text is empty
    ///   - Maximum number of classes (100) is exceeded
    /// 
    /// # Example
    /// ```
    /// use prefrontal::{ClassifierBuilder, ClassDefinition};
    /// 
    /// let builder = ClassifierBuilder::new()
    ///     .add_class(
    ///         ClassDefinition::new("tech", "Technology-related content")
    ///             .with_examples(vec!["programming", "software"])
    ///     );
    /// ```
    pub fn add_class(mut self, class: ClassDefinition) -> Result<Self, ClassifierError> {
        const MAX_CLASSES: usize = 100;
        
        // Get examples, defaulting to empty vec if None
        let examples = class.examples.unwrap_or_default();
        
        // Validate class data
        Self::validate_class_data(
            &class.label,
            &class.description,
            &examples
        )?;
        
        // Check number of classes
        if self.class_examples.len() >= MAX_CLASSES {
            return Err(ClassifierError::ValidationError(
                format!("Maximum number of classes ({}) exceeded", MAX_CLASSES)
            ));
        }
        
        // Store class data
        self.class_examples.insert(class.label.clone(), examples);
        self.class_descriptions.insert(class.label, class.description);
        
        Ok(self)
    }

    /// Builds and returns the final Classifier instance
    /// 
    /// # Returns
    /// * `Result<Classifier, ClassifierError>` - The constructed Classifier if successful, or an error if:
    ///   - No model and tokenizer paths are set
    ///   - No classes have been added
    ///   - Model characteristics are not set
    ///   - Failed to generate embeddings for any class examples
    /// 
    /// # Example
    /// ```no_run
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use prefrontal::{ClassifierBuilder, BuiltinModel, ClassDefinition};
    /// 
    /// let classifier = ClassifierBuilder::new()
    ///     .with_model(BuiltinModel::MiniLM)?
    ///     .add_class(
    ///         ClassDefinition::new("tech", "Technology content")
    ///             .with_examples(vec!["programming", "software"])
    ///     )?
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
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
            let avg_vector = average_vectors(&embedded_examples, model_characteristics.embedding_size);
            let prototype = normalize_vector(&avg_vector);
            embedded_prototypes.insert(label, prototype);
        }
        
        Ok(Classifier {
            model_path: self.model_path.take().unwrap(),
            tokenizer_path: self.tokenizer_path.take().unwrap(),
            tokenizer,
            session,
            embedded_prototypes: Arc::new(embedded_prototypes),
            class_descriptions: Arc::new(self.class_descriptions),
            model_characteristics: self.model_characteristics.take().unwrap(),
        })
    }

    /// Validates that the model has the expected input/output structure
    /// 
    /// # Arguments
    /// * `session` - The ONNX Runtime session to validate
    /// 
    /// # Returns
    /// * `Result<(), ClassifierError>` - Ok if validation passes, or an error if:
    ///   - The model doesn't have the required input tensors
    ///   - The model doesn't have any output tensors
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinModel;

    async fn setup_model() -> Result<ModelManager, Box<dyn std::error::Error>> {
        let manager = ModelManager::new_default()?;
        let model = BuiltinModel::MiniLM;
        
        if !manager.is_model_downloaded(model) {
            manager.download_model(model).await?;
        }
        assert!(manager.is_model_downloaded(model));
        Ok(manager)
    }

    #[tokio::test]
    async fn test_empty_class_handling() -> Result<(), Box<dyn std::error::Error>> {
        let manager = setup_model().await?;
        assert!(manager.is_model_downloaded(BuiltinModel::MiniLM));
        let result = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)?
            .add_class(
                ClassDefinition::new("empty", "Empty class")
                    .with_examples(vec![""])
            );
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_class_validation() -> Result<(), Box<dyn std::error::Error>> {
        let _manager = setup_model().await?;

        // Test invalid class label
        assert!(ClassifierBuilder::new()
            .with_model(BuiltinModel::MiniLM)?
            .add_class(ClassDefinition::new("", "Empty label"))
            .is_err());

        // Test invalid description
        assert!(ClassifierBuilder::new()
            .with_model(BuiltinModel::MiniLM)?
            .add_class(ClassDefinition::new("label", ""))
            .is_err());

        // Test invalid examples
        assert!(ClassifierBuilder::new()
            .with_model(BuiltinModel::MiniLM)?
            .add_class(
                ClassDefinition::new("label", "Test class")
                    .with_examples(vec![""])
            )
            .is_err());

        Ok(())
    }
} 