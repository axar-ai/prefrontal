/// Built-in models that can be used with the classifier
#[derive(Debug, Clone, Copy)]
pub enum BuiltinModel {
    /// Small and efficient model based on MiniLM architecture
    /// 
    /// Characteristics:
    /// - Embedding size: 384
    /// - Max sequence length: 256
    /// - Size: ~85MB
    /// - Good balance of speed and accuracy
    MiniLM,
}

/// Characteristics of a model, including its capabilities and resource requirements
#[derive(Debug, Clone)]
pub struct ModelCharacteristics {
    /// The dimensionality of embedding vectors produced by the model
    /// 
    /// This determines the size of the vectors used for similarity calculations.
    /// Higher dimensions can capture more nuanced relationships but use more memory.
    pub embedding_size: usize,

    /// Maximum number of tokens the model can process in a single input
    /// 
    /// Longer inputs will be truncated to this length. Each token roughly
    /// corresponds to 4-5 characters of text.
    pub max_sequence_length: usize,

    /// Approximate size of the model in megabytes when loaded into memory
    /// 
    /// This can help in capacity planning and determining if the model will
    /// fit in available system memory.
    pub model_size_mb: usize,
}

/// Information about a model's files and download locations
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub model_url: String,
    pub tokenizer_url: String,
    pub model_hash: String,
    pub tokenizer_hash: String,
}

impl BuiltinModel {
    /// Returns the characteristics of the model
    pub fn characteristics(&self) -> ModelCharacteristics {
        match self {
            BuiltinModel::MiniLM => ModelCharacteristics {
                embedding_size: 384,
                max_sequence_length: 256,
                model_size_mb: 85,
            }
        }
    }

    /// Returns the model info for downloading and verification
    pub fn get_model_info(&self) -> ModelInfo {
        match self {
            BuiltinModel::MiniLM => ModelInfo {
                name: "minilm".to_string(),
                model_url: "https://huggingface.co/axar-ai/minilm/resolve/main/model.onnx".to_string(),
                tokenizer_url: "https://huggingface.co/axar-ai/minilm/resolve/main/tokenizer.json".to_string(),
                model_hash: "37f1ea074b7166e87295fce31299287d5fb79f76b8b7227fccc8a9f2f1ba4e16".to_string(),
                tokenizer_hash: "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0".to_string(),
            }
        }
    }
} 