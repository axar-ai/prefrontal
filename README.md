# Prefrontal

A blazing fast text classifier for real-time agent routing, built in Rust. Prefrontal provides a simple yet powerful interface for routing conversations to the right agent or department based on text content, powered by transformer-based models.

## System Requirements

To use Prefrontal in your project, you need:

1. **Build Dependencies**
   - Linux: `sudo apt-get install cmake pkg-config libssl-dev`
   - macOS: `brew install cmake pkg-config openssl`
   - Windows: Install CMake and OpenSSL

Note: ONNX Runtime v2.0.0-rc.9 is automatically downloaded and managed by the crate.

## Features

- ⚡️ Blazing fast (~10ms) classification for real-time routing
- 🎯 Optimized for agent routing and conversation classification
- 🚀 Easy-to-use builder pattern interface
- 🔧 Support for both built-in and custom ONNX models
- 📊 Multiple class classification with confidence scores
- 🛠️ Automatic model management and downloading
- 🧪 Thread-safe for high-concurrency environments
- 📝 Comprehensive logging and error handling

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
prefrontal = "0.1.0"
```

## Quick Start

```rust
use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ModelManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download the model if not already present
    let manager = ModelManager::new_default()?;
    let model = BuiltinModel::MiniLM;

    if !manager.is_model_downloaded(model) {
        println!("Downloading model...");
        manager.download_model(model).await?;
    }

    // Initialize the classifier with built-in MiniLM model
    let classifier = Classifier::builder()
        .with_model(model)?
        .add_class(
            ClassDefinition::new(
                "technical_support",
                "Technical issues and product troubleshooting"
            ).with_examples(vec![
                "my app keeps crashing",
                "can't connect to the server",
                "integration setup help"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "billing",
                "Billing and subscription inquiries"
            ).with_examples(vec![
                "update credit card",
                "cancel subscription",
                "billing cycle question"
            ])
        )?
        .build()?;

    // Route incoming message
    let (department, scores) = classifier.predict("I need help setting up the API integration")?;
    println!("Route to: {}", department);
    println!("Confidence scores: {:?}", scores);

    Ok(())
}
```

## Built-in Models

### MiniLM

The MiniLM model is a small and efficient model optimized for text classification:

- **Embedding Size**: 384 dimensions

  - Determines the size of vectors used for similarity calculations
  - Balanced for capturing semantic relationships while being memory efficient

- **Max Sequence Length**: 256 tokens

  - Each token roughly corresponds to 4-5 characters
  - Longer inputs are automatically truncated

- **Model Size**: ~85MB
  - Compact size for efficient deployment
  - Good balance of speed and accuracy

Models are automatically downloaded from HuggingFace when needed:

- Model: `https://huggingface.co/axar-ai/minilm/resolve/main/model.onnx`
- Tokenizer: `https://huggingface.co/axar-ai/minilm/resolve/main/tokenizer.json`

## Runtime Configuration

You can fine-tune the ONNX runtime configuration for optimal performance:

```rust
use prefrontal::{Classifier, RuntimeConfig};

let config = RuntimeConfig {
    // Number of threads for parallel model execution
    // 0 = let ONNX Runtime decide (recommended)
    inter_threads: 0,

    // Number of threads for parallel computation within nodes
    // 0 = let ONNX Runtime decide (recommended)
    intra_threads: 0,

    // Graph optimization level (Level3 recommended for production)
    optimization_level: GraphOptimizationLevel::Level3,
};

let classifier = Classifier::builder()
    .with_runtime_config(config)
    // ... rest of configuration ...
    .build()?;
```

## Custom Models

You can use your own ONNX models:

```rust
let classifier = Classifier::builder()
    .with_custom_model(
        "path/to/model.onnx",
        "path/to/tokenizer.json",
        Some(512)  // Optional: custom sequence length
    )?
    .add_class(
        ClassDefinition::new(
            "custom_class",
            "Description of the custom class"
        ).with_examples(vec!["example text"])
    )?
    .build()?;
```

## Model Management

The library includes a model management system that handles downloading and verifying models:

```rust
use prefrontal::{ModelManager, BuiltinModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model = BuiltinModel::MiniLM;

    // Check if model is downloaded
    if !manager.is_model_downloaded(model) {
        println!("Downloading model...");
        manager.download_model(model).await?;
    }

    // Verify model integrity
    if !manager.verify_model(model)? {
        println!("Model verification failed, redownloading...");
        manager.download_model(model).await?;
    }

    Ok(())
}
```

## Model Cache Location

Models are stored in one of the following locations, in order of precedence:

1. The directory specified by the `PREFRONTAL_CACHE` environment variable, if set
2. The platform-specific cache directory:
   - Linux: `~/.cache/prefrontal/`
   - macOS: `~/Library/Caches/prefrontal/`
   - Windows: `%LOCALAPPDATA%\prefrontal\Cache\`
3. A `.prefrontal` directory in the current working directory

You can override the default location by:

```bash
# Set a custom cache directory
export PREFRONTAL_CACHE=/path/to/your/cache

# Or when running your application
PREFRONTAL_CACHE=/path/to/your/cache cargo run
```

## Class Definitions

Classes (departments or routing destinations) are defined with labels, descriptions, and optional examples:

```rust
// With examples for few-shot classification
let class = ClassDefinition::new(
    "technical_support",
    "Technical issues and product troubleshooting"
).with_examples(vec![
    "integration problems",
    "api errors"
]);

// Without examples for zero-shot classification
let class = ClassDefinition::new(
    "billing",
    "Payment and subscription related inquiries"
);

// Get routing configuration
let info = classifier.info();
println!("Number of departments: {}", info.num_classes);
```

Each class requires:

- A unique label that identifies the group
- A description that explains the category
- Optional examples to help train the classifier

## Error Handling

The library provides detailed error types:

```rust
pub enum ClassifierError {
    TokenizerError(String),   // Tokenizer-related errors
    ModelError(String),       // ONNX model errors
    BuildError(String),       // Construction errors
    PredictionError(String),  // Prediction-time errors
    ValidationError(String),  // Input validation errors
}
```

## Logging

The library uses the `log` crate for logging. Enable debug logging for detailed information:

```rust
use prefrontal::init_logger;

fn main() {
    init_logger();  // Initialize with default configuration
    // Or configure env_logger directly for more control
}
```

## Thread Safety

The classifier is thread-safe and can be shared across threads using `Arc`:

```rust
use std::sync::Arc;
use std::thread;

let classifier = Arc::new(classifier);

let mut handles = vec![];
for _ in 0..3 {
    let classifier = Arc::clone(&classifier);
    handles.push(thread::spawn(move || {
        classifier.predict("test text").unwrap();
    }));
}

for handle in handles {
    handle.join().unwrap();
}
```

## Performance

Benchmarks run on MacBook Pro M1, 16GB RAM:

### Text Processing

| Operation    | Text Length         | Time      |
| ------------ | ------------------- | --------- |
| Tokenization | Short (< 50 chars)  | ~23.2 µs  |
| Tokenization | Medium (~100 chars) | ~51.5 µs  |
| Tokenization | Long (~200 chars)   | ~107.0 µs |

### Classification

| Operation      | Scenario     | Time     |
| -------------- | ------------ | -------- |
| Embedding      | Single text  | ~11.1 ms |
| Classification | Single class | ~11.1 ms |
| Classification | 10 classes   | ~11.1 ms |
| Build          | 10 classes   | ~450 ms  |

Key Performance Characteristics:

- Sub-millisecond tokenization
- Consistent classification time regardless of number of classes
- Thread-safe for concurrent processing
- Optimized for real-time routing decisions

### Memory Usage

- Model size: ~85MB (MiniLM)
- Runtime memory: ~100MB per instance
- Shared memory when using multiple instances (thread-safe)

Run the benchmarks yourself:

```bash
cargo bench
```

## Contributing

### Developer Prerequisites

If you want to contribute to Prefrontal or build it from source, you'll need:

1. **Build Dependencies** (as listed in System Requirements)
2. **Rust Toolchain** (latest stable version)

### Development Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/prefrontal.git
cd prefrontal
```

2. Install dependencies as described in System Requirements

3. Build and test:

```bash
cargo build
cargo test
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run benchmarks
cargo bench
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
