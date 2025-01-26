# Prefrontal

A blazing fast text classifier for real-time agent routing, built in Rust. Prefrontal provides a simple yet powerful interface for routing conversations to the right agent or department based on text content, powered by transformer-based models.

## System Requirements

To use Prefrontal in your project, you need:

1. **ONNX Runtime** (v1.16.3 or later)

   - Linux: `sudo apt-get install libonnxruntime`
   - macOS: `brew install onnxruntime`
   - Windows: Download from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)

2. **System Dependencies**
   - Linux: `sudo apt-get install cmake pkg-config libssl-dev`
   - macOS: `brew install cmake pkg-config openssl`
   - Windows: Install CMake and OpenSSL

## Features

- ⚡️ Blazing fast (~10ms) classification for real-time routing
- 🎯 Optimized for agent routing and conversation classification
- 🚀 Easy-to-use builder pattern interface
- 🔧 Support for both built-in and custom ONNX models
- 📊 Multiple class classification with confidence scores
- 🛠️ Built-in model management (MiniLM included)
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
use prefrontal::{Classifier, BuiltinModel, ClassDefinition};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the classifier with built-in MiniLM model
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
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

    // Get classifier information
    let info = classifier.info();
    println!("Number of departments: {}", info.num_classes);
    println!("Department labels: {:?}", info.class_labels);
    println!("Department descriptions: {:?}", info.class_descriptions);

    Ok(())
}
```

## Built-in Models

### MiniLM

- Embedding size: 384
- Max sequence length: 256
- Model size: ~85MB
- Good balance of speed and accuracy

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

## Class Definitions

Departments or routing destinations are defined with labels, descriptions, and optional examples:

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
println!("Department labels: {:?}", info.class_labels);
println!("Department descriptions: {:?}", info.class_descriptions);
```

Each routing destination requires:

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
let classifier_clone = Arc::clone(&classifier);

thread::spawn(move || {
    classifier_clone.predict("test text").unwrap();
});
```

## Performance

Benchmarks run on MacBook Pro M1, 16GB RAM:

### Text Processing

| Operation    | Text Length         | Time     |
| ------------ | ------------------- | -------- |
| Tokenization | Short (< 50 chars)  | ~22.5 µs |
| Tokenization | Medium (~100 chars) | ~51.2 µs |
| Tokenization | Long (~200 chars)   | ~105 µs  |

### Classification

| Operation      | Scenario     | Time     |
| -------------- | ------------ | -------- |
| Embedding      | Single text  | ~11.5 ms |
| Classification | Single class | ~11.1 ms |
| Classification | 10 classes   | ~11.2 ms |
| Build          | 10 classes   | ~478 ms  |

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

If you want to contribute to Prefrontal or build it from source, you'll need additional tools:

1. **Git LFS** (required for working with model files)
   - Linux: `sudo apt-get install git-lfs`
   - macOS: `brew install git-lfs`
   - Windows: Download from [Git LFS](https://git-lfs.com)

### Development Setup

1. Clone the repository with Git LFS:

```bash
git lfs install
git clone https://github.com/yourusername/prefrontal.git
cd prefrontal
git lfs pull
```

2. Install dependencies as described in System Requirements

3. Build and test:

```bash
cargo build
cargo test
```

### Model Files

The repository includes pre-trained models in the `models/` directory:

- Only `.onnx` and `tokenizer.json` files are tracked
- Large model files are handled by Git LFS
- You can add custom models in any subdirectory under `models/`

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
