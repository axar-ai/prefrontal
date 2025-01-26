# Prefrontal

A blazing fast text classifier for real-time agent routing, built in Rust. Prefrontal provides a simple yet powerful interface for routing conversations to the right workflow or agent based on text content, powered by transformer-based models.

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

The descriptions are useful for:

- Documentation of routing logic
- Understanding the classifier's capabilities
- Future extensibility for zero-shot classification

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

## Performance Considerations

- Optimized for real-time routing decisions
- Models are loaded once during initialization
- Embeddings are computed efficiently using ONNX Runtime
- Predictions use optimized vector operations
- Memory usage depends on the model size and number of routing destinations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
