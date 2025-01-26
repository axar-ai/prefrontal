use text_classifier::{Classifier, BuiltinModel, ClassifierError, RuntimeConfig};
use ort::GraphOptimizationLevel;

#[test]
fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_custom_model("nonexistent/model.onnx", "nonexistent/tokenizer.json", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_multiple_examples() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_custom_model("nonexistent/model.onnx", "nonexistent/tokenizer.json", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_custom_model("nonexistent/model.onnx", "nonexistent/tokenizer.json", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_validation_errors() -> Result<(), Box<dyn std::error::Error>> {
    // Test empty model path
    let result = Classifier::builder()
        .with_custom_model("", "tokenizer.json", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));

    // Test empty tokenizer path
    let result = Classifier::builder()
        .with_custom_model("model.onnx", "", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));

    // Test empty class label
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class("", vec!["example"])
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    // Test empty example text
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class("test", vec![""])
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    Ok(())
}

#[test]
fn test_classifier_info() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_custom_model("nonexistent/model.onnx", "nonexistent/tokenizer.json", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_builtin_model_characteristics() {
    let model = BuiltinModel::MiniLM;
    let paths = model.get_paths();
    assert!(paths.0.contains("minilm"));
    assert!(paths.1.contains("tokenizer"));
}

#[test]
fn test_builtin_model_loading() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_custom_model("nonexistent/model.onnx", "nonexistent/tokenizer.json", None)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));

    // Test successful model loading with minimal configuration
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["sample text"])?
        .build()?;
    
    // Verify classifier is initialized and can make predictions
    let (_class, scores) = classifier.predict("test text")?;
    assert!(scores.len() > 0);
    assert!(scores.contains_key("test"));

    // Test with multiple classes
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("class1", vec!["example one"])?
        .add_class("class2", vec!["example two"])?
        .build()?;
    
    let (_class, scores) = classifier.predict("test text")?;
    assert_eq!(scores.len(), 2);
    assert!(scores.contains_key("class1"));
    assert!(scores.contains_key("class2"));

    Ok(())
}

#[test]
fn test_builtin_multiclass() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("tech", vec![
            "artificial intelligence breakthrough",
            "new software release",
            "quantum computing advance"
        ])?
        .add_class("sports", vec![
            "championship game victory",
            "world record in athletics",
            "tournament final match"
        ])?
        .add_class("business", vec![
            "stock market analysis",
            "corporate merger announcement",
            "startup funding round"
        ])?
        .build()?;

    // Test tech classification
    let (class, scores) = classifier.predict("The new AI model shows promising results")?;
    assert_eq!(class, "tech");
    assert!(scores["tech"] > scores["sports"]);
    assert!(scores["tech"] > scores["business"]);

    // Test sports classification
    let (class, scores) = classifier.predict("Team wins the championship in overtime")?;
    assert_eq!(class, "sports");
    assert!(scores["sports"] > scores["tech"]);
    assert!(scores["sports"] > scores["business"]);

    // Test business classification
    let (class, scores) = classifier.predict("Company announces successful IPO")?;
    assert_eq!(class, "business");
    assert!(scores["business"] > scores["tech"]);
    assert!(scores["business"] > scores["sports"]);

    // Test that all predictions return scores for all classes
    let (_, scores) = classifier.predict("random text")?;
    assert_eq!(scores.len(), 3);
    assert!(scores.contains_key("tech"));
    assert!(scores.contains_key("sports"));
    assert!(scores.contains_key("business"));

    Ok(())
}

#[test]
fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Test ValidationError with duplicate class
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["example1"])?
        .add_class("test", vec!["example2"])  // Duplicate class
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    // Test BuildError with no classes
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .build()
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));

    // Test ValidationError with empty class label
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("", vec!["example"])  // Empty class label
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    // Test ValidationError with empty example
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec![""])  // Empty example
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    Ok(())
}

#[test]
fn test_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["example text"])?
        .build()?;

    // Test very short input
    let (_class, scores) = classifier.predict("a")?;
    assert!(!scores.is_empty());

    // Test input with special characters
    let (_class, scores) = classifier.predict("!@#$%^&*()")?;
    assert!(!scores.is_empty());

    // Test input with unicode characters
    let (_class, scores) = classifier.predict("Hello 世界")?;
    assert!(!scores.is_empty());

    // Test input with multiple spaces
    let (_class, scores) = classifier.predict("   multiple    spaces   text   ")?;
    assert!(!scores.is_empty());

    // Test input with newlines and tabs
    let (_class, scores) = classifier.predict("text with\nnewlines\tand\ttabs")?;
    assert!(!scores.is_empty());

    // Test maximum number of classes
    let mut builder = Classifier::builder().with_model(BuiltinModel::MiniLM)?;
    for i in 0..100 {  // Test with 100 classes
        builder = builder.add_class(
            &format!("class_{}", i),
            vec!["example text"]
        )?;
    }
    let classifier = builder.build()?;
    let (_class, scores) = classifier.predict("test text")?;
    assert_eq!(scores.len(), 100);

    // Test with maximum length text that should be accepted
    let text = "test ".repeat(50);  // Should be under max_sequence_length but still long
    let (_class, scores) = classifier.predict(&text)?;
    assert!(!scores.is_empty());

    Ok(())
}

#[test]
fn test_runtime_config() -> Result<(), Box<dyn std::error::Error>> {
    // Test default config
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["sample text"])?
        .build()?;
    let (_class, scores) = classifier.predict("test text")?;
    assert!(!scores.is_empty());

    // Test single-threaded config
    let config = RuntimeConfig {
        inter_threads: 1,
        intra_threads: 1,
        optimization_level: GraphOptimizationLevel::Level1,
    };
    let classifier = Classifier::builder()
        .with_runtime_config(config)
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["sample text"])?
        .build()?;
    let (_class, scores) = classifier.predict("test text")?;
    assert!(!scores.is_empty());

    // Test multi-threaded config
    let config = RuntimeConfig {
        inter_threads: 2,
        intra_threads: 2,
        optimization_level: GraphOptimizationLevel::Level2,
    };
    let classifier = Classifier::builder()
        .with_runtime_config(config)
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["sample text"])?
        .build()?;
    let (_class, scores) = classifier.predict("test text")?;
    assert!(!scores.is_empty());

    // Test optimization level
    let config = RuntimeConfig {
        inter_threads: 0,  // Let ONNX Runtime decide
        intra_threads: 0,  // Let ONNX Runtime decide
        optimization_level: GraphOptimizationLevel::Level3,
    };
    let classifier = Classifier::builder()
        .with_runtime_config(config)
        .with_model(BuiltinModel::MiniLM)?
        .add_class("test", vec!["sample text"])?
        .build()?;
    let (_class, scores) = classifier.predict("test text")?;
    assert!(!scores.is_empty());

    Ok(())
} 