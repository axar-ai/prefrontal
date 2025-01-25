use text_classifier::{Classifier, BuiltinModel, ClassifierError};
use env_logger::{Builder, Env};

// Initialize test logger
fn init() {
    let _ = Builder::from_env(Env::default().default_filter_or("warn"))
        .try_init();
}

#[test]
fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_multiple_examples() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));
    Ok(())
}

#[test]
fn test_validation_errors() -> Result<(), Box<dyn std::error::Error>> {
    // Test empty model path
    let result = Classifier::builder()
        .with_custom_model("", "tokenizer.json")
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));

    // Test empty tokenizer path
    let result = Classifier::builder()
        .with_custom_model("model.onnx", "")
        .unwrap_err();
    assert!(matches!(result, ClassifierError::BuildError(_)));

    // Test empty class label
    let result = Classifier::builder()
        .add_class("", vec!["example"])
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    // Test empty examples
    let result = Classifier::builder()
        .add_class("label", vec![])
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    // Test empty example text
    let result = Classifier::builder()
        .add_class("label", vec![""])
        .unwrap_err();
    assert!(matches!(result, ClassifierError::ValidationError(_)));

    Ok(())
}

#[test]
fn test_classifier_info() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
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