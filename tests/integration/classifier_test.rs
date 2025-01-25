use text_classifier::{Classifier, BuiltinModel};
use std::path::PathBuf;

fn get_test_paths() -> (PathBuf, PathBuf) {
    let model_path = PathBuf::from("models/onnx-minilm/model.onnx");
    let tokenizer_path = PathBuf::from("models/onnx-minilm/tokenizer.json");
    (model_path, tokenizer_path)
}

#[test]
fn test_end_to_end_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("sports", vec!["football game", "basketball match"])
        .build()?;

    let (class, scores) = classifier.predict("soccer match")?;
    
    assert_eq!(class, "sports");
    assert!(scores.contains_key("sports"));
    assert!(scores["sports"] > 0.5);
    Ok(())
}

#[test]
fn test_custom_model_paths() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tokenizer_path) = get_test_paths();
    let classifier = Classifier::builder()
        .with_custom_model(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap()
        )
        .add_class("sports", vec!["football game", "basketball match"])
        .build()?;

    let (class, scores) = classifier.predict("soccer match")?;
    
    assert_eq!(class, "sports");
    assert!(scores.contains_key("sports"));
    assert!(scores["sports"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_classes_custom_paths() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tokenizer_path) = get_test_paths();
    let classifier = Classifier::builder()
        .with_custom_model(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap()
        )
        .add_class("sports", vec!["football game", "basketball match"])
        .add_class("tech", vec!["computer program", "software code"])
        .build()?;

    let (class_sports, scores_sports) = classifier.predict("soccer match")?;
    let (class_tech, scores_tech) = classifier.predict("python programming")?;

    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "tech");
    assert!(scores_sports["sports"] > scores_sports["tech"]);
    assert!(scores_tech["tech"] > scores_tech["sports"]);
    Ok(())
}

#[test]
fn test_validation_errors() {
    // Test custom model path validation
    assert!(Classifier::with_custom("", "tokenizer.json").is_err());
    assert!(Classifier::with_custom("model.onnx", "").is_err());
    assert!(Classifier::with_custom("nonexistent.onnx", "nonexistent.json").is_err());
    
    // Test with built-in model
    let mut classifier = Classifier::with_builtin(BuiltinModel::MiniLM)
        .expect("Failed to create classifier");
    
    assert!(classifier.add_class("", vec!["example"]).is_err());
    assert!(classifier.add_class("label", vec![]).is_err());
    assert!(classifier.add_class("label", vec![""]).is_err());
}

#[test]
fn test_builtin_model_characteristics() {
    let model = BuiltinModel::MiniLM;
    let chars = model.characteristics();
    assert_eq!(chars.embedding_size, 384);
    assert_eq!(chars.max_sequence_length, 256);
    assert_eq!(chars.model_size_mb, 85);
}

#[test]
fn test_same_results_builtin_and_custom() -> Result<(), Box<dyn std::error::Error>> {
    // Create two classifiers - one with built-in model and one with custom paths
    let builtin = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("sports", vec!["football game", "basketball match"])
        .build()?;

    let (model_path, tokenizer_path) = get_test_paths();
    let custom = Classifier::builder()
        .with_custom_model(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap()
        )
        .add_class("sports", vec!["football game", "basketball match"])
        .build()?;

    // Make predictions with both
    let (class1, scores1) = builtin.predict("soccer match")?;
    let (class2, scores2) = custom.predict("soccer match")?;

    // Verify they give same results
    assert_eq!(class1, class2);
    assert_eq!(scores1["sports"], scores2["sports"]);
    Ok(())
} 