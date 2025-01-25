use text_classifier::{Classifier, BuiltinModel};

#[test]
fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::with_builtin(BuiltinModel::MiniLM)?;

    classifier.add_class("test", vec!["test example"])?;
    classifier.build()?;
    
    let (class, scores) = classifier.predict("test example")?;
    assert_eq!(class, "test");
    assert!(scores["test"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_examples() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::with_builtin(BuiltinModel::MiniLM)?;

    classifier.add_class("test", vec!["example one", "example two"])?;
    classifier.build()?;

    let (class, scores) = classifier.predict("test example")?;
    assert_eq!(class, "test");
    assert!(scores["test"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::with_builtin(BuiltinModel::MiniLM)?;

    classifier.add_class("sports", vec!["football game", "basketball match"])?;
    classifier.add_class("tech", vec!["computer program", "software code"])?;
    classifier.build()?;

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